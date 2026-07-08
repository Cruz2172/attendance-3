"""
FaceAttend - Facial Recognition Attendance System
Complete version with Student Self-Enrollment, Teacher Approval, Student Search,
and Full Student Dashboard Functionality

UPDATED: Single teacher invite link for all courses

DEBUGGED VERSION -- fixes applied on top of the version you uploaded:

  1. CRASH BUG in generate_student_invite(): `invite` was only assigned
     inside the `else` branch (when creating a brand-new InviteCode). The
     `if existing_invite and existing_invite.is_valid():` branch only set
     `invite_code`, never `invite` -- but the render_template() call at the
     bottom of the function referenced `invite=invite` unconditionally.
     That meant the route worked the very first time a teacher generated a
     link (no existing invite yet, so the `else` branch ran), but crashed
     with `UnboundLocalError: local variable 'invite' referenced before
     assignment` every time after that, since revisiting the page to reuse
     an already-valid invite takes the `if` branch. Fixed by assigning
     `invite = existing_invite` in that branch too. (Already present in
     the uploaded version -- kept as-is.)

  2. VARIABLE SHADOWING BUG in mark_attendance(): the loop
     `for session in sessions:` shadowed the imported Flask `session`
     object for the rest of that function's scope. Nothing after the loop
     currently reads `session['user_id']` again, so it happened not to
     crash today -- but it's a landmine: any future edit that adds a
     `session[...]` reference after that loop would silently try to read
     it off an AttendanceSession model instance instead of the real Flask
     session, causing a confusing crash. Fixed by renaming the loop
     variable to `att_session`.
"""

import os
import json
import base64
import secrets
import traceback
from datetime import datetime, timedelta
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import RequestEntityTooLarge
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

import cv2
import numpy as np
import face_recognition

app = Flask(__name__)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)

# Session configuration
app.config['SESSION_COOKIE_SECURE'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faceattend.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# CSRF protection - disabled for development
app.config['WTF_CSRF_ENABLED'] = False
app.config['WTF_CSRF_CHECK_DEFAULT'] = False
app.config['WTF_CSRF_SECRET_KEY'] = secrets.token_hex(32)

csrf = CSRFProtect(app)

CORS(app)
db = SQLAlchemy(app)

FACES_DIR = 'uploads/faces'
os.makedirs(FACES_DIR, exist_ok=True)

# Face matching thresholds
CONFIDENCE_THRESHOLD = 0.5
DUPLICATE_THRESHOLD = 0.5


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='student')
    is_approved = db.Column(db.Boolean, default=False)
    invite_code = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    courses_teaching = db.relationship('Course', backref='teacher', lazy=True, foreign_keys='Course.teacher_id')
    enrollments = db.relationship('CourseEnrollment', backref='student', lazy=True, foreign_keys='CourseEnrollment.student_id')
    enrollment_requests = db.relationship('CourseEnrollmentRequest', backref='student', lazy=True, foreign_keys='CourseEnrollmentRequest.student_id')
    attendance_records = db.relationship('CourseAttendanceRecord', backref='student', lazy=True, foreign_keys='CourseAttendanceRecord.student_id')
    student_profile = db.relationship('Student', backref='user', lazy=True, uselist=False, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'fullname': self.fullname,
            'email': self.email,
            'role': self.role,
            'is_approved': self.is_approved,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Student(db.Model):
    __tablename__ = 'students'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), unique=True, nullable=False)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    fullname = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    course = db.Column(db.String(120), nullable=False)
    level = db.Column(db.String(20), nullable=True)
    face_encoding = db.Column(db.Text, nullable=False)
    face_image = db.Column(db.String(256))
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.Index('idx_student_student_id', 'student_id'),
        db.Index('idx_student_email', 'email'),
        db.Index('idx_student_user_id', 'user_id'),
    )

    def get_encoding(self):
        try:
            return np.array(json.loads(self.face_encoding))
        except Exception as e:
            app.logger.error(f"Error loading encoding for student {self.id}: {str(e)}")
            return np.array([])

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'fullname': self.fullname,
            'email': self.email,
            'course': self.course,
            'level': self.level,
            'face_image': self.face_image,
            'enrolled_at': self.enrolled_at.isoformat() if self.enrolled_at else None,
        }


class AttendanceRecord(db.Model):
    __tablename__ = 'attendance'
    __table_args__ = (
        db.UniqueConstraint('student_id', 'date', name='unique_daily_attendance'),
    )
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), db.ForeignKey('students.student_id'), nullable=False)
    student_name = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(10), nullable=False)
    time = db.Column(db.String(8), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'student_id': self.student_id,
            'student_name': self.student_name,
            'date': self.date,
            'time': self.time,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class InviteCode(db.Model):
    __tablename__ = 'invite_codes'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False)
    used = db.Column(db.Boolean, default=False)
    used_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True)
    max_uses = db.Column(db.Integer, default=1)
    uses_count = db.Column(db.Integer, default=0)
    is_teacher_invite = db.Column(db.Boolean, default=False)

    def is_valid(self):
        if self.used and self.max_uses != -1:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        if self.max_uses != -1 and self.uses_count >= self.max_uses:
            return False
        return True


class Course(db.Model):
    __tablename__ = 'courses'

    id = db.Column(db.Integer, primary_key=True)
    course_code = db.Column(db.String(20), unique=True, nullable=False)
    course_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    schedule = db.Column(db.String(100))
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    enrollments = db.relationship('CourseEnrollment', backref='course', lazy=True, cascade='all, delete-orphan')
    sessions = db.relationship('AttendanceSession', backref='course', lazy=True, cascade='all, delete-orphan')
    enrollment_requests = db.relationship('CourseEnrollmentRequest', backref='course', lazy=True, cascade='all, delete-orphan')
    enrollment_tokens = db.relationship('CourseEnrollmentToken', backref='course', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'course_code': self.course_code,
            'course_name': self.course_name,
            'description': self.description,
            'schedule': self.schedule,
            'teacher_id': self.teacher_id,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def get_enrolled_count(self):
        try:
            return CourseEnrollment.query.filter_by(course_id=self.id, status='active').count()
        except Exception:
            return 0

    def get_pending_requests_count(self):
        try:
            return CourseEnrollmentRequest.query.filter_by(
                course_id=self.id,
                status='pending'
            ).count()
        except Exception:
            return 0

    def get_teacher_name(self):
        try:
            teacher = User.query.get(self.teacher_id)
            return teacher.fullname if teacher else "Unknown Teacher"
        except Exception:
            return "Unknown Teacher"


class CourseEnrollment(db.Model):
    __tablename__ = 'course_enrollments'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='active')

    __table_args__ = (
        db.UniqueConstraint('student_id', 'course_id', name='unique_course_enrollment'),
        db.Index('idx_enrollment_student', 'student_id'),
        db.Index('idx_enrollment_course', 'course_id'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'course_id': self.course_id,
            'enrolled_at': self.enrolled_at.isoformat() if self.enrolled_at else None,
            'status': self.status
        }


class CourseEnrollmentRequest(db.Model):
    __tablename__ = 'course_enrollment_requests'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')
    requested_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    remarks = db.Column(db.String(200), nullable=True)

    __table_args__ = (db.UniqueConstraint('student_id', 'course_id', name='unique_enrollment_request'),)

    def to_dict(self):
        student = User.query.get(self.student_id)
        reviewer = User.query.get(self.reviewed_by) if self.reviewed_by else None

        return {
            'id': self.id,
            'student_id': self.student_id,
            'student_name': student.fullname if student else 'Unknown',
            'student_email': student.email if student else 'Unknown',
            'course_id': self.course_id,
            'status': self.status,
            'requested_at': self.requested_at.isoformat() if self.requested_at else None,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'reviewed_by': self.reviewed_by,
            'reviewer_name': reviewer.fullname if reviewer else None,
            'remarks': self.remarks
        }


class AttendanceSession(db.Model):
    __tablename__ = 'attendance_sessions'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    session_date = db.Column(db.Date, nullable=False, default=lambda: datetime.utcnow().date())
    session_time = db.Column(db.Time, nullable=False, default=lambda: datetime.utcnow().time())
    topic = db.Column(db.String(200))
    status = db.Column(db.String(20), default='active')
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    session_token = db.Column(db.String(50), unique=True, nullable=False)

    records = db.relationship('CourseAttendanceRecord', backref='session', lazy=True, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'course_id': self.course_id,
            'session_date': self.session_date.isoformat() if self.session_date else None,
            'session_time': self.session_time.strftime('%H:%M:%S') if self.session_time else None,
            'topic': self.topic,
            'status': self.status,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'session_token': self.session_token
        }

    def get_present_count(self):
        try:
            return CourseAttendanceRecord.query.filter_by(session_id=self.id).count()
        except Exception:
            return 0

    def get_total_enrolled(self):
        try:
            return CourseEnrollment.query.filter_by(course_id=self.course_id).count()
        except Exception:
            return 0


class CourseAttendanceRecord(db.Model):
    __tablename__ = 'course_attendance_records'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_sessions.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    marked_at = db.Column(db.DateTime, default=datetime.utcnow)
    face_confidence = db.Column(db.Float)
    status = db.Column(db.String(20), default='present')
    remarks = db.Column(db.String(200))

    __table_args__ = (db.UniqueConstraint('session_id', 'student_id', name='unique_course_attendance'),)

    def to_dict(self):
        student = User.query.get(self.student_id)
        return {
            'id': self.id,
            'session_id': self.session_id,
            'student_id': self.student_id,
            'student_name': student.fullname if student else 'Unknown',
            'marked_at': self.marked_at.isoformat() if self.marked_at else None,
            'face_confidence': self.face_confidence,
            'status': self.status,
            'remarks': self.remarks
        }


class CourseEnrollmentToken(db.Model):
    __tablename__ = 'course_enrollment_tokens'

    id = db.Column(db.Integer, primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=False)
    token_code = db.Column(db.String(10), unique=True, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    max_uses = db.Column(db.Integer, default=50)
    used_count = db.Column(db.Integer, default=0)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def is_valid(self):
        if not self.is_active:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        if self.max_uses != -1 and self.used_count >= self.max_uses:
            return False
        return True

    def get_enrollment_link(self):
        return f"/courses/enroll/{self.token_code}"

    def to_dict(self):
        return {
            'id': self.id,
            'course_id': self.course_id,
            'token_code': self.token_code,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'max_uses': self.max_uses,
            'used_count': self.used_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_active': self.is_active,
            'enrollment_link': self.get_enrollment_link()
        }


# ─────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'warning')
            return redirect(url_for('login'))

        user = User.query.get(session['user_id'])
        if not user:
            session.clear()
            flash('Your account no longer exists. Please login again.', 'warning')
            return redirect(url_for('login'))

        if not user.is_approved:
            session.clear()
            flash('Your account has been deactivated. Please contact an administrator.', 'warning')
            return redirect(url_for('login'))

        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'warning')
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or user.role != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


def teacher_or_admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first', 'warning')
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or user.role not in ('admin', 'teacher'):
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


# ─────────────────────────────────────────────
# Face encoding cache
# ─────────────────────────────────────────────

_known_encodings_cache = None
_known_students_cache = None


def get_known_faces():
    global _known_encodings_cache, _known_students_cache
    if _known_encodings_cache is None:
        try:
            students = Student.query.all()
            _known_encodings_cache = []
            _known_students_cache = []
            for student in students:
                try:
                    encoding = student.get_encoding()
                    if encoding is not None and len(encoding) > 0:
                        _known_encodings_cache.append(encoding)
                        _known_students_cache.append(student)
                except Exception as e:
                    app.logger.error(f"Error loading encoding for student {student.id}: {str(e)}")
                    continue
        except Exception as e:
            app.logger.error(f"Error in get_known_faces: {str(e)}")
            return [], []
    return _known_encodings_cache, _known_students_cache


def invalidate_face_cache():
    global _known_encodings_cache, _known_students_cache
    _known_encodings_cache = None
    _known_students_cache = None


def decode_image_from_base64(data_url):
    try:
        if not data_url:
            return None, None

        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        else:
            return None, None

        try:
            image_bytes = base64.b64decode(data_url)
        except Exception as e:
            app.logger.error(f"Base64 decode error: {str(e)}")
            return None, None

        if not image_bytes:
            return None, None

        nparr = np.frombuffer(image_bytes, np.uint8)
        if len(nparr) == 0:
            return None, None

        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)

        return img_bgr, img_rgb

    except Exception as e:
        app.logger.error(f"Error in decode_image_from_base64: {str(e)}")
        return None, None


# ─────────────────────────────────────────────
# Context Processors
# ─────────────────────────────────────────────

@app.context_processor
def inject_pending_count():
    if 'user_id' in session and session.get('user_role') == 'admin':
        try:
            pending_count = User.query.filter_by(is_approved=False).count()
            return dict(pending_count=pending_count)
        except Exception:
            return dict(pending_count=0)
    return dict(pending_count=0)


@app.context_processor
def inject_pending_enrollment_requests():
    if 'user_id' in session and session.get('user_role') == 'teacher':
        try:
            teacher_courses = Course.query.filter_by(teacher_id=session['user_id']).all()
            course_ids = [c.id for c in teacher_courses]

            if course_ids:
                pending_count = CourseEnrollmentRequest.query.filter(
                    CourseEnrollmentRequest.course_id.in_(course_ids),
                    CourseEnrollmentRequest.status == 'pending'
                ).count()
                return dict(pending_enrollment_requests=pending_count)
        except Exception:
            pass
    return dict(pending_enrollment_requests=0)


@app.context_processor
def inject_now():
    return dict(now=datetime.now())


@app.context_processor
def inject_user():
    return dict(User=User)


@app.context_processor
def inject_current_user_role():
    return dict(current_user_role=session.get('user_role'))


@app.context_processor
def inject_current_user():
    """Makes the logged-in user's record available globally in templates"""
    if 'user_id' in session:
        try:
            return dict(user=User.query.get(session['user_id']))
        except Exception:
            return dict(user=None)
    return dict(user=None)


# ─────────────────────────────────────────────
# Public routes
# ─────────────────────────────────────────────

@app.route('/')
def home():
    """Home page with system statistics"""
    try:
        total_students = Student.query.count()
        total_teachers = User.query.filter_by(role='teacher').count()
        total_courses = Course.query.count()

        today = datetime.now().date().isoformat()
        total_students_enrolled = Student.query.count()
        present_today = AttendanceRecord.query.filter_by(date=today).count()

        if total_students_enrolled > 0:
            attendance_today = round((present_today / total_students_enrolled) * 100, 2)
        else:
            attendance_today = 0

        return render_template('home.html',
                             total_students=total_students,
                             total_teachers=total_teachers,
                             total_courses=total_courses,
                             attendance_today=attendance_today,
                             now=datetime.now())

    except Exception as e:
        app.logger.error(f"Home page error: {str(e)}")
        return render_template('home.html',
                             total_students=0,
                             total_teachers=0,
                             total_courses=0,
                             attendance_today=0,
                             now=datetime.now())


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            fullname = request.form.get('fullname')
            email = request.form.get('email')
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            invite_code = request.form.get('invite_code')

            if not all([fullname, email, password, confirm_password]):
                flash('All fields are required!', 'danger')
                return redirect(url_for('register'))

            if password != confirm_password:
                flash('Passwords do not match!', 'danger')
                return redirect(url_for('register'))

            if len(password) < 6:
                flash('Password must be at least 6 characters!', 'danger')
                return redirect(url_for('register'))

            if User.query.filter_by(email=email).first():
                flash('Email already registered!', 'danger')
                return redirect(url_for('register'))

            role = 'student'
            is_approved = False
            invite_obj = None
            teacher_courses = []

            if invite_code:
                invite_obj = InviteCode.query.filter_by(code=invite_code).first()

                if not invite_obj:
                    flash('Invalid invite code! The code does not exist.', 'danger')
                    return redirect(url_for('register'))

                if not invite_obj.is_valid():
                    if invite_obj.expires_at and datetime.utcnow() > invite_obj.expires_at:
                        flash('This invite code has expired!', 'danger')
                    elif invite_obj.used and invite_obj.max_uses != -1:
                        flash('This invite code has already been used!', 'danger')
                    elif invite_obj.max_uses != -1 and invite_obj.uses_count >= invite_obj.max_uses:
                        flash('This invite code has reached its maximum uses!', 'danger')
                    else:
                        flash('This invite code is no longer valid!', 'danger')
                    return redirect(url_for('register'))

                if invite_obj.is_teacher_invite:
                    role = 'student'
                    is_approved = True

                    teacher = User.query.get(invite_obj.created_by)
                    if teacher:
                        teacher_courses = Course.query.filter_by(
                            teacher_id=teacher.id,
                            is_active=True
                        ).all()

                        if not teacher_courses:
                            flash('This teacher has no active courses at the moment.', 'warning')
                            return redirect(url_for('register'))
                    else:
                        flash('Invalid invite code! Teacher not found.', 'danger')
                        return redirect(url_for('register'))
                else:
                    role = invite_obj.role
                    is_approved = True if role in ('admin', 'teacher') else False

                invite_obj.uses_count += 1
                if invite_obj.max_uses != -1 and invite_obj.uses_count >= invite_obj.max_uses:
                    invite_obj.used = True

            new_user = User(
                fullname=fullname,
                email=email,
                password=generate_password_hash(password),
                role=role,
                is_approved=is_approved,
                invite_code=invite_code if invite_code else None,
                created_at=datetime.utcnow()
            )
            db.session.add(new_user)
            db.session.commit()

            if invite_obj and invite_obj.is_teacher_invite:
                invite_obj.used_by = new_user.id
                db.session.commit()

            if teacher_courses:
                enrolled_count = 0
                for course in teacher_courses:
                    try:
                        existing_enrollment = CourseEnrollment.query.filter_by(
                            student_id=new_user.id,
                            course_id=course.id
                        ).first()

                        if not existing_enrollment:
                            enrollment = CourseEnrollment(
                                student_id=new_user.id,
                                course_id=course.id,
                                enrolled_at=datetime.utcnow(),
                                status='active'
                            )
                            db.session.add(enrollment)
                            enrolled_count += 1
                    except Exception as e:
                        app.logger.error(f"Error enrolling student in course {course.id}: {str(e)}")

                db.session.commit()

                if enrolled_count > 0:
                    flash(f'✅ Registration successful! You have been automatically enrolled in {enrolled_count} course(s).', 'success')
                else:
                    flash('✅ Registration successful!', 'success')

            elif is_approved:
                flash(f'✅ Registration successful! You can login as {role}.', 'success')
            else:
                flash('✅ Registration successful! Your account is pending admin approval.', 'info')

            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Registration error: {str(e)}")
            app.logger.error(traceback.format_exc())
            flash(f'Registration error: {str(e)}', 'danger')
            return redirect(url_for('register'))

    invite_code = request.args.get('code', '')
    return render_template('register.html', invite_code=invite_code)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')

            if not email or not password:
                flash('Email and password are required!', 'danger')
                return redirect(url_for('login'))

            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                if not user.is_approved:
                    flash('Your account is pending approval. Please contact an administrator.', 'warning')
                    return redirect(url_for('login'))

                session.permanent = True
                session['user_id'] = user.id
                session['user_name'] = user.fullname
                session['user_email'] = user.email
                session['user_role'] = user.role

                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))

            flash('Invalid email or password!', 'danger')
            return redirect(url_for('login'))

        except Exception as e:
            app.logger.error(f"Login error: {str(e)}")
            flash(f'Login error: {str(e)}', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


# ─────────────────────────────────────────────
# Dashboard Routes
# ─────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard - redirects to role-specific dashboards"""
    try:
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))

        user = User.query.get(session['user_id'])
        if not user:
            session.clear()
            flash('User not found. Please login again.', 'danger')
            return redirect(url_for('login'))

        if not user.is_approved:
            session.clear()
            flash('Your account is pending approval.', 'warning')
            return redirect(url_for('login'))

        if user.role == 'student':
            return redirect(url_for('student_dashboard'))
        elif user.role == 'teacher':
            return redirect(url_for('teacher_dashboard'))
        elif user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid user role.', 'danger')
            return redirect(url_for('home'))

    except Exception as e:
        app.logger.error(f"Dashboard error: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash('Error loading dashboard.', 'danger')
        return redirect(url_for('home'))


@app.route('/student/dashboard')
@login_required
def student_dashboard():
    """Student dashboard - Simplified for registration and face enrollment"""
    try:
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))

        user = User.query.get(session['user_id'])
        if not user:
            session.clear()
            flash('User not found. Please login again.', 'danger')
            return redirect(url_for('login'))

        if not user.is_approved:
            session.clear()
            flash('Your account is pending approval.', 'warning')
            return redirect(url_for('login'))

        if user.role != 'student':
            flash('Access denied. Student only area.', 'danger')
            return redirect(url_for('dashboard'))

        student_profile = Student.query.filter_by(user_id=user.id).first()
        has_face_enrolled = student_profile is not None

        invite_code = request.args.get('code', '')

        return render_template('student/dashboard.html',
                             user=user,
                             has_face_enrolled=has_face_enrolled,
                             invite_code=invite_code,
                             now=datetime.now())

    except Exception as e:
        app.logger.error(f"Student dashboard error: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash('Error loading student dashboard. Please try again.', 'danger')
        return redirect(url_for('home'))


@app.route('/student/courses')
@login_required
def student_courses():
    """List all courses the student is enrolled in"""
    try:
        if session.get('user_role') != 'student':
            flash('Access denied.', 'danger')
            return redirect(url_for('dashboard'))

        user = User.query.get(session['user_id'])

        enrollments = CourseEnrollment.query.filter_by(
            student_id=user.id,
            status='active'
        ).all()

        courses = []
        course_stats = {}

        for enrollment in enrollments:
            course = Course.query.get(enrollment.course_id)
            if course:
                courses.append(course)

                course_sessions = AttendanceSession.query.filter_by(course_id=course.id).all()
                attended = 0
                for att_session in course_sessions:
                    record = CourseAttendanceRecord.query.filter_by(
                        session_id=att_session.id,
                        student_id=user.id
                    ).first()
                    if record:
                        attended += 1

                total = len(course_sessions)
                percentage = round((attended / total) * 100, 2) if total > 0 else 0

                course_stats[course.id] = {
                    'total_sessions': total,
                    'attended': attended,
                    'percentage': percentage
                }

        return render_template('student/my_courses.html',
                             courses=courses,
                             course_stats=course_stats)

    except Exception as e:
        app.logger.error(f"Student courses error: {str(e)}")
        flash('Error loading courses.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/student/course/<int:course_id>')
@login_required
def student_course_detail(course_id):
    """View detailed course information for a student"""
    try:
        if session.get('user_role') != 'student':
            flash('Access denied.', 'danger')
            return redirect(url_for('dashboard'))

        course = Course.query.get_or_404(course_id)
        user = User.query.get(session['user_id'])

        enrollment = CourseEnrollment.query.filter_by(
            student_id=user.id,
            course_id=course_id,
            status='active'
        ).first()

        if not enrollment:
            flash('You are not enrolled in this course.', 'danger')
            return redirect(url_for('student_courses'))

        course_sessions = AttendanceSession.query.filter_by(
            course_id=course_id
        ).order_by(AttendanceSession.session_date.desc()).all()

        attended_session_ids = []
        for att_session in course_sessions:
            record = CourseAttendanceRecord.query.filter_by(
                session_id=att_session.id,
                student_id=user.id
            ).first()
            if record:
                attended_session_ids.append(att_session.id)

        total = len(course_sessions)
        attended = len(attended_session_ids)
        percentage = round((attended / total) * 100, 2) if total > 0 else 0

        stats = {
            'total_sessions': total,
            'attended': attended,
            'percentage': percentage
        }

        return render_template('student/course_detail.html',
                             course=course,
                             sessions=course_sessions,
                             attended_session_ids=attended_session_ids,
                             stats=stats)

    except Exception as e:
        app.logger.error(f"Student course detail error: {str(e)}")
        flash('Error loading course details.', 'danger')
        return redirect(url_for('student_courses'))


@app.route('/student/attendance-history')
@login_required
def student_attendance_history():
    """View complete attendance history for a student"""
    try:
        if session.get('user_role') != 'student':
            flash('Access denied.', 'danger')
            return redirect(url_for('dashboard'))

        user = User.query.get(session['user_id'])

        course_id = request.args.get('course', type=int)
        from_date = request.args.get('from_date')
        to_date = request.args.get('to_date')

        enrollments = CourseEnrollment.query.filter_by(
            student_id=user.id,
            status='active'
        ).all()
        courses = [Course.query.get(e.course_id) for e in enrollments if Course.query.get(e.course_id)]

        query = CourseAttendanceRecord.query.filter_by(student_id=user.id)

        if course_id:
            sessions_in_course = AttendanceSession.query.filter_by(course_id=course_id).all()
            session_ids = [s.id for s in sessions_in_course]
            query = query.filter(CourseAttendanceRecord.session_id.in_(session_ids))

        if from_date:
            try:
                from_date_obj = datetime.strptime(from_date, '%Y-%m-%d')
                query = query.filter(CourseAttendanceRecord.marked_at >= from_date_obj)
            except ValueError:
                pass

        if to_date:
            try:
                to_date_obj = datetime.strptime(to_date, '%Y-%m-%d')
                to_date_obj = to_date_obj + timedelta(days=1)
                query = query.filter(CourseAttendanceRecord.marked_at <= to_date_obj)
            except ValueError:
                pass

        records = query.order_by(CourseAttendanceRecord.marked_at.desc()).all()

        all_sessions = []
        for course in courses:
            course_sessions = AttendanceSession.query.filter_by(course_id=course.id).all()
            all_sessions.extend(course_sessions)

        total_sessions = len(all_sessions)
        present_count = len(records)
        absent_count = total_sessions - present_count
        attendance_rate = round((present_count / total_sessions) * 100, 2) if total_sessions > 0 else 0

        formatted_records = []
        for record in records:
            att_session = AttendanceSession.query.get(record.session_id)
            if att_session:
                course = Course.query.get(att_session.course_id)
                formatted_records.append({
                    'course_code': course.course_code if course else 'Unknown',
                    'course_name': course.course_name if course else 'Unknown',
                    'date': att_session.session_date,
                    'time': att_session.session_time,
                    'topic': att_session.topic,
                    'marked_at': record.marked_at
                })

        return render_template('student/attendance_history.html',
                             records=formatted_records,
                             courses=courses,
                             selected_course=course_id,
                             from_date=from_date,
                             to_date=to_date,
                             total_sessions=total_sessions,
                             present_count=present_count,
                             absent_count=absent_count,
                             attendance_rate=attendance_rate)

    except Exception as e:
        app.logger.error(f"Student attendance history error: {str(e)}")
        flash('Error loading attendance history.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/student/profile', methods=['GET', 'POST'])
@login_required
def student_profile():
    """Student profile management"""
    try:
        if session.get('user_role') != 'student':
            flash('Access denied.', 'danger')
            return redirect(url_for('dashboard'))

        user = User.query.get(session['user_id'])
        student = Student.query.filter_by(user_id=user.id).first()

        if request.method == 'POST':
            fullname = request.form.get('fullname')
            email = request.form.get('email')
            student_id = request.form.get('student_id', '').strip()
            course = request.form.get('course')
            level = request.form.get('level')

            if fullname:
                user.fullname = fullname

            if email and email != user.email:
                existing = User.query.filter_by(email=email).first()
                if existing and existing.id != user.id:
                    flash('Email is already taken.', 'danger')
                    return redirect(url_for('student_profile'))
                user.email = email

            if student:
                if student_id:
                    existing = Student.query.filter_by(student_id=student_id).first()
                    if existing and existing.id != student.id:
                        flash('Student ID is already taken.', 'danger')
                        return redirect(url_for('student_profile'))
                    student.student_id = student_id
                if course:
                    student.course = course
                if level:
                    student.level = level
                if fullname:
                    student.fullname = fullname
                if email:
                    existing_email = Student.query.filter_by(email=email).first()
                    if existing_email and existing_email.id != student.id:
                        flash('Email is already used by another enrolled student.', 'danger')
                        return redirect(url_for('student_profile'))
                    student.email = email
            elif student_id:
                existing = Student.query.filter_by(student_id=student_id).first()
                if existing:
                    flash(f'Student ID "{student_id}" is already taken.', 'danger')
                    return redirect(url_for('student_profile'))

                student = Student(
                    user_id=user.id,
                    student_id=student_id,
                    fullname=fullname or user.fullname,
                    email=email or user.email,
                    course=course or '',
                    level=level or '',
                    face_encoding=json.dumps([])
                )
                db.session.add(student)

            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('student_profile'))

        return render_template('student/profile.html',
                             user=user,
                             student=student)

    except IntegrityError as e:
        db.session.rollback()
        app.logger.error(f"Student profile IntegrityError: {str(e)}")
        flash('That Student ID or email is already in use. Please choose another.', 'danger')
        return redirect(url_for('student_profile'))

    except OperationalError as e:
        db.session.rollback()
        app.logger.error(f"Student profile OperationalError: {str(e)}")
        flash('Database is busy. Please try again in a moment.', 'danger')
        return redirect(url_for('student_profile'))

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Student profile error: {str(e)}")
        flash('Error updating profile. Please try again.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/student/change-password', methods=['POST'])
@login_required
def student_change_password():
    """Change student password"""
    try:
        if session.get('user_role') != 'student':
            flash('Access denied.', 'danger')
            return redirect(url_for('dashboard'))

        user = User.query.get(session['user_id'])

        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        if not all([current_password, new_password, confirm_password]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('student_profile'))

        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return redirect(url_for('student_profile'))

        if len(new_password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('student_profile'))

        if not check_password_hash(user.password, current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('student_profile'))

        user.password = generate_password_hash(new_password)
        db.session.commit()

        flash('Password changed successfully!', 'success')
        return redirect(url_for('student_profile'))

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Student change password error: {str(e)}")
        flash(f'Error changing password: {str(e)}', 'danger')
        return redirect(url_for('student_profile'))


@app.route('/student/attendance-stats')
@login_required
def student_attendance_stats():
    """View attendance statistics with charts"""
    try:
        if session.get('user_role') != 'student':
            flash('Access denied.', 'danger')
            return redirect(url_for('dashboard'))

        user = User.query.get(session['user_id'])

        enrollments = CourseEnrollment.query.filter_by(
            student_id=user.id,
            status='active'
        ).all()
        courses = [Course.query.get(e.course_id) for e in enrollments if Course.query.get(e.course_id)]

        return render_template('student/attendance_stats.html', courses=courses)

    except Exception as e:
        app.logger.error(f"Student attendance stats error: {str(e)}")
        flash('Error loading statistics.', 'danger')
        return redirect(url_for('dashboard'))


# ─────────────────────────────────────────────
# API ROUTES FOR STUDENT
# ─────────────────────────────────────────────

@app.route('/api/student/attendance-stats')
@login_required
def api_student_attendance_stats():
    """API endpoint for student attendance statistics"""
    try:
        if session.get('user_role') != 'student':
            return jsonify({'error': 'Access denied'}), 403

        user = User.query.get(session['user_id'])
        course_id = request.args.get('course', type=int)

        enrollments = CourseEnrollment.query.filter_by(
            student_id=user.id,
            status='active'
        ).all()

        all_sessions = []
        for enrollment in enrollments:
            if course_id and enrollment.course_id != course_id:
                continue
            course_sessions = AttendanceSession.query.filter_by(
                course_id=enrollment.course_id
            ).all()
            all_sessions.extend(course_sessions)

        present_sessions = []
        for att_session in all_sessions:
            record = CourseAttendanceRecord.query.filter_by(
                session_id=att_session.id,
                student_id=user.id
            ).first()
            if record:
                present_sessions.append(att_session)

        total = len(all_sessions)
        present = len(present_sessions)
        absent = total - present
        rate = round((present / total) * 100, 2) if total > 0 else 0

        trend_data = []
        for i in range(30, -1, -1):
            date = datetime.now().date() - timedelta(days=i)
            day_present = 0
            day_total = 0

            for att_session in all_sessions:
                if att_session.session_date == date:
                    day_total += 1
                    record = CourseAttendanceRecord.query.filter_by(
                        session_id=att_session.id,
                        student_id=user.id
                    ).first()
                    if record:
                        day_present += 1

            if day_total > 0 or i == 0:
                trend_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'present': day_present,
                    'absent': day_total - day_present
                })

        monthly_data = []
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        today = datetime.now().date()

        year, month = today.year, today.month
        window = []
        for _ in range(6):
            window.append((year, month))
            month -= 1
            if month < 1:
                month = 12
                year -= 1
        window.reverse()

        for (yr, m) in window:
            month_sessions = [s for s in all_sessions if s.session_date.year == yr and s.session_date.month == m]
            month_present = 0
            for att_session in month_sessions:
                record = CourseAttendanceRecord.query.filter_by(
                    session_id=att_session.id,
                    student_id=user.id
                ).first()
                if record:
                    month_present += 1

            month_total = len(month_sessions)
            month_rate = round((month_present / month_total) * 100, 2) if month_total > 0 else 0

            if month_total > 0:
                monthly_data.append({
                    'month': f"{months[m - 1]} {yr}",
                    'total': month_total,
                    'present': month_present,
                    'absent': month_total - month_present,
                    'rate': month_rate
                })

        return jsonify({
            'total_sessions': total,
            'present': present,
            'absent': absent,
            'attendance_rate': rate,
            'trend_data': trend_data,
            'monthly_data': monthly_data
        })

    except Exception as e:
        app.logger.error(f"API student attendance stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────
# TEACHER DASHBOARD ROUTES
# ─────────────────────────────────────────────

@app.route('/teacher/dashboard')
@login_required
def teacher_dashboard():
    """Teacher dashboard - shows only students enrolled in teacher's courses"""
    try:
        if session.get('user_role') != 'teacher':
            flash('Access denied.', 'danger')
            return redirect(url_for('dashboard'))

        user = User.query.get(session['user_id'])
        if not user:
            flash('User not found.', 'danger')
            return redirect(url_for('login'))

        # Get teacher's courses
        courses = Course.query.filter_by(teacher_id=user.id).all()

        total_students = 0
        pending_requests = 0
        active_sessions = 0
        all_student_ids = []

        for course in courses:
            enrolled = course.get_enrolled_count()
            pending = course.get_pending_requests_count()
            total_students += enrolled
            pending_requests += pending

            active = AttendanceSession.query.filter_by(
                course_id=course.id,
                status='active'
            ).first()
            if active:
                active_sessions += 1

            # Get unique student IDs for this course
            enrollments = CourseEnrollment.query.filter_by(
                course_id=course.id,
                status='active'
            ).all()
            for enrollment in enrollments:
                if enrollment.student_id not in all_student_ids:
                    all_student_ids.append(enrollment.student_id)

        # Get unique students count (in case a student is in multiple courses)
        unique_students = len(all_student_ids)

        # Get recent student enrollments for this teacher's courses
        recent_students = []
        if all_student_ids:
            recent_students = User.query.filter(
                User.id.in_(all_student_ids)
            ).order_by(User.created_at.desc()).limit(10).all()

        return render_template('teacher/dashboard.html',
                             user=user,
                             courses=courses,
                             total_students=unique_students,  # Unique students only
                             total_enrollments=total_students,  # Total enrollments across all courses
                             pending_requests=pending_requests,
                             active_sessions=active_sessions,
                             recent_students=recent_students,
                             now=datetime.now())

    except Exception as e:
        app.logger.error(f"Error in teacher dashboard: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash('Error loading teacher dashboard.', 'danger')
        return redirect(url_for('dashboard'))


# ─────────────────────────────────────────────
# ADMIN DASHBOARD ROUTES
# ─────────────────────────────────────────────

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    """Admin dashboard"""
    try:
        if session.get('user_role') != 'admin':
            flash('Access denied.', 'danger')
            return redirect(url_for('dashboard'))

        user = User.query.get(session['user_id'])
        if not user:
            flash('User not found.', 'danger')
            return redirect(url_for('login'))

        total_users = User.query.count()
        total_students = Student.query.count()
        total_courses = Course.query.count()
        pending_approvals = User.query.filter_by(is_approved=False).count()

        recent_users = User.query.order_by(User.created_at.desc()).limit(10).all()

        recent_attendance = AttendanceRecord.query.order_by(
            AttendanceRecord.timestamp.desc()
        ).limit(10).all()

        return render_template('admin/dashboard.html',
                             user=user,
                             total_users=total_users,
                             total_students=total_students,
                             total_courses=total_courses,
                             pending_approvals=pending_approvals,
                             recent_users=recent_users,
                             recent_attendance=recent_attendance,
                             now=datetime.now())

    except Exception as e:
        app.logger.error(f"Error in admin dashboard: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash('Error loading admin dashboard.', 'danger')
        return redirect(url_for('dashboard'))


# ─────────────────────────────────────────────
# Student Enrollment (Face Enrollment) Route
# ─────────────────────────────────────────────

@app.route('/enroll', methods=['GET', 'POST'])
@login_required
def enroll():
    if request.method == 'POST':
        try:
            student_id = request.form.get('student_id', '').strip()
            fullname = request.form.get('fullname', '').strip()
            email = request.form.get('email', '').strip()
            course = request.form.get('course', '').strip()
            level = request.form.get('level', '').strip()
            face_data = request.form.get('face_data', '')
            from_dashboard = request.form.get('from_dashboard', 'false')
            invite_code = request.form.get('invite_code', '').strip()

            if not all([student_id, fullname, email, course, face_data]):
                flash('All fields are required!', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            if not level:
                flash('Please select your level!', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            if '@' not in email or '.' not in email:
                flash('Please enter a valid email address!', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            if len(student_id) < 3:
                flash('Student ID must be at least 3 characters!', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            if not face_data.startswith('data:image/'):
                flash('Invalid image data. Please capture a new photo.', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            existing_student = Student.query.filter_by(user_id=session['user_id']).first()
            if existing_student:
                flash(f'You already have a face enrollment as {existing_student.fullname}!', 'warning')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            existing_student_id = Student.query.filter_by(student_id=student_id).first()
            if existing_student_id:
                flash(f'Student ID "{student_id}" is already enrolled as {existing_student_id.fullname}!', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            existing_email = Student.query.filter_by(email=email).first()
            if existing_email:
                flash(f'Email "{email}" is already enrolled as {existing_email.fullname}!', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            img_bgr, img_rgb = decode_image_from_base64(face_data)
            if img_bgr is None or img_rgb is None:
                flash('Could not decode the captured image. Please retake the photo.', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            face_encodings = face_recognition.face_encodings(img_rgb)
            if len(face_encodings) == 0:
                flash('No face detected. Please retake the photo with your face clearly visible.', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            if len(face_encodings) > 1:
                flash('Multiple faces detected. Please ensure only one person is in frame.', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            face_encoding_array = face_encodings[0]

            known_encodings, known_students = get_known_faces()
            if known_encodings and len(known_encodings) > 0:
                face_distances = face_recognition.face_distance(known_encodings, face_encoding_array)
                closest_idx = int(np.argmin(face_distances))
                closest_distance = float(face_distances[closest_idx])

                if closest_distance < DUPLICATE_THRESHOLD:
                    matched_student = known_students[closest_idx]
                    flash(f'This face is already enrolled as {matched_student.fullname} ({matched_student.student_id})!', 'danger')
                    if from_dashboard == 'true':
                        return redirect(url_for('student_dashboard'))
                    return render_template('enroll.html')

            face_encoding = json.dumps(face_encoding_array.tolist())
            face_filename = f"{student_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            face_path = os.path.join(FACES_DIR, face_filename)

            os.makedirs(os.path.dirname(face_path), exist_ok=True)

            if not cv2.imwrite(face_path, img_bgr):
                flash('Failed to save face image. Please try again.', 'danger')
                if from_dashboard == 'true':
                    return redirect(url_for('student_dashboard'))
                return render_template('enroll.html')

            new_student = Student(
                user_id=session['user_id'],
                student_id=student_id,
                fullname=fullname,
                email=email,
                course=course,
                level=level,
                face_encoding=face_encoding,
                face_image=face_path,
                enrolled_at=datetime.utcnow()
            )

            db.session.add(new_student)
            db.session.commit()
            invalidate_face_cache()

            if invite_code:
                try:
                    invite = InviteCode.query.filter_by(
                        code=invite_code,
                        is_teacher_invite=True,
                        used=False
                    ).first()

                    if invite:
                        teacher = User.query.get(invite.created_by)
                        if teacher:
                            teacher_courses = Course.query.filter_by(
                                teacher_id=teacher.id,
                                is_active=True
                            ).all()

                            enrolled_count = 0
                            for course_obj in teacher_courses:
                                existing = CourseEnrollment.query.filter_by(
                                    student_id=new_student.user_id,
                                    course_id=course_obj.id
                                ).first()
                                if not existing:
                                    enrollment = CourseEnrollment(
                                        student_id=new_student.user_id,
                                        course_id=course_obj.id,
                                        enrolled_at=datetime.utcnow(),
                                        status='active'
                                    )
                                    db.session.add(enrollment)
                                    enrolled_count += 1

                            if enrolled_count > 0:
                                db.session.commit()
                                flash(f'Student "{fullname}" enrolled successfully and added to {enrolled_count} course(s)!', 'success')
                            else:
                                flash(f'Student "{fullname}" enrolled successfully!', 'success')
                        else:
                            flash(f'Student "{fullname}" enrolled successfully!', 'success')
                    else:
                        flash(f'Student "{fullname}" enrolled successfully!', 'success')
                except Exception as e:
                    app.logger.error(f"Error enrolling in courses via invite: {str(e)}")
                    flash(f'Student "{fullname}" enrolled successfully!', 'success')
            else:
                flash(f'Student "{fullname}" enrolled successfully!', 'success')

            if from_dashboard == 'true':
                return redirect(url_for('student_dashboard'))
            return redirect(url_for('dashboard'))

        except IntegrityError as e:
            db.session.rollback()
            app.logger.error(f"IntegrityError: {str(e)}")
            error_text = str(e).lower()
            if 'student_id' in error_text:
                flash('This Student ID is already registered!', 'danger')
            elif 'email' in error_text:
                flash('This email is already registered!', 'danger')
            elif 'user_id' in error_text:
                flash('You are already enrolled!', 'danger')
            else:
                flash('A database error occurred. Please try again.', 'danger')

            from_dashboard = request.form.get('from_dashboard', 'false')
            if from_dashboard == 'true':
                return redirect(url_for('student_dashboard'))
            return render_template('enroll.html')

        except OperationalError as e:
            db.session.rollback()
            app.logger.error(f"OperationalError during enrollment: {str(e)}")
            if 'database is locked' in str(e):
                flash('Database is busy. Please try again in a moment.', 'danger')
            else:
                flash('A database error occurred. Please try again.', 'danger')

            from_dashboard = request.form.get('from_dashboard', 'false')
            if from_dashboard == 'true':
                return redirect(url_for('student_dashboard'))
            return render_template('enroll.html')

        except Exception as e:
            db.session.rollback()
            app.logger.exception('Error during enrollment')
            flash(f'Error during enrollment: {str(e)}', 'danger')

            from_dashboard = request.form.get('from_dashboard', 'false')
            if from_dashboard == 'true':
                return redirect(url_for('student_dashboard'))
            return render_template('enroll.html')

    return render_template('enroll.html')


# ─────────────────────────────────────────────
# Student Course Enrollment Routes
# ─────────────────────────────────────────────

@app.route('/courses/browse')
@login_required
def browse_courses():
    try:
        if session.get('user_role') != 'student':
            flash('Only students can browse courses.', 'danger')
            return redirect(url_for('dashboard'))

        courses = Course.query.filter_by(is_active=True).all()

        enrolled_course_ids = [e.course_id for e in CourseEnrollment.query.filter_by(
            student_id=session['user_id'], status='active'
        ).all()]

        requested_course_ids = [r.course_id for r in CourseEnrollmentRequest.query.filter_by(
            student_id=session['user_id'], status='pending'
        ).all()]

        rejected_course_ids = [r.course_id for r in CourseEnrollmentRequest.query.filter_by(
            student_id=session['user_id'], status='rejected'
        ).all()]

        rejected_requests = CourseEnrollmentRequest.query.filter_by(
            student_id=session['user_id'], status='rejected'
        ).all()
        rejected_remarks = {r.course_id: r.remarks for r in rejected_requests if r.remarks}

        course_tokens = {}
        for course in courses:
            token = CourseEnrollmentToken.query.filter_by(
                course_id=course.id,
                is_active=True
            ).first()
            if token and token.is_valid():
                course_tokens[course.id] = token.token_code

        direct_enroll_course_ids = list(course_tokens.keys())

        return render_template('courses/browse_courses.html',
                             courses=courses,
                             enrolled_course_ids=enrolled_course_ids,
                             requested_course_ids=requested_course_ids,
                             rejected_course_ids=rejected_course_ids,
                             rejected_remarks=rejected_remarks,
                             course_tokens=course_tokens,
                             direct_enroll_course_ids=direct_enroll_course_ids)

    except Exception as e:
        app.logger.error(f"Error browsing courses: {str(e)}")
        flash('Error loading courses.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/courses/request-enrollment/<int:course_id>', methods=['POST'])
@login_required
def request_course_enrollment(course_id):
    try:
        if session.get('user_role') != 'student':
            flash('Only students can enroll in courses.', 'danger')
            return redirect(url_for('dashboard'))

        course = Course.query.get_or_404(course_id)

        existing_enrollment = CourseEnrollment.query.filter_by(
            student_id=session['user_id'],
            course_id=course_id
        ).first()

        if existing_enrollment:
            flash('You are already enrolled in this course!', 'warning')
            return redirect(url_for('browse_courses'))

        existing_request = CourseEnrollmentRequest.query.filter_by(
            student_id=session['user_id'],
            course_id=course_id,
            status='pending'
        ).first()

        if existing_request:
            flash('You already have a pending enrollment request for this course.', 'info')
            return redirect(url_for('browse_courses'))

        rejected_request = CourseEnrollmentRequest.query.filter_by(
            student_id=session['user_id'],
            course_id=course_id,
            status='rejected'
        ).first()

        if rejected_request:
            flash('Your previous enrollment request was rejected. Please contact the teacher.', 'warning')
            return redirect(url_for('browse_courses'))

        request_obj = CourseEnrollmentRequest(
            student_id=session['user_id'],
            course_id=course_id,
            status='pending',
            requested_at=datetime.utcnow()
        )

        db.session.add(request_obj)
        db.session.commit()

        flash(f'Enrollment request sent for {course.course_code} - {course.course_name}. Waiting for teacher approval.', 'success')

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error requesting enrollment: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')

    return redirect(url_for('browse_courses'))


# ─────────────────────────────────────────────
# Teacher Routes
# ─────────────────────────────────────────────

@app.route('/teacher/enrollment-requests')
@login_required
@teacher_or_admin_required
def manage_enrollment_requests():
    try:
        user = User.query.get(session['user_id'])

        if user.role == 'admin':
            requests = CourseEnrollmentRequest.query.order_by(
                CourseEnrollmentRequest.requested_at.desc()
            ).all()
        else:
            teacher_courses = Course.query.filter_by(teacher_id=user.id).all()
            course_ids = [c.id for c in teacher_courses]

            if course_ids:
                requests = CourseEnrollmentRequest.query.filter(
                    CourseEnrollmentRequest.course_id.in_(course_ids)
                ).order_by(
                    CourseEnrollmentRequest.requested_at.desc()
                ).all()
            else:
                requests = []

        pending_requests = [r for r in requests if r.status == 'pending']
        approved_requests = [r for r in requests if r.status == 'approved']
        rejected_requests = [r for r in requests if r.status == 'rejected']

        return render_template('teacher/enrollment_requests.html',
                             pending_requests=pending_requests,
                             approved_requests=approved_requests,
                             rejected_requests=rejected_requests,
                             total=len(requests))

    except Exception as e:
        app.logger.error(f"Error loading enrollment requests: {str(e)}")
        flash('Error loading enrollment requests.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/teacher/enrollment-requests/<int:request_id>/approve', methods=['POST'])
@login_required
@teacher_or_admin_required
def approve_enrollment_request(request_id):
    try:
        request_obj = CourseEnrollmentRequest.query.get_or_404(request_id)
        course = Course.query.get(request_obj.course_id)

        if session.get('user_role') != 'admin' and course.teacher_id != session['user_id']:
            flash('You do not have permission to approve this request.', 'danger')
            return redirect(url_for('manage_enrollment_requests'))

        if request_obj.status != 'pending':
            flash(f'This request has already been {request_obj.status}.', 'warning')
            return redirect(url_for('manage_enrollment_requests'))

        existing = CourseEnrollment.query.filter_by(
            student_id=request_obj.student_id,
            course_id=request_obj.course_id
        ).first()

        if existing:
            flash('Student is already enrolled in this course.', 'info')
            request_obj.status = 'approved'
            request_obj.reviewed_at = datetime.utcnow()
            request_obj.reviewed_by = session['user_id']
            db.session.commit()
            return redirect(url_for('manage_enrollment_requests'))

        enrollment = CourseEnrollment(
            student_id=request_obj.student_id,
            course_id=request_obj.course_id,
            enrolled_at=datetime.utcnow(),
            status='active'
        )

        request_obj.status = 'approved'
        request_obj.reviewed_at = datetime.utcnow()
        request_obj.reviewed_by = session['user_id']
        request_obj.remarks = 'Approved by teacher'

        db.session.add(enrollment)
        db.session.commit()

        student = User.query.get(request_obj.student_id)
        flash(f'Enrollment approved for {student.fullname} in {course.course_code}!', 'success')

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error approving enrollment: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')

    return redirect(url_for('manage_enrollment_requests'))


@app.route('/teacher/enrollment-requests/<int:request_id>/reject', methods=['POST'])
@login_required
@teacher_or_admin_required
def reject_enrollment_request(request_id):
    try:
        request_obj = CourseEnrollmentRequest.query.get_or_404(request_id)
        course = Course.query.get(request_obj.course_id)

        if session.get('user_role') != 'admin' and course.teacher_id != session['user_id']:
            flash('You do not have permission to reject this request.', 'danger')
            return redirect(url_for('manage_enrollment_requests'))

        if request_obj.status != 'pending':
            flash(f'This request has already been {request_obj.status}.', 'warning')
            return redirect(url_for('manage_enrollment_requests'))

        remarks = request.form.get('remarks', 'Rejected by teacher')

        request_obj.status = 'rejected'
        request_obj.reviewed_at = datetime.utcnow()
        request_obj.reviewed_by = session['user_id']
        request_obj.remarks = remarks

        db.session.commit()

        student = User.query.get(request_obj.student_id)
        flash(f'Enrollment rejected for {student.fullname} in {course.course_code}.', 'warning')

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error rejecting enrollment: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')

    return redirect(url_for('manage_enrollment_requests'))


# ─────────────────────────────────────────────
# TEACHER INVITE ROUTES - UPDATED
# ─────────────────────────────────────────────

@app.route('/teacher/generate-student-invite', methods=['GET', 'POST'])
@login_required
@teacher_or_admin_required
def generate_student_invite():
    """Generate a single invite link that enrolls students in ALL teacher's courses"""
    try:
        user = User.query.get(session['user_id'])

        teacher_courses = Course.query.filter_by(teacher_id=user.id, is_active=True).all()

        if not teacher_courses:
            flash('You need to create at least one course first!', 'warning')
            return redirect(url_for('list_courses'))

        if request.method == 'POST':
            existing_invite = InviteCode.query.filter_by(
                created_by=user.id,
                is_teacher_invite=True,
                used=False
            ).first()

            # FIX: `invite` must be assigned in BOTH branches -- previously
            # only the "create a new one" branch set it, so reusing an
            # already-valid invite (the normal case on every visit after
            # the first) raised UnboundLocalError on the render_template()
            # call below, which references `invite=invite` unconditionally.
            if existing_invite and existing_invite.is_valid():
                invite = existing_invite
                invite_code = existing_invite.code
                flash('✅ Using existing valid invite link.', 'info')
            else:
                code = secrets.token_urlsafe(12)
                expires_at = datetime.utcnow() + timedelta(days=30)

                invite = InviteCode(
                    code=code,
                    role='student',
                    is_teacher_invite=True,
                    created_by=user.id,
                    max_uses=1000,
                    expires_at=expires_at,
                    created_at=datetime.utcnow(),
                    used=False,
                    uses_count=0
                )

                db.session.add(invite)
                db.session.commit()
                invite_code = code

            courses = Course.query.filter_by(teacher_id=user.id, is_active=True).all()
            registration_link = f"{request.host_url}register?code={invite_code}"

            flash('✅ Student registration link generated successfully!', 'success')

            return render_template('teacher/teacher_invite.html',
                                 teacher=user,
                                 courses=courses,
                                 registration_link=registration_link,
                                 invite_code=invite_code,
                                 invite=invite)

        return render_template('teacher/generate_teacher_invite.html',
                             teacher=user,
                             courses=teacher_courses)

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error generating teacher invite: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash(f'Error generating invite: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/teacher/invite-status')
@login_required
@teacher_or_admin_required
def teacher_invite_status():
    """Check the status of the teacher's invite link"""
    try:
        user = User.query.get(session['user_id'])

        invite = InviteCode.query.filter_by(
            created_by=user.id,
            is_teacher_invite=True
        ).first()

        if not invite:
            return jsonify({
                'has_invite': False,
                'message': 'No invite link found. Generate one first.'
            })

        return jsonify({
            'has_invite': True,
            'code': invite.code,
            'uses_count': invite.uses_count,
            'max_uses': invite.max_uses,
            'expires_at': invite.expires_at.isoformat() if invite.expires_at else None,
            'is_valid': invite.is_valid(),
            'created_at': invite.created_at.isoformat() if invite.created_at else None,
            'link': f"{request.host_url}register?code={invite.code}"
        })

    except Exception as e:
        app.logger.error(f"Error checking invite status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/teacher/generate-token', methods=['POST'])
@login_required
@teacher_or_admin_required
def generate_course_token_teacher():
    try:
        course_id = request.form.get('course_id')
        max_uses = request.form.get('max_uses', 50)
        expires_days = request.form.get('expires_days', 7)

        if not course_id:
            flash('Please select a course.', 'danger')
            return redirect(url_for('manage_links'))

        course = Course.query.get_or_404(course_id)

        if session.get('user_role') != 'admin' and course.teacher_id != session['user_id']:
            flash('You do not have permission to generate links for this course.', 'danger')
            return redirect(url_for('manage_links'))

        token_code = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(8))
        expires_at = datetime.utcnow() + timedelta(days=int(expires_days))

        token = CourseEnrollmentToken(
            course_id=course_id,
            token_code=token_code,
            expires_at=expires_at,
            max_uses=int(max_uses),
            created_by=session['user_id'],
            created_at=datetime.utcnow(),
            is_active=True
        )

        db.session.add(token)
        db.session.commit()

        flash(f'Enrollment link generated for {course.course_code}!', 'success')

        enrollment_link = f"{request.host_url}courses/enroll/{token_code}"

        return render_template('teacher/token_generated.html',
                             token=token,
                             course=course,
                             enrollment_link=enrollment_link)

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error generating token: {str(e)}")
        flash(f'Error generating link: {str(e)}', 'danger')
        return redirect(url_for('manage_links'))


@app.route('/teacher/manage-links', methods=['GET'])
@login_required
@teacher_or_admin_required
def manage_links():
    try:
        user = User.query.get(session['user_id'])

        if user.role == 'admin':
            teacher_courses = Course.query.all()
        else:
            teacher_courses = Course.query.filter_by(teacher_id=user.id).all()

        active_links = []
        expired_links = []

        for course in teacher_courses:
            tokens = CourseEnrollmentToken.query.filter_by(
                course_id=course.id,
                is_active=True
            ).all()

            for token in tokens:
                if token.is_valid():
                    active_links.append({
                        'id': token.id,
                        'token_code': token.token_code,
                        'course': course,
                        'max_uses': token.max_uses,
                        'used_count': token.used_count,
                        'created_at': token.created_at,
                        'expires_at': token.expires_at
                    })
                else:
                    expired_links.append({
                        'id': token.id,
                        'token_code': token.token_code,
                        'course': course,
                        'max_uses': token.max_uses,
                        'used_count': token.used_count,
                        'created_at': token.created_at,
                        'expires_at': token.expires_at
                    })

        return render_template('teacher/manage_links.html',
                             teacher_courses=teacher_courses,
                             active_links=active_links,
                             expired_links=expired_links)

    except Exception as e:
        app.logger.error(f"Error managing links: {str(e)}")
        flash('Error loading enrollment links.', 'danger')
        return redirect(url_for('dashboard'))


# ─────────────────────────────────────────────
# Student Enrollment via Link Routes
# ─────────────────────────────────────────────

@app.route('/courses/enroll/<token_code>')
@login_required
def enroll_via_link(token_code):
    try:
        if session.get('user_role') != 'student':
            flash('Only students can enroll in courses.', 'danger')
            return redirect(url_for('dashboard'))

        token = CourseEnrollmentToken.query.filter_by(
            token_code=token_code,
            is_active=True
        ).first()

        if not token:
            flash('Invalid or expired enrollment link.', 'danger')
            return redirect(url_for('browse_courses'))

        if not token.is_valid():
            if token.expires_at and datetime.utcnow() > token.expires_at:
                flash('This enrollment link has expired.', 'danger')
            elif token.max_uses != -1 and token.used_count >= token.max_uses:
                flash('This enrollment link has reached its maximum uses.', 'danger')
            else:
                flash('This enrollment link is no longer valid.', 'danger')
            return redirect(url_for('browse_courses'))

        course = Course.query.get(token.course_id)
        if not course:
            flash('Course not found.', 'danger')
            return redirect(url_for('browse_courses'))

        existing = CourseEnrollment.query.filter_by(
            student_id=session['user_id'],
            course_id=course.id
        ).first()

        if existing:
            flash(f'You are already enrolled in {course.course_code} - {course.course_name}!', 'info')
            return redirect(url_for('view_course', course_id=course.id))

        pending = CourseEnrollmentRequest.query.filter_by(
            student_id=session['user_id'],
            course_id=course.id,
            status='pending'
        ).first()

        if pending:
            flash(f'You already have a pending enrollment request for {course.course_code}.', 'info')
            return redirect(url_for('browse_courses'))

        enrollment = CourseEnrollment(
            student_id=session['user_id'],
            course_id=course.id,
            enrolled_at=datetime.utcnow(),
            status='active'
        )

        token.used_count += 1

        db.session.add(enrollment)
        db.session.commit()

        flash(f'Successfully enrolled in {course.course_code} - {course.course_name}!', 'success')
        return redirect(url_for('view_course', course_id=course.id))

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error enrolling via link: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('browse_courses'))


@app.route('/courses/enroll', methods=['GET', 'POST'])
@login_required
def enroll_with_token():
    try:
        if session.get('user_role') != 'student':
            flash('Only students can enroll in courses.', 'danger')
            return redirect(url_for('dashboard'))

        if request.method == 'POST':
            token_code = request.form.get('token_code', '').strip().upper()

            if not token_code or len(token_code) != 8:
                flash('Please enter a valid 8-character token.', 'danger')
                return render_template('courses/enroll_course.html', course=None)

            token = CourseEnrollmentToken.query.filter_by(
                token_code=token_code,
                is_active=True
            ).first()

            if not token:
                flash('Invalid enrollment token. Please check with your teacher.', 'danger')
                return render_template('courses/enroll_course.html', course=None)

            if not token.is_valid():
                if token.expires_at and datetime.utcnow() > token.expires_at:
                    flash('This enrollment token has expired.', 'danger')
                elif token.max_uses != -1 and token.used_count >= token.max_uses:
                    flash('This enrollment token has reached its maximum uses.', 'danger')
                else:
                    flash('This enrollment token is no longer valid.', 'danger')
                return render_template('courses/enroll_course.html', course=None)

            course = Course.query.get(token.course_id)
            if not course:
                flash('Course not found.', 'danger')
                return render_template('courses/enroll_course.html', course=None)

            existing = CourseEnrollment.query.filter_by(
                student_id=session['user_id'],
                course_id=course.id
            ).first()

            if existing:
                flash(f'You are already enrolled in {course.course_code} - {course.course_name}!', 'info')
                return redirect(url_for('view_course', course_id=course.id))

            pending = CourseEnrollmentRequest.query.filter_by(
                student_id=session['user_id'],
                course_id=course.id,
                status='pending'
            ).first()

            if pending:
                flash(f'You already have a pending enrollment request for {course.course_code}.', 'info')
                return redirect(url_for('browse_courses'))

            enrollment = CourseEnrollment(
                student_id=session['user_id'],
                course_id=course.id,
                enrolled_at=datetime.utcnow(),
                status='active'
            )

            token.used_count += 1
            if token.max_uses != -1 and token.used_count >= token.max_uses:
                token.is_active = False

            db.session.add(enrollment)
            db.session.commit()

            flash(f'Successfully enrolled in {course.course_code} - {course.course_name}!', 'success')
            return redirect(url_for('view_course', course_id=course.id))

        return render_template('courses/enroll_course.html', course=None)

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error enrolling with token: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')
        return redirect(url_for('browse_courses'))


# ─────────────────────────────────────────────
# Course Management Routes
# ─────────────────────────────────────────────

@app.route('/courses')
@login_required
def list_courses():
    try:
        user = User.query.get(session['user_id'])

        if not user:
            flash('User not found. Please login again.', 'danger')
            return redirect(url_for('login'))

        if user.role == 'teacher':
            courses = Course.query.filter_by(teacher_id=user.id).all()
            return render_template('courses/teacher_courses.html', courses=courses)

        elif user.role == 'student':
            return redirect(url_for('student_courses'))

        elif user.role == 'admin':
            courses = Course.query.all()
            return render_template('courses/admin_courses.html', courses=courses)

        else:
            flash('Invalid user role.', 'danger')
            return redirect(url_for('dashboard'))

    except Exception as e:
        app.logger.error(f"Error in list_courses: {str(e)}")
        flash('An error occurred while loading your courses. Please try again.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/courses/create', methods=['GET', 'POST'])
@login_required
@teacher_or_admin_required
def create_course():
    if request.method == 'POST':
        try:
            course_code = request.form.get('course_code', '').strip().upper()
            course_name = request.form.get('course_name', '').strip()
            description = request.form.get('description', '').strip()
            schedule = request.form.get('schedule', '').strip()

            if not course_code or not course_name:
                flash('Course code and name are required!', 'danger')
                return render_template('courses/create_course.html')

            existing = Course.query.filter_by(course_code=course_code).first()
            if existing:
                flash(f'Course code "{course_code}" already exists!', 'danger')
                return render_template('courses/create_course.html')

            course = Course(
                course_code=course_code,
                course_name=course_name,
                description=description,
                schedule=schedule,
                teacher_id=session['user_id'],
                created_at=datetime.utcnow(),
                is_active=True
            )

            db.session.add(course)
            db.session.commit()

            flash(f'Course "{course_code} - {course_name}" created successfully!', 'success')
            return redirect(url_for('view_course', course_id=course.id))

        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error creating course: {str(e)}")
            flash(f'Error: {str(e)}', 'danger')
            return render_template('courses/create_course.html')

    return render_template('courses/create_course.html')


@app.route('/courses/<int:course_id>')
@login_required
def view_course(course_id):
    try:
        course = Course.query.get_or_404(course_id)
        user = User.query.get(session['user_id'])

        if user.role == 'student':
            return redirect(url_for('student_course_detail', course_id=course_id))

        if user.role == 'teacher' and course.teacher_id != user.id and session.get('user_role') != 'admin':
            flash('You can only view your own courses.', 'danger')
            return redirect(url_for('list_courses'))

        enrollments = CourseEnrollment.query.filter_by(course_id=course_id).all()
        enrolled_students = [User.query.get(e.student_id) for e in enrollments]

        sessions = AttendanceSession.query.filter_by(course_id=course_id).order_by(
            AttendanceSession.session_date.desc()
        ).all()

        pending_requests = []
        enrollment_tokens = []

        if user.role == 'teacher' or user.role == 'admin':
            pending_requests = CourseEnrollmentRequest.query.filter_by(
                course_id=course_id,
                status='pending'
            ).all()

            enrollment_tokens = CourseEnrollmentToken.query.filter_by(
                course_id=course_id,
                is_active=True
            ).all()

        return render_template('courses/view_course.html',
                             course=course,
                             enrolled_students=enrolled_students,
                             sessions=sessions,
                             pending_requests=pending_requests,
                             enrollment_tokens=enrollment_tokens)

    except Exception as e:
        app.logger.error(f"Error viewing course: {str(e)}")
        flash('Error loading course.', 'danger')
        return redirect(url_for('list_courses'))


@app.route('/courses/<int:course_id>/start-session', methods=['GET', 'POST'])
@login_required
@teacher_or_admin_required
def start_attendance_session(course_id):
    course = Course.query.get_or_404(course_id)

    try:
        if course.teacher_id != session['user_id'] and session.get('user_role') != 'admin':
            flash('You can only start sessions for your own courses.', 'danger')
            return redirect(url_for('view_course', course_id=course_id))

        active = AttendanceSession.query.filter_by(course_id=course_id, status='active').first()
        if active:
            flash(f'There is already an active session for this course (started at {active.session_time}).', 'warning')
            return redirect(url_for('view_session', session_id=active.id))

        if request.method == 'POST':
            topic = request.form.get('topic', 'Regular Class')
            session_token = ''.join(secrets.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') for _ in range(6))

            new_session = AttendanceSession(
                course_id=course_id,
                session_date=datetime.now().date(),
                session_time=datetime.now().time(),
                topic=topic,
                status='active',
                created_by=session['user_id'],
                created_at=datetime.utcnow(),
                session_token=session_token
            )

            db.session.add(new_session)
            db.session.commit()

            flash(f'Attendance session started for {course.course_code}! Token: {session_token}', 'success')
            return redirect(url_for('view_session', session_id=new_session.id))

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error starting session: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')

    return render_template('courses/start_session.html', course=course)


@app.route('/courses/session/<int:session_id>')
@login_required
def view_session(session_id):
    try:
        session_obj = AttendanceSession.query.get_or_404(session_id)
        course = Course.query.get(session_obj.course_id)
        user = User.query.get(session['user_id'])

        if user.role == 'student':
            enrollment = CourseEnrollment.query.filter_by(
                student_id=user.id,
                course_id=course.id
            ).first()
            if not enrollment:
                flash('Access denied.', 'danger')
                return redirect(url_for('list_courses'))
        elif user.role == 'teacher' and course.teacher_id != user.id and session.get('user_role') != 'admin':
            flash('Access denied.', 'danger')
            return redirect(url_for('list_courses'))

        records = CourseAttendanceRecord.query.filter_by(session_id=session_id).all()

        enrollments = CourseEnrollment.query.filter_by(course_id=course.id).all()
        enrolled_users = [User.query.get(e.student_id) for e in enrollments]

        attended_ids = [r.student_id for r in records]

        return render_template('courses/view_session.html',
                             session=session_obj,
                             course=course,
                             records=records,
                             enrolled_users=enrolled_users,
                             attended_ids=attended_ids)

    except Exception as e:
        app.logger.error(f"Error viewing session: {str(e)}")
        flash('Error loading session.', 'danger')
        return redirect(url_for('list_courses'))


@app.route('/courses/session/<int:session_id>/end', methods=['POST'])
@login_required
@teacher_or_admin_required
def end_attendance_session(session_id):
    session_obj = AttendanceSession.query.get_or_404(session_id)
    course = Course.query.get_or_404(session_obj.course_id)

    try:
        if course.teacher_id != session['user_id'] and session.get('user_role') != 'admin':
            flash('Access denied.', 'danger')
            return redirect(url_for('list_courses'))

        session_obj.status = 'closed'
        db.session.commit()

        present_count = CourseAttendanceRecord.query.filter_by(session_id=session_id).count()
        total_enrolled = CourseEnrollment.query.filter_by(course_id=course.id).count()

        flash(f'Session ended. {present_count}/{total_enrolled} students marked present.', 'info')

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error ending session: {str(e)}")
        flash(f'Error: {str(e)}', 'danger')

    return redirect(url_for('view_course', course_id=course.id))


# ─────────────────────────────────────────────
# Course Attendance Marking with Session Token Validation
# ─────────────────────────────────────────────

@app.route('/courses/session/<int:session_id>/mark', methods=['POST'])
@login_required
def mark_course_attendance(session_id):
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data received'})

        session_token = data.get('session_token')
        if not session_token:
            return jsonify({'success': False, 'message': 'Session token required'})

        session_obj = AttendanceSession.query.get(session_id)
        if not session_obj:
            return jsonify({'success': False, 'message': 'Invalid session'})

        if session_obj.session_token != session_token:
            return jsonify({'success': False, 'message': 'Invalid session token'})

        if session_obj.status != 'active':
            return jsonify({'success': False, 'message': 'This attendance session is closed'})

        course = Course.query.get(session_obj.course_id)
        if not course:
            return jsonify({'success': False, 'message': 'Course not found'})

        enrollment = CourseEnrollment.query.filter_by(
            student_id=session['user_id'],
            course_id=course.id,
            status='active'
        ).first()

        if not enrollment:
            return jsonify({'success': False, 'message': 'You are not enrolled in this course'})

        existing = CourseAttendanceRecord.query.filter_by(
            session_id=session_id,
            student_id=session['user_id']
        ).first()

        if existing:
            return jsonify({
                'success': False,
                'message': f'{session["user_name"]} already marked attendance for this session'
            })

        img_bgr, img_rgb = decode_image_from_base64(data['image'])
        if img_bgr is None:
            return jsonify({'success': False, 'message': 'Could not decode image'})

        face_encodings = face_recognition.face_encodings(img_rgb)

        if not face_encodings:
            return jsonify({'success': False, 'message': 'No face detected. Please look at the camera.'})

        if len(face_encodings) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected.'})

        student = Student.query.filter_by(user_id=session['user_id']).first()

        if not student:
            return jsonify({
                'success': False,
                'message': 'Please enroll your face first. Go to the Enrollment page.'
            })

        stored_encoding = student.get_encoding()
        if len(stored_encoding) == 0:
            return jsonify({
                'success': False,
                'message': 'Invalid face encoding. Please re-enroll your face.'
            })

        face_to_compare = face_encodings[0]

        face_distance = face_recognition.face_distance([stored_encoding], face_to_compare)[0]
        confidence = max(0, min(100, (1 - face_distance) * 100))

        app.logger.info(f"Face distance: {face_distance}, Confidence: {confidence}%")

        if face_distance > CONFIDENCE_THRESHOLD:
            return jsonify({
                'success': False,
                'message': 'Face not recognized. Please try again with better lighting.'
            })

        record = CourseAttendanceRecord(
            session_id=session_id,
            student_id=session['user_id'],
            face_confidence=confidence,
            status='present',
            marked_at=datetime.utcnow()
        )

        db.session.add(record)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Attendance marked for {session["user_name"]}',
            'student_name': session['user_name'],
            'student_id': session['user_id'],
            'confidence': round(confidence, 2),
            'time': datetime.now().strftime('%H:%M:%S')
        })

    except Exception as e:
        db.session.rollback()
        app.logger.exception(f'Error marking course attendance: {str(e)}')
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/mark_attendance')
@login_required
def mark_attendance():
    """Mark attendance page with course selection"""
    try:
        # Get user
        user = User.query.get(session['user_id'])
        if not user:
            flash('User not found.', 'danger')
            return redirect(url_for('login'))

        # Get courses based on role
        courses = []
        if user.role == 'student':
            enrollments = CourseEnrollment.query.filter_by(
                student_id=user.id,
                status='active'
            ).all()
            for e in enrollments:
                course = Course.query.get(e.course_id)
                if course:
                    courses.append(course)
        elif user.role == 'teacher':
            courses = Course.query.filter_by(teacher_id=user.id, is_active=True).all()
        elif user.role == 'admin':
            courses = Course.query.filter_by(is_active=True).all()

        # Get selected course
        selected_course_id = request.args.get('course', type=int)
        selected_course = None
        active_session = None
        recent_attendance = []

        # FIX: previously this block silently set `selected_course = None`
        # whenever the access check failed, and the code below it would
        # then silently fall back to `courses[0]` with zero explanation.
        # From the user's side that looked like "clicking a different
        # course does nothing / camera only ever shows one course" -- the
        # page was quietly reverting to your first course every time.
        # Now we flash the actual reason instead of swallowing it.
        if selected_course_id:
            candidate_course = Course.query.get(selected_course_id)
            if not candidate_course:
                flash('That course could not be found.', 'warning')
            else:
                access_ok = True
                if user.role == 'student':
                    enrollment = CourseEnrollment.query.filter_by(
                        student_id=user.id,
                        course_id=selected_course_id,
                        status='active'
                    ).first()
                    if not enrollment:
                        access_ok = False
                        flash(
                            f'You do not have an active enrollment in '
                            f'{candidate_course.course_code} - {candidate_course.course_name}, '
                            f'so attendance can\'t be marked for it yet. If a teacher told you '
                            f'this request was approved, check Browse Courses or contact them -- '
                            f'the enrollment record may not have been created correctly.',
                            'warning'
                        )
                elif user.role == 'teacher' and candidate_course.teacher_id != user.id:
                    access_ok = False
                    flash('You can only mark attendance for your own courses.', 'danger')

                if access_ok:
                    selected_course = candidate_course

        # If no course selected but courses exist, select first one
        if not selected_course and courses:
            selected_course = courses[0]
            selected_course_id = selected_course.id

        # Get active session for selected course
        if selected_course:
            active_session = AttendanceSession.query.filter_by(
                course_id=selected_course.id,
                status='active'
            ).first()

            # Get recent attendance
            # NOTE (fix): renamed loop var from `session` to `att_session`.
            # The original code did `for session in sessions:`, which
            # shadowed the imported Flask `session` object for the rest of
            # this function. It happened to be harmless today because
            # nothing after the loop reads `session[...]` again in this
            # route, but it was a landmine for any future edit.
            sessions = AttendanceSession.query.filter_by(
                course_id=selected_course.id
            ).order_by(AttendanceSession.session_date.desc()).limit(5).all()

            for att_session in sessions:
                records = CourseAttendanceRecord.query.filter_by(
                    session_id=att_session.id
                ).all()
                for record in records:
                    student = User.query.get(record.student_id)
                    if student:
                        recent_attendance.append({
                            'student_name': student.fullname,
                            'time': record.marked_at.strftime('%I:%M %p') if record.marked_at else 'N/A',
                            'date': record.marked_at.strftime('%b %d, %Y') if record.marked_at else 'N/A',
                            'confidence': record.face_confidence
                        })

        return render_template('mark_attendance.html',
                             user=user,
                             courses=courses,
                             selected_course=selected_course,
                             selected_course_id=selected_course_id,
                             active_session=active_session,
                             recent_attendance=recent_attendance,
                             now=datetime.now())

    except Exception as e:
        app.logger.error(f"Error loading mark attendance: {str(e)}")
        app.logger.error(traceback.format_exc())
        flash('Error loading attendance page. Please try again.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/recognize_face', methods=['POST'])
@login_required
def recognize_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data received'})

        img_bgr, img_rgb = decode_image_from_base64(data['image'])
        if img_bgr is None:
            return jsonify({'success': False, 'message': 'Could not decode image'})

        face_encodings = face_recognition.face_encodings(img_rgb)

        if not face_encodings:
            return jsonify({'success': False, 'message': 'No face detected. Please look at the camera.'})

        if len(face_encodings) > 1:
            return jsonify({'success': False, 'message': 'Multiple faces detected. Please ensure only one person is in frame.'})

        known_encodings, known_students = get_known_faces()
        if not known_encodings:
            return jsonify({'success': False, 'message': 'No enrolled students found. Please enroll first.'})

        face_to_compare = face_encodings[0]
        face_distances = face_recognition.face_distance(known_encodings, face_to_compare)
        best_idx = int(np.argmin(face_distances))
        best_distance = float(face_distances[best_idx])

        if best_distance < CONFIDENCE_THRESHOLD:
            matched = known_students[best_idx]
            today = datetime.now().date().isoformat()

            already = AttendanceRecord.query.filter_by(
                student_id=matched.student_id, date=today
            ).first()

            if already:
                return jsonify({'success': False,
                                'message': f'{matched.fullname} already marked attendance today!'})

            record = AttendanceRecord(
                student_id=matched.student_id,
                student_name=matched.fullname,
                date=today,
                time=datetime.now().strftime('%H:%M:%S')
            )
            db.session.add(record)
            db.session.commit()

            return jsonify({
                'success': True,
                'student_name': matched.fullname,
                'student_id': matched.student_id,
                'time': record.time,
                'message': f'Attendance marked for {matched.fullname}'
            })

        return jsonify({'success': False, 'message': 'Face not recognised. Please ensure you are enrolled and well-lit.'})

    except Exception as e:
        db.session.rollback()
        app.logger.exception('recognize_face error')
        return jsonify({'success': False, 'message': f'Error processing face: {str(e)}'})


@app.route('/attendance_stats')
@login_required
def attendance_stats():
    try:
        total_students = Student.query.count()
        today = datetime.now().date().isoformat()
        present_today = AttendanceRecord.query.filter_by(date=today).count()
        absent_today = total_students - present_today
        attendance_rate = round((present_today / total_students) * 100, 2) if total_students > 0 else 0

        return jsonify({
            'total_students': total_students,
            'present_today': present_today,
            'absent_today': absent_today,
            'attendance_rate': attendance_rate
        })
    except Exception as e:
        app.logger.error(f"Error getting attendance stats: {str(e)}")
        return jsonify({'error': 'Error loading stats'}), 500


@app.route('/students')
@login_required
@teacher_or_admin_required
def get_students():
    """List students. Admins see everyone; teachers only see students
    actively enrolled in one of their own courses."""
    try:
        user = User.query.get(session['user_id'])

        if user.role == 'teacher':
            teacher_courses = Course.query.filter_by(teacher_id=user.id).all()
            course_ids = [c.id for c in teacher_courses]

            if course_ids:
                enrollments = CourseEnrollment.query.filter(
                    CourseEnrollment.course_id.in_(course_ids),
                    CourseEnrollment.status == 'active'
                ).all()
                student_user_ids = [e.student_id for e in enrollments]
                students = Student.query.filter(Student.user_id.in_(student_user_ids)).all()
            else:
                students = []

            return jsonify([{
                'student_id': s.student_id,
                'fullname': s.fullname,
                'course': s.course,
            } for s in students])

        students = Student.query.all()
        return jsonify([s.to_dict() for s in students])
    except Exception as e:
        app.logger.error(f"Error getting students: {str(e)}")
        return jsonify({'error': 'Error loading students'}), 500


# ─────────────────────────────────────────────
# STUDENT SEARCH ROUTE
# ─────────────────────────────────────────────

@app.route('/search_students')
@login_required
@teacher_or_admin_required
def search_students():
    """Search students with role-based access control."""
    try:
        q = request.args.get('q', '').strip()
        user = User.query.get(session['user_id'])

        query = Student.query

        if q:
            like = f'%{q}%'
            query = query.filter(
                db.or_(
                    Student.fullname.ilike(like),
                    Student.student_id.ilike(like),
                    Student.email.ilike(like),
                    Student.course.ilike(like)
                )
            )

        if user.role == 'teacher':
            teacher_courses = Course.query.filter_by(teacher_id=user.id).all()
            course_ids = [c.id for c in teacher_courses]

            if course_ids:
                enrollments = CourseEnrollment.query.filter(
                    CourseEnrollment.course_id.in_(course_ids),
                    CourseEnrollment.status == 'active'
                ).all()
                student_ids = [e.student_id for e in enrollments]
                query = query.filter(Student.user_id.in_(student_ids))
            else:
                query = query.filter(db.false())

        results = query.all()

        students_data = []
        for student in results:
            user_account = User.query.get(student.user_id)
            is_approved = user_account.is_approved if user_account else False

            students_data.append({
                'id': student.id,
                'student_id': student.student_id,
                'fullname': student.fullname,
                'email': student.email,
                'course': student.course,
                'level': student.level,
                'role': 'student',
                'is_approved': is_approved,
                'face_image': student.face_image
            })

        return jsonify(students_data)

    except Exception as e:
        app.logger.error(f"Error searching students: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Error searching students'}), 500


@app.route('/search_students_page')
@login_required
@teacher_or_admin_required
def search_students_page():
    """Render the student search page"""
    return render_template('search_students.html')


@app.route('/admin/delete-student/<int:student_id>', methods=['POST'])
@login_required
@admin_required
def delete_student(student_id):
    """Delete a student record (admin only)"""
    try:
        student = Student.query.get_or_404(student_id)
        student_name = student.fullname

        if student.face_image and os.path.exists(student.face_image):
            try:
                os.remove(student.face_image)
            except Exception as e:
                app.logger.warning(f"Could not delete face image: {str(e)}")

        db.session.delete(student)
        db.session.commit()
        invalidate_face_cache()

        return jsonify({'success': True, 'message': f'Student {student_name} deleted successfully'})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting student: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────
# Admin Routes
# ─────────────────────────────────────────────

@app.route('/admin/users')
@login_required
@admin_required
def manage_users():
    try:
        users = User.query.order_by(User.created_at.desc()).all()

        stats = {
            'total': len(users),
            'pending': sum(1 for u in users if not u.is_approved),
            'approved': sum(1 for u in users if u.is_approved),
            'admins': sum(1 for u in users if u.role == 'admin'),
            'teachers': sum(1 for u in users if u.role == 'teacher'),
            'students': sum(1 for u in users if u.role == 'student'),
        }

        return render_template('admin/users.html', users=users, stats=stats)
    except Exception as e:
        app.logger.error(f"Error loading users: {str(e)}")
        flash('Error loading users.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/admin/approve_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def approve_user(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if user.is_approved:
            flash(f'User {user.fullname} is already approved.', 'info')
            return redirect(url_for('manage_users'))

        user.is_approved = True
        db.session.commit()

        flash(f'User {user.fullname} has been approved!', 'success')
        return redirect(url_for('manage_users'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error approving user: {str(e)}")
        flash('Error approving user.', 'danger')
        return redirect(url_for('manage_users'))


@app.route('/admin/reject_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def reject_user(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if user.is_approved:
            flash('Cannot reject an approved user.', 'warning')
            return redirect(url_for('manage_users'))

        user_name = user.fullname

        db.session.delete(user)
        db.session.commit()

        flash(f'User {user_name} has been rejected and removed.', 'danger')
        return redirect(url_for('manage_users'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error rejecting user: {str(e)}")
        flash('Error rejecting user.', 'danger')
        return redirect(url_for('manage_users'))


@app.route('/admin/change_role/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def change_user_role(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if user.id == session['user_id']:
            flash('You cannot change your own role!', 'warning')
            return redirect(url_for('manage_users'))

        new_role = request.form.get('role')
        if new_role not in ('admin', 'teacher', 'student'):
            flash('Invalid role selected!', 'danger')
            return redirect(url_for('manage_users'))

        old_role = user.role
        user.role = new_role
        db.session.commit()

        flash(f'Role updated from {old_role} to {new_role} for {user.fullname}', 'success')
        return redirect(url_for('manage_users'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error changing user role: {str(e)}")
        flash('Error changing role.', 'danger')
        return redirect(url_for('manage_users'))


@app.route('/admin/toggle_approval/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def toggle_approval(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if user.id == session['user_id']:
            flash('You cannot change your own approval status!', 'warning')
            return redirect(url_for('manage_users'))

        user.is_approved = not user.is_approved
        db.session.commit()

        status = "approved" if user.is_approved else "unapproved"
        flash(f'{user.fullname} has been {status}.', 'success' if user.is_approved else 'warning')
        return redirect(url_for('manage_users'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error toggling approval: {str(e)}")
        flash('Error toggling approval.', 'danger')
        return redirect(url_for('manage_users'))


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if user.id == session['user_id']:
            flash('You cannot delete your own account!', 'warning')
            return redirect(url_for('manage_users'))

        user_name = user.fullname

        db.session.delete(user)
        db.session.commit()

        flash(f'User {user_name} has been deleted.', 'danger')
        return redirect(url_for('manage_users'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting user: {str(e)}")
        flash('Error deleting user.', 'danger')
        return redirect(url_for('manage_users'))


@app.route('/admin/invites')
@login_required
@admin_required
def manage_invites():
    try:
        invites = InviteCode.query.order_by(InviteCode.created_at.desc()).all()
        return render_template('admin/invites.html', invites=invites, now=datetime.now())
    except Exception as e:
        app.logger.error(f"Error loading invites: {str(e)}")
        flash('Error loading invites.', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/admin/generate_invite', methods=['GET', 'POST'])
@login_required
@admin_required
def generate_invite():
    if request.method == 'POST':
        try:
            role = request.form.get('role', 'student')
            max_uses = request.form.get('max_uses', 1)
            expires_days = request.form.get('expires_days')

            if role not in ('admin', 'teacher', 'student'):
                flash('Invalid role selected!', 'danger')
                return redirect(url_for('generate_invite'))

            code = secrets.token_urlsafe(12)

            expires_at = None
            if expires_days and expires_days.isdigit():
                expires_at = datetime.utcnow() + timedelta(days=int(expires_days))
            else:
                expires_at = datetime.utcnow() + timedelta(days=7)

            invite = InviteCode(
                code=code,
                role=role,
                created_by=session['user_id'],
                max_uses=int(max_uses) if max_uses else 1,
                expires_at=expires_at,
                created_at=datetime.utcnow(),
                used=False,
                uses_count=0
            )

            db.session.add(invite)
            db.session.commit()

            flash(f'{role.capitalize()} invite code generated successfully!', 'success')

            invite_link = f"{request.host_url}register?code={code}"
            return render_template('admin/invite_success.html',
                                 invite=invite,
                                 invite_link=invite_link)

        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error generating invite: {str(e)}")
            flash(f'Error generating invite: {str(e)}', 'danger')
            return redirect(url_for('manage_invites'))

    return render_template('admin/generate_invite.html')


@app.route('/admin/delete_invite/<int:invite_id>', methods=['POST'])
@login_required
@admin_required
def delete_invite(invite_id):
    try:
        invite = InviteCode.query.get_or_404(invite_id)

        if invite.used:
            flash('Cannot delete a used invite code.', 'warning')
            return redirect(url_for('manage_invites'))

        db.session.delete(invite)
        db.session.commit()
        flash('Invite code deleted successfully.', 'success')
        return redirect(url_for('manage_invites'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error deleting invite: {str(e)}")
        flash('Error deleting invite.', 'danger')
        return redirect(url_for('manage_invites'))


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.route('/api/courses/<int:course_id>/active-sessions')
@login_required
def api_active_sessions(course_id):
    try:
        course = Course.query.get_or_404(course_id)

        if session.get('user_role') == 'student':
            enrollment = CourseEnrollment.query.filter_by(
                student_id=session['user_id'],
                course_id=course_id
            ).first()
            if not enrollment:
                return jsonify({'error': 'Not enrolled in this course'}), 403

        sessions = AttendanceSession.query.filter_by(
            course_id=course_id,
            status='active'
        ).order_by(AttendanceSession.session_date.desc()).all()

        return jsonify({
            'sessions': [{
                'id': s.id,
                'session_date': s.session_date.isoformat() if s.session_date else None,
                'session_time': s.session_time.strftime('%H:%M:%S') if s.session_time else None,
                'topic': s.topic,
                'status': s.status,
                'session_token': s.session_token
            } for s in sessions]
        })
    except Exception as e:
        app.logger.error(f"Error in api_active_sessions: {str(e)}")
        return jsonify({'error': 'Error loading sessions'}), 500


@app.route('/api/sessions/<int:session_id>')
@login_required
def api_session_details(session_id):
    try:
        session_obj = AttendanceSession.query.get_or_404(session_id)
        course = Course.query.get(session_obj.course_id)

        if session.get('user_role') == 'student':
            enrollment = CourseEnrollment.query.filter_by(
                student_id=session['user_id'],
                course_id=course.id
            ).first()
            if not enrollment:
                return jsonify({'error': 'Not enrolled'}), 403

        return jsonify({
            'id': session_obj.id,
            'course_id': course.id,
            'course_code': course.course_code,
            'session_date': session_obj.session_date.isoformat() if session_obj.session_date else None,
            'session_time': session_obj.session_time.strftime('%H:%M:%S') if session_obj.session_time else None,
            'topic': session_obj.topic,
            'status': session_obj.status,
            'session_token': session_obj.session_token,
            'created_at': session_obj.created_at.isoformat() if session_obj.created_at else None
        })
    except Exception as e:
        app.logger.error(f"Error in api_session_details: {str(e)}")
        return jsonify({'error': 'Error loading session details'}), 500


# ─────────────────────────────────────────────
# CAMERA TEST ROUTE
# ─────────────────────────────────────────────

@app.route('/camera_test')
@login_required
def camera_test():
    return render_template('camera_test.html')


# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    flash('Your session expired or the form was invalid. Please try again.', 'danger')
    return redirect(url_for('login'))


@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(e):
    return jsonify({'success': False, 'message': 'Uploaded image is too large.'}), 413


@app.errorhandler(403)
def handle_forbidden(e):
    return render_template('403.html'), 403


@app.errorhandler(404)
def handle_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def handle_internal_error(e):
    app.logger.error(f"500 Error: {str(e)}")
    flash('An internal server error occurred. Please try again.', 'danger')
    return redirect(url_for('home'))


# ─────────────────────────────────────────────
# Init DB and run
# ─────────────────────────────────────────────

def init_db():
    with app.app_context():
        try:
            db.session.execute(text('PRAGMA foreign_keys=ON'))
            db.session.commit()

            db.create_all()
            print("=" * 50)
            print("Database initialized successfully!")
            print("Tables created:", ', '.join(db.metadata.tables.keys()))
            print("=" * 50)

            admin_email = os.environ.get('ADMIN_EMAIL', 'admin@faceattend.com')
            admin = User.query.filter_by(email=admin_email).first()

            if not admin:
                admin_password = os.environ.get('ADMIN_PASSWORD')
                generated = False
                if not admin_password:
                    admin_password = secrets.token_urlsafe(9)
                    generated = True

                admin = User(
                    fullname='Admin',
                    email=admin_email,
                    password=generate_password_hash(admin_password),
                    role='admin',
                    is_approved=True,
                    created_at=datetime.utcnow()
                )
                db.session.add(admin)
                db.session.commit()

                print(f"Admin user created: {admin_email}")
                if generated:
                    print(f"Generated password (SAVE THIS NOW): {admin_password}")
                else:
                    print("Password set from ADMIN_PASSWORD environment variable.")
            else:
                print(f"Admin user already exists: {admin_email}")

            if os.environ.get('SEED_DEMO_DATA', '').lower() == 'true':
                teacher = User.query.filter_by(role='teacher').first()
                if not teacher:
                    teacher_password = secrets.token_urlsafe(9)
                    teacher = User(
                        fullname='Sample Teacher',
                        email='teacher@school.com',
                        password=generate_password_hash(teacher_password),
                        role='teacher',
                        is_approved=True,
                        created_at=datetime.utcnow()
                    )
                    db.session.add(teacher)
                    db.session.commit()
                    print(f"Sample teacher created: teacher@school.com / {teacher_password}")
                else:
                    print("Teacher user already exists")

                if Course.query.count() == 0:
                    teacher = User.query.filter_by(role='teacher').first() or admin
                    sample_course = Course(
                        course_code='CS101',
                        course_name='Introduction to Computer Science',
                        description='This is a sample course created by the system.',
                        schedule='Monday and Wednesday 10:00 AM - 11:30 AM',
                        teacher_id=teacher.id,
                        is_active=True,
                        created_at=datetime.utcnow()
                    )
                    db.session.add(sample_course)
                    db.session.commit()
                    print("Sample course created: CS101")
                else:
                    print("Courses already exist")
            else:
                print("Demo data seeding skipped (set SEED_DEMO_DATA=true to enable)")

            print("=" * 50)
            print("Database setup complete!")
            print(f"Current students: {Student.query.count()}")
            print(f"Current users: {User.query.count()}")
            print(f"Current courses: {Course.query.count()}")
            print("=" * 50)

        except Exception as e:
            print(f"Database initialization error: {str(e)}")
            traceback.print_exc()
            db.session.rollback()


if __name__ == '__main__':
    print("Starting Face Attendance System...")
    print("Server running at: http://localhost:5000")

    init_db()

    app.run(debug=True, host='0.0.0.0', port=5000)