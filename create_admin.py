#!/usr/bin/env python3
"""
Create Admin User for FaceAttend
Run: python create_admin.py
"""

import os
import sys
import importlib.util
from datetime import datetime
from werkzeug.security import generate_password_hash
from sqlalchemy import text  # ← ADD THIS IMPORT


# ─────────────────────────────────────────────
# Find the main app file
# ─────────────────────────────────────────────

def find_app_file():
    """Find the main Flask application file"""
    possible_names = ['app.py', 'server.py', 'main.py', 'application.py', 'run.py']
    
    for name in possible_names:
        if os.path.exists(name):
            return name
    
    for file in os.listdir('.'):
        if file.endswith('.py') and file not in ['create_admin.py', 'create_admin.pyc']:
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if 'Flask(' in content or 'Flask(__name__)' in content:
                        return file
            except:
                continue
    
    return None


# ─────────────────────────────────────────────
# Import the app dynamically
# ─────────────────────────────────────────────

def import_app_module(file_name):
    """Dynamically import the app module"""
    module_name = file_name.replace('.py', '')
    
    try:
        return __import__(module_name)
    except ImportError:
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"❌ Dynamic import failed: {str(e)}")
            raise


# ─────────────────────────────────────────────
# Create admin user
# ─────────────────────────────────────────────

def create_admin_interactive():
    """Create admin user interactively"""
    print("\n" + "=" * 60)
    print("🔐 FaceAttend - Interactive Admin Creator")
    print("=" * 60)
    
    # Find the app file
    app_file = find_app_file()
    
    if not app_file:
        print("\n❌ Error: Could not find Flask application file!")
        print("   Looking for: app.py, server.py, main.py, etc.")
        sys.exit(1)
    
    print(f"\n📁 Found application file: {app_file}")
    
    try:
        module = import_app_module(app_file)
        
        app = getattr(module, 'app', None)
        db = getattr(module, 'db', None)
        User = getattr(module, 'User', None)
        
        if not app or not db or not User:
            print("\n❌ Error: Could not find required objects in the module!")
            sys.exit(1)
        
        print(f"✅ Successfully imported: {app_file}")
        print(f"   Database: {app.config.get('SQLALCHEMY_DATABASE_URI', 'Unknown')}")
        
    except Exception as e:
        print(f"\n❌ Error importing application: {str(e)}")
        sys.exit(1)
    
    with app.app_context():
        print("\n" + "-" * 60)
        print("📊 Creating database tables if they don't exist...")
        
        # ───── FIX: Create tables with proper SQLAlchemy 2.0 syntax ─────
        try:
            # Enable foreign keys using text() - FIXED
            db.session.execute(text('PRAGMA foreign_keys=ON'))
            db.session.commit()
            
            # Create all tables
            db.create_all()
            print("✅ Database tables created successfully!")
            
            # Check if tables exist now
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"   Tables found: {', '.join(tables) if tables else 'None'}")
            
        except Exception as e:
            print(f"❌ Error creating tables: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # ───── Now check for existing admin ─────
        try:
            existing_admin = User.query.filter_by(role='admin').first()
            if existing_admin:
                print("\n⚠️  An admin already exists!")
                print(f"   Name: {existing_admin.fullname}")
                print(f"   Email: {existing_admin.email}")
                print("")
                
                choice = input("Do you want to create another admin? (y/n): ").strip().lower()
                if choice != 'y':
                    print("\n✅ Using existing admin.")
                    print(f"   Login: {existing_admin.email}")
                    return
        except Exception as e:
            print(f"⚠️  Could not check for existing admin: {str(e)}")
        
        print("\n📝 Enter admin details (press Enter for defaults):")
        print("-" * 40)
        
        # Get email
        email = input("Admin Email : ").strip()
        if not email:
            email = "email"
        
        # Check if email exists
        try:
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                print(f"\n⚠️  User '{email}' already exists!")
                print(f"   Name: {existing_user.fullname}")
                print(f"   Role: {existing_user.role}")
                
                if existing_user.role == 'admin':
                    print(f"✅ {email} is already an admin!")
                    return
                
                choice = input(f"Make {existing_user.fullname} an admin? (y/n): ").strip().lower()
                if choice == 'y':
                    existing_user.role = 'admin'
                    existing_user.is_approved = True
                    db.session.commit()
                    print(f"\n✅ {existing_user.fullname} is now an admin!")
                    return
                else:
                    print("\nExiting...")
                    return
        except Exception as e:
            print(f"⚠️  Could not check for existing user: {str(e)}")
        
        # Get full name
        fullname = input("Admin Full Name : ").strip()
        if not fullname:
            fullname = "fullname"
        
        # Get password
        while True:
            password = input("Admin Password [min 6 chars]: ").strip()
            if not password:
                password = "Admin123!"
                print(f"Using default password: {password}")
                print("⚠️  CHANGE THIS PASSWORD IMMEDIATELY AFTER LOGIN!")
                break
            
            if len(password) < 6:
                print("❌ Password must be at least 6 characters!")
                continue
            
            confirm = input("Confirm Password: ").strip()
            if password != confirm:
                print("❌ Passwords don't match!")
                continue
            
            break
        
        # Create the admin user
        try:
            admin = User(
                fullname=fullname,
                email=email,
                password=generate_password_hash(password),
                role='admin',
                is_approved=True,
                created_at=datetime.utcnow()
            )
            
            db.session.add(admin)
            db.session.commit()
            
            print("\n" + "=" * 60)
            print("✅ ADMIN USER CREATED SUCCESSFULLY!")
            print("=" * 60)
            print(f"   Name: {fullname}")
            print(f"   Email: {email}")
            print(f"   Password: {password}")
            print(f"   Role: admin")
            print("=" * 60)
            print("\n🔗 Login at:")
            print("   http://localhost:5000/login")
            print("   http://localhost:5000/dashboard")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Error creating admin: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("\nUsage:")
            print("  python create_admin.py          # Interactive mode")
            sys.exit(0)
    
    print("\n" + "─" * 60)
    print("🔐 FaceAttend - Admin Creation Tool")
    print("─" * 60)
    print("\nChoose an option:")
    print("  1. Create admin interactively (recommended)")
    print("  2. Exit")
    print("─" * 60)
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == '1':
        create_admin_interactive()
    else:
        print("\nExiting...")
        sys.exit(0)