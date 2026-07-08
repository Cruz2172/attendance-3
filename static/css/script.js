// script.js - Facial Recognition Attendance System

console.log('Script loaded successfully');

// ─────────────────────────────────────────────
// Global Functions
// ─────────────────────────────────────────────

function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta) return meta.content;
    const input = document.querySelector('input[name="csrf_token"]');
    if (input) return input.value;
    return '';
}

function showLoading(element, message = 'Processing...') {
    if (element) {
        element.disabled = true;
        element.innerHTML = `<i class="fas fa-spinner fa-spin me-2"></i> ${message}`;
    }
}

function hideLoading(element, originalText) {
    if (element) {
        element.disabled = false;
        element.innerHTML = originalText || 'Submit';
    }
}

// ─────────────────────────────────────────────
// Sound Effects
// ─────────────────────────────────────────────

const SoundEffects = {
    _context: null,
    _sounds: {},
    
    init: function() {
        try {
            this._context = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) {
            console.warn('Web Audio API not supported');
            this._context = null;
        }
    },
    
    playSuccess: function() {
        if (!this._context) return;
        try {
            if (this._context.state === 'suspended') {
                this._context.resume();
            }
            const oscillator = this._context.createOscillator();
            const gainNode = this._context.createGain();
            oscillator.connect(gainNode);
            gainNode.connect(this._context.destination);
            oscillator.frequency.setValueAtTime(880, this._context.currentTime);
            oscillator.frequency.setValueAtTime(1108.73, this._context.currentTime + 0.1);
            oscillator.frequency.setValueAtTime(1318.51, this._context.currentTime + 0.2);
            gainNode.gain.setValueAtTime(0, this._context.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.3, this._context.currentTime + 0.01);
            gainNode.gain.linearRampToValueAtTime(0.2, this._context.currentTime + 0.3);
            gainNode.gain.exponentialRampToValueAtTime(0.001, this._context.currentTime + 0.5);
            oscillator.type = 'sine';
            oscillator.start(this._context.currentTime);
            oscillator.stop(this._context.currentTime + 0.5);
        } catch (e) {
            console.warn('Could not play success sound:', e);
        }
    },
    
    playWarning: function() {
        if (!this._context) return;
        try {
            if (this._context.state === 'suspended') {
                this._context.resume();
            }
            const oscillator = this._context.createOscillator();
            const gainNode = this._context.createGain();
            oscillator.connect(gainNode);
            gainNode.connect(this._context.destination);
            oscillator.frequency.setValueAtTime(300, this._context.currentTime);
            oscillator.frequency.setValueAtTime(250, this._context.currentTime + 0.2);
            gainNode.gain.setValueAtTime(0, this._context.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.2, this._context.currentTime + 0.01);
            gainNode.gain.exponentialRampToValueAtTime(0.001, this._context.currentTime + 0.3);
            oscillator.type = 'sawtooth';
            oscillator.start(this._context.currentTime);
            oscillator.stop(this._context.currentTime + 0.3);
        } catch (e) {
            console.warn('Could not play warning sound:', e);
        }
    },
    
    playAlreadyMarked: function() {
        if (!this._context) return;
        try {
            if (this._context.state === 'suspended') {
                this._context.resume();
            }
            const oscillator = this._context.createOscillator();
            const gainNode = this._context.createGain();
            oscillator.connect(gainNode);
            gainNode.connect(this._context.destination);
            oscillator.frequency.setValueAtTime(660, this._context.currentTime);
            oscillator.frequency.setValueAtTime(523.25, this._context.currentTime + 0.15);
            gainNode.gain.setValueAtTime(0, this._context.currentTime);
            gainNode.gain.linearRampToValueAtTime(0.15, this._context.currentTime + 0.01);
            gainNode.gain.exponentialRampToValueAtTime(0.001, this._context.currentTime + 0.3);
            oscillator.type = 'sine';
            oscillator.start(this._context.currentTime);
            oscillator.stop(this._context.currentTime + 0.3);
        } catch (e) {
            console.warn('Could not play already marked sound:', e);
        }
    }
};

// Initialize sounds when the page loads
document.addEventListener('DOMContentLoaded', function() {
    SoundEffects.init();
});

// ─────────────────────────────────────────────
// Camera Utilities
// ─────────────────────────────────────────────

const CameraUtils = {
    isSupported: function() {
        return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    },

    start: function(videoElement) {
        return new Promise(function(resolve, reject) {
            if (!videoElement) {
                reject(new Error('Video element not found'));
                return;
            }

            navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            })
            .then(function(stream) {
                videoElement.srcObject = stream;
                videoElement.onloadedmetadata = function() {
                    videoElement.play()
                        .then(function() {
                            resolve(stream);
                        })
                        .catch(function(err) {
                            reject(err);
                        });
                };
                videoElement.onerror = function(err) {
                    reject(err);
                };
            })
            .catch(function(error) {
                reject(error);
            });
        });
    },

    stop: function(stream) {
        if (stream) {
            stream.getTracks().forEach(function(track) {
                track.stop();
            });
        }
    },

    capture: function(videoElement, canvasElement) {
        if (!videoElement || !canvasElement) {
            console.error('Video or canvas element missing');
            return null;
        }

        if (videoElement.readyState < 2) {
            console.warn('Video not ready for capture');
            return null;
        }

        try {
            const context = canvasElement.getContext('2d');
            const width = videoElement.videoWidth || 640;
            const height = videoElement.videoHeight || 480;
            
            canvasElement.width = width;
            canvasElement.height = height;
            context.drawImage(videoElement, 0, 0, width, height);
            
            return canvasElement.toDataURL('image/jpeg', 0.8);
        } catch (e) {
            console.error('Capture error:', e);
            return null;
        }
    }
};

// ─────────────────────────────────────────────
// Page Initialization
// ─────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded - initializing page');
    
    const path = window.location.pathname;
    console.log('Current path:', path);
    
    // Check for student dashboard
    if (path.includes('student/dashboard')) {
        if (document.getElementById('video') && document.getElementById('startCameraBtn')) {
            initStudentDashboardCamera();
        }
    }
    // Check for enrollment page
    else if (path.includes('enroll') && !path.includes('token')) {
        if (document.getElementById('enrollVideo') || document.getElementById('startCameraBtn')) {
            initEnrollPage();
        }
    } 
    // Check for attendance page
    else if (path.includes('mark_attendance') || path.includes('recognize')) {
        if (document.getElementById('attendanceVideo') || document.getElementById('startAttendanceBtn')) {
            initAttendancePage();
        }
    } 
    // Check for dashboard
    else if (path.includes('dashboard')) {
        initDashboardPage();
    } 
    // Check for login
    else if (path.includes('login')) {
        initLoginPage();
    } 
    // Check for register
    else if (path.includes('register')) {
        initRegisterPage();
    }
});

// ─────────────────────────────────────────────
// Student Dashboard Camera
// ─────────────────────────────────────────────

function initStudentDashboardCamera() {
    console.log('Initializing student dashboard camera');
    
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startBtn = document.getElementById('startCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const stopBtn = document.getElementById('stopCameraBtn');
    const resetBtn = document.getElementById('resetBtn');
    const submitBtn = document.getElementById('submitBtn');
    const faceDataInput = document.getElementById('face_data');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const cameraLoading = document.getElementById('cameraLoading');
    const captureStatus = document.getElementById('captureStatus');
    const capturedPreview = document.getElementById('capturedPreview');
    const form = document.getElementById('enrollForm');

    // Check if we're on the dashboard with camera
    if (!video || !startBtn) {
        console.log('Dashboard camera elements not found, skipping initialization');
        return;
    }

    let stream = null;
    let isRunning = false;
    let isCaptured = false;

    // Check camera support
    if (!CameraUtils.isSupported()) {
        if (captureStatus) {
            captureStatus.innerHTML = '<div class="alert alert-danger">Camera not supported in this browser</div>';
        }
        if (startBtn) startBtn.disabled = true;
        return;
    }

    // Start Camera
    if (startBtn) {
        startBtn.addEventListener('click', function() {
            console.log('Dashboard: Start camera clicked');
            
            isCaptured = false;
            isRunning = false;
            if (captureBtn) captureBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = true;
            if (resetBtn) resetBtn.disabled = true;
            if (submitBtn) submitBtn.disabled = true;
            if (capturedPreview) capturedPreview.innerHTML = '';
            
            if (captureStatus) {
                captureStatus.innerHTML = '<div class="alert alert-info">Requesting camera access...</div>';
            }
            startBtn.disabled = true;
            if (cameraLoading) cameraLoading.style.display = 'block';
            if (videoPlaceholder) videoPlaceholder.style.display = 'none';

            CameraUtils.start(video)
                .then(function(mediaStream) {
                    stream = mediaStream;
                    video.style.display = 'block';
                    if (videoPlaceholder) videoPlaceholder.style.display = 'none';
                    if (cameraLoading) cameraLoading.style.display = 'none';
                    
                    isRunning = true;
                    if (captureBtn) captureBtn.disabled = false;
                    if (stopBtn) stopBtn.disabled = false;
                    startBtn.disabled = true;
                    if (resetBtn) resetBtn.disabled = true;
                    
                    if (captureStatus) {
                        captureStatus.innerHTML = '<div class="alert alert-success">Camera ready! Click "Capture Face" to take a photo.</div>';
                    }
                    console.log('Dashboard: Camera started successfully');
                })
                .catch(function(error) {
                    console.error('Dashboard: Camera error:', error);
                    if (cameraLoading) cameraLoading.style.display = 'none';
                    if (videoPlaceholder) videoPlaceholder.style.display = 'block';
                    startBtn.disabled = false;
                    
                    let errorMsg = 'Camera error: ';
                    if (error.name === 'NotAllowedError') {
                        errorMsg += 'Please allow camera access in your browser settings.';
                    } else if (error.name === 'NotFoundError') {
                        errorMsg += 'No camera found on this device.';
                    } else if (error.name === 'NotReadableError') {
                        errorMsg += 'Camera is in use by another application.';
                    } else {
                        errorMsg += error.message;
                    }
                    
                    if (captureStatus) {
                        captureStatus.innerHTML = `<div class="alert alert-danger">${errorMsg}</div>`;
                    }
                });
        });
    }

    // Capture Face
    if (captureBtn) {
        captureBtn.addEventListener('click', function() {
            console.log('Dashboard: Capture clicked');
            
            if (!isRunning || !stream) {
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-warning">Camera is not running. Please start camera first.</div>';
                }
                return;
            }

            if (video.readyState < 2 || video.videoWidth === 0) {
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-warning">Camera not ready. Please wait.</div>';
                }
                return;
            }

            const imageData = CameraUtils.capture(video, canvas);
            
            if (imageData && imageData.length > 1000) {
                if (faceDataInput) faceDataInput.value = imageData;
                isCaptured = true;
                
                // Show preview
                if (capturedPreview) {
                    capturedPreview.innerHTML = `
                        <h6 class="text-success"><i class="fas fa-check-circle"></i> Face Captured!</h6>
                        <img src="${imageData}" alt="Captured Face" class="img-fluid" style="max-width: 200px; border-radius: 10px; border: 3px solid #28a745;">
                        <p class="text-muted small mt-1">You can now submit the form to enroll.</p>
                    `;
                }
                
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-success">Face captured successfully! Click "Enroll Student" to complete enrollment.</div>';
                }
                
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-user-check me-2"></i> Enroll Student';
                }
                if (captureBtn) captureBtn.disabled = true;
                if (resetBtn) resetBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = true;
                
                console.log('Dashboard: Face captured successfully');
            } else {
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-danger">Failed to capture image. Please try again.</div>';
                }
            }
        });
    }

    // Stop Camera
    if (stopBtn) {
        stopBtn.addEventListener('click', function() {
            console.log('Dashboard: Stop camera clicked');
            if (stream) {
                CameraUtils.stop(stream);
                stream = null;
                video.srcObject = null;
                video.style.display = 'none';
                if (videoPlaceholder) videoPlaceholder.style.display = 'block';
                
                isRunning = false;
                if (captureBtn) captureBtn.disabled = true;
                stopBtn.disabled = true;
                startBtn.disabled = false;
                if (resetBtn) resetBtn.disabled = true;
                
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-info">Camera stopped.</div>';
                }
            }
        });
    }

    // Reset
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            console.log('Dashboard: Reset clicked');
            if (faceDataInput) faceDataInput.value = '';
            isCaptured = false;
            if (capturedPreview) capturedPreview.innerHTML = '';
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-user-check me-2"></i> Enroll Student';
            }
            if (captureBtn) captureBtn.disabled = false;
            resetBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            if (captureStatus) {
                captureStatus.innerHTML = '<div class="alert alert-info">Reset. Click "Capture Face" to retake.</div>';
            }
        });
    }

    // Form Submit
    if (form) {
        form.addEventListener('submit', function(e) {
            console.log('Dashboard: Form submitted');
            
            if (!isCaptured || !faceDataInput || !faceDataInput.value || faceDataInput.value.length < 1000) {
                e.preventDefault();
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-danger">Please capture a face photo before enrolling.</div>';
                }
                return false;
            }
            
            // Show loading state
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Enrolling...';
            }
            
            console.log('Dashboard: Form submission approved');
            return true;
        });
    }

    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        if (stream) CameraUtils.stop(stream);
    });

    console.log('Student dashboard camera initialization complete');
}

// ─────────────────────────────────────────────
// Enroll Page
// ─────────────────────────────────────────────

function initEnrollPage() {
    console.log('Initializing enroll page');
    
    const video = document.getElementById('enrollVideo');
    const canvas = document.getElementById('canvas');
    const startBtn = document.getElementById('startCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const stopBtn = document.getElementById('stopCameraBtn');
    const resetBtn = document.getElementById('resetBtn');
    const submitBtn = document.getElementById('submitBtn');
    const faceDataInput = document.getElementById('face_data');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const cameraLoading = document.getElementById('cameraLoading');
    const captureStatus = document.getElementById('captureStatus');
    const form = document.getElementById('enrollForm');

    // Check if we're on the enroll page
    if (!video && !startBtn) {
        console.log('Not on enroll page, skipping initialization');
        return;
    }

    let stream = null;
    let isRunning = false;
    let isCaptured = false;

    // Check camera support
    if (!CameraUtils.isSupported()) {
        if (captureStatus) {
            captureStatus.innerHTML = '<div class="alert alert-danger">Camera not supported in this browser</div>';
        }
        if (startBtn) startBtn.disabled = true;
        return;
    }

    if (startBtn) {
        startBtn.addEventListener('click', function() {
            console.log('Start camera clicked');
            
            isCaptured = false;
            isRunning = false;
            if (captureBtn) captureBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = true;
            if (resetBtn) resetBtn.disabled = true;
            if (submitBtn) submitBtn.disabled = true;
            
            if (captureStatus) {
                captureStatus.innerHTML = '<div class="alert alert-info">Requesting camera access...</div>';
            }
            startBtn.disabled = true;
            if (cameraLoading) cameraLoading.style.display = 'flex';

            CameraUtils.start(video)
                .then(function(mediaStream) {
                    stream = mediaStream;
                    video.style.display = 'block';
                    if (videoPlaceholder) videoPlaceholder.style.display = 'none';
                    if (cameraLoading) cameraLoading.style.display = 'none';
                    
                    isRunning = true;
                    if (captureBtn) captureBtn.disabled = false;
                    if (stopBtn) stopBtn.disabled = false;
                    startBtn.disabled = true;
                    if (resetBtn) resetBtn.disabled = true;
                    
                    if (captureStatus) {
                        captureStatus.innerHTML = '<div class="alert alert-success">Camera ready! Click "Capture Face" to take a photo.</div>';
                    }
                    console.log('Camera started successfully');
                })
                .catch(function(error) {
                    console.error('Camera error:', error);
                    if (cameraLoading) cameraLoading.style.display = 'none';
                    startBtn.disabled = false;
                    
                    let errorMsg = 'Camera error: ';
                    if (error.name === 'NotAllowedError') {
                        errorMsg += 'Please allow camera access in your browser settings.';
                    } else if (error.name === 'NotFoundError') {
                        errorMsg += 'No camera found on this device.';
                    } else if (error.name === 'NotReadableError') {
                        errorMsg += 'Camera is in use by another application.';
                    } else {
                        errorMsg += error.message;
                    }
                    
                    if (captureStatus) {
                        captureStatus.innerHTML = `<div class="alert alert-danger">${errorMsg}</div>`;
                    }
                });
        });
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', function() {
            console.log('Capture clicked');
            
            if (!isRunning || !stream) {
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-warning">Camera is not running. Please start camera first.</div>';
                }
                return;
            }

            if (video.readyState < 2 || video.videoWidth === 0) {
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-warning">Camera not ready. Please wait.</div>';
                }
                return;
            }

            const imageData = CameraUtils.capture(video, canvas);
            
            if (imageData && imageData.length > 1000) {
                if (faceDataInput) faceDataInput.value = imageData;
                isCaptured = true;
                
                if (captureStatus) {
                    captureStatus.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle"></i> Face captured successfully!
                            <br><small>You can now click "Enroll Student" to complete enrollment.</small>
                        </div>
                    `;
                }
                
                if (submitBtn) {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-user-check me-2"></i> Enroll Student';
                }
                if (captureBtn) captureBtn.disabled = true;
                if (resetBtn) resetBtn.disabled = false;
                if (stopBtn) stopBtn.disabled = true;
                
                console.log('Face captured successfully');
            } else {
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-danger">Failed to capture image. Please try again.</div>';
                }
            }
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', function() {
            console.log('Stop camera clicked');
            if (stream) {
                CameraUtils.stop(stream);
                stream = null;
                video.srcObject = null;
                video.style.display = 'none';
                if (videoPlaceholder) videoPlaceholder.style.display = 'block';
                
                isRunning = false;
                if (captureBtn) captureBtn.disabled = true;
                stopBtn.disabled = true;
                startBtn.disabled = false;
                if (resetBtn) resetBtn.disabled = true;
                
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-info">Camera stopped.</div>';
                }
            }
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            console.log('Reset clicked');
            if (faceDataInput) faceDataInput.value = '';
            isCaptured = false;
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-user-check me-2"></i> Enroll Student';
            }
            if (captureBtn) captureBtn.disabled = false;
            resetBtn.disabled = true;
            if (stopBtn) stopBtn.disabled = false;
            if (captureStatus) {
                captureStatus.innerHTML = '<div class="alert alert-info">Reset. Click "Capture Face" to retake.</div>';
            }
        });
    }

    if (form) {
        form.addEventListener('submit', function(e) {
            console.log('Form submitted');
            
            if (!isCaptured || !faceDataInput || !faceDataInput.value || faceDataInput.value.length < 1000) {
                e.preventDefault();
                if (captureStatus) {
                    captureStatus.innerHTML = '<div class="alert alert-danger">Please capture a face photo before enrolling.</div>';
                }
                return false;
            }
            
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i> Enrolling...';
            }
            
            console.log('Form submission approved');
            return true;
        });
    }

    window.addEventListener('beforeunload', function() {
        if (stream) CameraUtils.stop(stream);
    });

    console.log('Enroll page initialization complete');
}

// ─────────────────────────────────────────────
// Attendance Page
// ─────────────────────────────────────────────

function initAttendancePage() {
    console.log('Initializing attendance page');
    
    const video = document.getElementById('attendanceVideo');
    const canvas = document.getElementById('canvas');
    const startBtn = document.getElementById('startAttendanceBtn');
    const stopBtn = document.getElementById('stopAttendanceBtn');
    const statusDiv = document.getElementById('attendanceStatus');
    const statusMsg = document.getElementById('statusMessage');
    const resultDiv = document.getElementById('attendanceResult');
    const attendanceSummary = document.getElementById('attendanceSummary');
    const presentCount = document.getElementById('presentCount');
    const totalCount = document.getElementById('totalCount');
    const absentCount = document.getElementById('absentCount');
    const attendanceList = document.getElementById('attendanceList');
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    const cameraLoading = document.getElementById('cameraLoading');

    // Check if we're on the attendance page
    if (!video && !startBtn) {
        console.log('Not on attendance page, skipping initialization');
        return;
    }

    let stream = null;
    let interval = null;
    let isRunning = false;
    let isProcessing = false;
    let markedStudents = [];
    let totalStudents = 0;

    function loadTotalStudents() {
        fetch('/students')
            .then(response => response.json())
            .then(data => {
                totalStudents = data.length;
                if (totalCount) totalCount.textContent = totalStudents;
            })
            .catch(error => console.error('Error loading students:', error));
    }
    loadTotalStudents();

    function loadTodayAttendance() {
        fetch('/attendance_stats')
            .then(response => response.json())
            .then(data => {
                if (presentCount) presentCount.textContent = data.present_today || 0;
                if (totalCount) totalCount.textContent = data.total_students || 0;
                if (absentCount) absentCount.textContent = (data.total_students || 0) - (data.present_today || 0);
            })
            .catch(error => console.error('Error loading attendance stats:', error));
    }

    function updateAttendanceSummary() {
        if (!attendanceSummary) return;
        
        attendanceSummary.style.display = 'block';
        
        if (presentCount) presentCount.textContent = markedStudents.length;
        if (totalCount) totalCount.textContent = totalStudents || markedStudents.length;
        if (absentCount) absentCount.textContent = (totalStudents || markedStudents.length) - markedStudents.length;
        
        if (attendanceList) {
            if (markedStudents.length === 0) {
                attendanceList.innerHTML = '<small class="text-muted">No students marked yet...</small>';
            } else {
                let html = '<div class="marked-student-list">';
                markedStudents.forEach((student, index) => {
                    html += `
                        <div class="marked-student-item">
                            <span>${index + 1}. ${student.name}</span>
                            <span class="text-muted small">${student.time}</span>
                        </div>
                    `;
                });
                html += '</div>';
                attendanceList.innerHTML = html;
                if (attendanceList.scrollTop) {
                    attendanceList.scrollTop = attendanceList.scrollHeight;
                }
            }
        }
    }

    if (!CameraUtils.isSupported()) {
        if (statusDiv) {
            statusDiv.className = 'alert alert-danger';
            if (statusMsg) statusMsg.textContent = 'Camera not supported';
        }
        if (startBtn) startBtn.disabled = true;
        return;
    }

    if (startBtn) {
        startBtn.addEventListener('click', function() {
            console.log('Start attendance camera');
            
            markedStudents = [];
            if (resultDiv) resultDiv.innerHTML = '';
            if (attendanceSummary) attendanceSummary.style.display = 'none';
            if (attendanceList) attendanceList.innerHTML = '<small class="text-muted">Waiting for first student...</small>';
            
            if (statusDiv) {
                statusDiv.className = 'alert alert-info';
                if (statusMsg) statusMsg.textContent = 'Starting camera...';
            }
            startBtn.disabled = true;
            if (cameraLoading) cameraLoading.style.display = 'flex';

            CameraUtils.start(video)
                .then(function(mediaStream) {
                    stream = mediaStream;
                    video.style.display = 'block';
                    if (videoPlaceholder) videoPlaceholder.style.display = 'none';
                    if (cameraLoading) cameraLoading.style.display = 'none';
                    
                    if (stopBtn) stopBtn.disabled = false;
                    
                    if (statusDiv) {
                        statusDiv.className = 'alert alert-success';
                        if (statusMsg) statusMsg.textContent = 'Camera ready! Looking for faces...';
                    }
                    
                    loadTodayAttendance();
                    isRunning = true;
                    
                    if (interval) clearInterval(interval);
                    interval = setInterval(function() {
                        if (isRunning && !isProcessing) {
                            recognizeFace();
                        }
                    }, 2000);
                    
                    console.log('Camera started, recognition active');
                })
                .catch(function(error) {
                    console.error('Camera error:', error);
                    if (cameraLoading) cameraLoading.style.display = 'none';
                    
                    if (statusDiv) {
                        statusDiv.className = 'alert alert-danger';
                        let msg = 'Camera error: ';
                        if (error.name === 'NotAllowedError') msg += 'Please allow camera access.';
                        else if (error.name === 'NotFoundError') msg += 'No camera found.';
                        else msg += error.message;
                        if (statusMsg) statusMsg.textContent = msg;
                    }
                    startBtn.disabled = false;
                });
        });
    }

    if (stopBtn) {
        stopBtn.addEventListener('click', function() {
            console.log('Stop camera');
            stopAttendance();
        });
    }

    function stopAttendance() {
        isRunning = false;
        
        if (interval) {
            clearInterval(interval);
            interval = null;
        }
        
        if (stream) {
            CameraUtils.stop(stream);
            stream = null;
            if (video) {
                video.srcObject = null;
                video.style.display = 'none';
                if (videoPlaceholder) videoPlaceholder.style.display = 'block';
            }
        }
        
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;
        
        if (statusDiv) {
            statusDiv.className = 'alert alert-info';
            if (statusMsg) statusMsg.textContent = `Attendance session ended. ${markedStudents.length} students marked.`;
        }
        
        console.log('Camera stopped');
    }

    function recognizeFace() {
        if (!stream || !video || !video.videoWidth || video.readyState < 2) {
            return;
        }
        
        isProcessing = true;
        
        if (statusDiv) {
            statusDiv.className = 'alert alert-info';
            if (statusMsg) statusMsg.textContent = '🔍 Scanning for faces...';
        }
        
        const imageData = CameraUtils.capture(video, canvas);
        
        if (!imageData) {
            isProcessing = false;
            return;
        }
        
        fetch('/recognize_face', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(function(response) {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.message || `Server responded with ${response.status}`);
                });
            }
            return response.json();
        })
        .then(function(data) {
            isProcessing = false;
            console.log('Recognition response:', data);
            
            if (data.success) {
                SoundEffects.playSuccess();
                
                const studentName = data.student_name || 'Student';
                const studentId = data.student_id || 'N/A';
                
                if (!markedStudents.find(s => s.id === studentId)) {
                    markedStudents.push({
                        id: studentId,
                        name: studentName,
                        time: data.time || new Date().toLocaleTimeString()
                    });
                    updateAttendanceSummary();
                }
                
                if (resultDiv) {
                    resultDiv.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle me-2"></i>
                            ✅ ${studentName} marked present! (${data.time || new Date().toLocaleTimeString()})
                        </div>
                    `;
                }
                
                if (statusDiv) {
                    statusDiv.className = 'alert alert-success';
                    if (statusMsg) statusMsg.textContent = `✅ ${studentName} marked present! Waiting for next student...`;
                }
                
                setTimeout(function() {
                    if (resultDiv && isRunning) {
                        resultDiv.innerHTML = '';
                    }
                }, 3000);
                
                loadTodayAttendance();
                
            } else if (data.message && (data.message.includes('already') || data.message.includes('Already'))) {
                SoundEffects.playAlreadyMarked();
                
                if (resultDiv) {
                    resultDiv.innerHTML = `
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle me-2"></i>
                            ${data.message}
                        </div>
                    `;
                }
                
                if (statusDiv) {
                    statusDiv.className = 'alert alert-info';
                    if (statusMsg) statusMsg.textContent = data.message;
                }
                
                setTimeout(function() {
                    if (resultDiv && isRunning) {
                        resultDiv.innerHTML = '';
                    }
                }, 2000);
                
            } else {
                SoundEffects.playWarning();
                
                if (statusDiv) {
                    statusDiv.className = 'alert alert-warning';
                    if (statusMsg) statusMsg.textContent = data.message || 'Face not recognized. Please look at the camera.';
                }
                
                if (resultDiv) {
                    resultDiv.innerHTML = `
                        <div class="alert alert-warning">
                            <i class="fas fa-user-slash me-2"></i>
                            ${data.message || 'Face not recognized. Please ensure you are enrolled and look directly at the camera.'}
                        </div>
                    `;
                }
            }
        })
        .catch(function(error) {
            isProcessing = false;
            console.error('Recognition error:', error);
            
            SoundEffects.playWarning();
            
            if (statusDiv) {
                statusDiv.className = 'alert alert-danger';
                if (statusMsg) statusMsg.textContent = error.message || 'Error processing. Please try again.';
            }
        });
    }

    window.addEventListener('beforeunload', function() {
        if (stream) CameraUtils.stop(stream);
        if (interval) clearInterval(interval);
    });
}

// ─────────────────────────────────────────────
// Dashboard Page
// ─────────────────────────────────────────────

function initDashboardPage() {
    console.log('Dashboard initialized');
    // Check if we're on the dashboard
    if (!document.querySelector('.stat-total')) {
        return;
    }
    setInterval(function() {
        if (document.hidden) return;
        fetch('/attendance_stats')
            .then(function(response) { return response.json(); })
            .then(function(data) {
                const total = document.querySelector('.stat-total');
                const present = document.querySelector('.stat-present');
                const absent = document.querySelector('.stat-absent');
                const rate = document.querySelector('.stat-rate');
                
                if (total && data.total_students !== undefined) total.textContent = data.total_students;
                if (present && data.present_today !== undefined) present.textContent = data.present_today;
                if (absent && data.absent_today !== undefined) absent.textContent = data.absent_today;
                if (rate && data.attendance_rate !== undefined) rate.textContent = data.attendance_rate + '%';
            })
            .catch(function(error) {
                console.error('Stats refresh error:', error);
            });
    }, 30000);
}

// ─────────────────────────────────────────────
// Login Page
// ─────────────────────────────────────────────

function initLoginPage() {
    console.log('Login page initialized');
}

// ─────────────────────────────────────────────
// Register Page
// ─────────────────────────────────────────────

function initRegisterPage() {
    console.log('Register page initialized');
    
    const password = document.getElementById('password');
    const confirm = document.getElementById('confirm_password');
    const passwordMismatch = document.getElementById('passwordMismatch');
    
    if (password && confirm) {
        // Real-time password validation
        confirm.addEventListener('input', function() {
            if (passwordMismatch) {
                if (this.value && password.value !== this.value) {
                    passwordMismatch.textContent = 'Passwords do not match';
                    passwordMismatch.className = 'text-danger';
                    this.classList.add('is-invalid');
                    password.classList.add('is-invalid');
                } else {
                    passwordMismatch.textContent = '';
                    this.classList.remove('is-invalid');
                    password.classList.remove('is-invalid');
                }
            }
        });
        
        // Password strength indicator
        password.addEventListener('input', function() {
            const strength = document.getElementById('passwordStrength');
            if (strength) {
                const val = this.value;
                let strengthText = '';
                let strengthClass = '';
                
                if (val.length === 0) {
                    strengthText = '';
                } else if (val.length < 6) {
                    strengthText = 'Weak';
                    strengthClass = 'text-danger';
                } else if (val.length < 10) {
                    strengthText = 'Medium';
                    strengthClass = 'text-warning';
                } else {
                    strengthText = 'Strong';
                    strengthClass = 'text-success';
                }
                
                strength.textContent = strengthText;
                strength.className = strengthClass;
            }
        });
    }
}

// ─────────────────────────────────────────────
// Global Error Handler
// ─────────────────────────────────────────────

window.addEventListener('error', function(e) {
    console.error('Global error caught:', e.message, e.filename, e.lineno);
});

console.log('Script initialization complete');