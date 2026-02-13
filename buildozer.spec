[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = com.ganesh
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

version = 0.1

# PINNED: Requirements for Camera4Kivy and TFLite
requirements = python3, kivy==2.3.0, hostpython3, numpy, tflite-runtime, camera4kivy, pillow, jnius, sh, gestures4kivy

# MANDATORY for Camera4Kivy
p4a.branch = master
p4a.hook = camera4kivy

# PINNED: Stable Cython for Kivy 2.3.0
android.pip_dependencies = cython==0.29.33

orientation = portrait
fullscreen = 0

# --- UPDATED SECTION ---
# Fixed: android.sdk is deprecated if used under [app], but we keep it 33 for consistency
android.api = 33
# Increased minapi to 24 to resolve POSIX header conflicts with Python 3.11
android.minapi = 24
android.ndk = 25b
# -----------------------

android.accept_sdk_license = True

# Permissions for Android 13+
android.permissions = CAMERA, INTERNET, READ_MEDIA_IMAGES, READ_MEDIA_VIDEO, WAKE_LOCK

android.wakelock = True
android.archs = arm64-v8a

# --- CRITICAL ERROR FIXES ---
# Fix for: "error: implicit declaration of function 'endgrent' is invalid in C99"
# This tells the compiler to ignore the missing function declaration that caused the crash.
android.extra_cflags = "-Wno-implicit-function-declaration"

# Optional but recommended: ensures the build system uses the correct toolchain settings
android.ndk_api = 24
# ----------------------------

[buildozer]
log_level = 2
warn_on_root = 1
