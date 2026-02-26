[app]

# (str) Title of your application
title = AI Vision Assistant

# (str) Package name
package.name = aivisionassistant

# (str) Package domain (needed for android packaging)
package.domain = org.aivision

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,tflite,txt,java

# (list) Directory to exclude
source.exclude_dirs = tests, bin, venv, .venv, .git, .github

# (str) Application versioning (method 1)
version = 1.0

# --- REQUIREMENTS ---
# FIXED: Added cython==0.29.33 to fix the "jnius.c not found" error.
# FIXED: Moved tflite-runtime to the end of the list.
requirements = python3, kivy==2.3.0, cython==0.29.33, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, tflite-runtime

# (str) Supported orientation (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# --- ANDROID SETTINGS ---
# UPDATED: API 34 is now required for Google Play and is more stable.
android.api = 34
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24

# --- PERMISSIONS & FEATURES ---
# FIXED: Cleaned up permissions for modern Android (API 33+).
android.permissions = CAMERA, VIBRATE, FLASHLIGHT, INTERNET
android.features = android.hardware.camera.flash

android.private_storage = True
android.accept_sdk_license = True

# --- CAMERA & TFLITE INTEGRATION ---
android.enable_androidx = True
p4a.hook = camerax_provider/gradle_options.py
android.add_src = camerax_provider/camerax_src

# FIXED: Removed hardcoded sdk_path and ndk_path. 
# Buildozer/GitHub Actions will find these automatically. 
# Hardcoding them causes "File not found" errors during build.

# --- GRADLE & ARCHITECTURE ---
android.gradle_options = "packagingOptions { pickFirst 'lib/arm64-v8a/libc++_shared.so'; pickFirst 'lib/armeabi-v7a/libc++_shared.so'; pickFirst 'lib/x86/libc++_shared.so'; pickFirst 'lib/x86_64/libc++_shared.so'; exclude 'META-INF/INDEX.LIST' }"
android.add_gradle_repositories = "https://maven.google.com"

# UPDATED: Added armeabi-v7a for better device compatibility.
android.archs = arm64-v8a, armeabi-v7a

android.no_byte_compile_python = True
p4a.branch = master

[buildozer]
# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2
warn_on_root = 1
