[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples
android.exclude_src = bin/*, .google*, tests/*, **/test/*, **/tests/*, **/idle_test/*, **/lib-tk/*, **/lib2to3/*, **/doc/*, **/docs/*, **/examples/*

version = 1.0
orientation = portrait
fullscreen = 0

# Requirements: Use standard recipes where possible. 
# Added sqlite3 for local data storage and ensured cython is pinned for compatibility.
requirements = python3, kivy==2.3.0, cython==3.0.11, camera4kivy, gestures4kivy, pillow, jnius, numpy, tflite-runtime, android, sqlite3

# Essential hook for CameraX provider
p4a.hook = camerax_provider/gradle_options.py

# Target API 34 is correct for current Play Store requirements
android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True

# Updated Permissions: Added MANAGE_EXTERNAL_STORAGE for newer Android versions 
# if you need to save images to public folders, and kept existing essential ones.
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, WAKE_LOCK, RECORD_AUDIO

android.wakelock = True
# Building for both 64-bit and 32-bit ensures wider device compatibility
android.archs = arm64-v8a, armeabi-v7a

# Optimization flags to prevent common C-compilation errors during build
android.extra_cflags = -Wno-error=implicit-function-declaration -fno-lto

[buildozer]
log_level = 2
warn_on_root = 1
