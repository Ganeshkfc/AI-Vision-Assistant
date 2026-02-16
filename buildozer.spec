[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
# Optimized extensions to include model files explicitly
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples

version = 1.0
orientation = portrait
fullscreen = 0

# Requirements: Use 'pyjnius' instead of 'jnius' and pinned cython
requirements = python3, kivy==2.3.0, cython==3.0.11, camera4kivy, gestures4kivy, pillow, pyjnius, numpy, tflite-runtime, android, sqlite3

# Essential hook for CameraX provider (Ensure this folder exists in your repo)
p4a.hook = camerax_provider/gradle_options.py

android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True

# Permissions: Standard set for AI/Vision apps
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, WAKE_LOCK, RECORD_AUDIO
android.wakelock = True

# OPTIMIZATION: Build only for 64-bit to prevent 'tflite' compilation timeouts
android.archs = arm64-v8a

# Optimization flags to prevent common C-compilation errors
android.extra_cflags = -Wno-error=implicit-function-declaration -fno-lto

[buildozer]
# Changed to level 1 to prevent log buffer overflow on GitHub
log_level = 1
warn_on_root = 1
