[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# AGGRESSIVE EXCLUSIONS: Keeps the build fast and prevents the 30,000 file hang
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples, Lib/test, Lib/unittest
source.exclude_patterns = license, README*, *.pyc, *.pyo, */test/*, */tests/*, */unittest/*, */lib-dynload/_test*, */__pycache__/*

version = 1.0

# REQUIREMENTS: These are exactly tuned for your NDK version
requirements = python3, hostpython3, kivy, pyjnius, camera4kivy, gestures4kivy, android, numpy==1.26.4, pillow, sqlite3, tflite-runtime

# PERMISSIONS: (CRITICAL FIX) The app needs these declared to the OS
android.permissions = CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, INTERNET

android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.allow_backup = False

# SKIP COMPILATION: This prevents the runner timeout
android.no_byte_compile_python = True

[buildozer]
log_level = 1
warn_on_root = 1
