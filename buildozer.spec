[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# Aggressive exclusions to keep the file count low
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples, Lib/test, Lib/unittest
source.exclude_patterns = license, README*, *.pyc, *.pyo, */test/*, */tests/*, */unittest/*, */lib-dynload/_test*, */__pycache__/*

version = 1.0

# PINNED versions to ensure compatibility with Android NDK 25b
requirements = python3, hostpython3, kivy, pyjnius, camera4kivy, gestures4kivy, android, numpy==1.26.4, pillow, sqlite3, tflite-runtime, cython==0.29.37

android.permissions = CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, INTERNET

android.api = 34
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True

# ONLY arm64-v8a to save 50% of build time and storage space
android.archs = arm64-v8a
android.allow_backup = False

# Critical to prevent runner timeout during NumPy/TFLite install
android.no_byte_compile_python = True

[buildozer]
log_level = 2
warn_on_root = 1
