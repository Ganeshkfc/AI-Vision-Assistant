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

# ADDED cython to requirements for a smoother build
requirements = python3,hostpython3,kivy,pyjnius,camera4kivy,gestures4kivy,android,numpy,pillow,sqlite3,tflite-runtime,cython

android.permissions = CAMERA, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, INTERNET

android.api = 34
# UPDATED to 24 (Required for modern NumPy/TFLite support)
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True

# ONLY arm64-v8a to save 50% of build time and storage space
android.archs = arm64-v8a
android.allow_backup = False

# Critical to prevent runner timeout
android.no_byte_compile_python = True

[buildozer]
log_level = 1
warn_on_root = 1
