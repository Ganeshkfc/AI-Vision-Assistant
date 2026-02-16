[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# EXCLUSIONS: Keep this clean to prevent the 30,000 file hang
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples, Lib/test, Lib/unittest
source.exclude_patterns = license, README*, *.pyc, *.pyo, */test/*, */tests/*

version = 1.0
# FIXED REQUIREMENTS: Let Buildozer choose the best Kivy/Cython for the NDK
requirements = python3, kivy, camera4kivy, gestures4kivy, pyjnius, android, numpy, pillow, sqlite3

android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.allow_backup = False

# SKIP COMPILATION: Critical for GitHub Runners
android.no_byte_compile_python = True

[buildozer]
log_level = 1
warn_on_root = 1
