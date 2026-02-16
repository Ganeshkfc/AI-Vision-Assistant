[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# EXCLUSIONS: Stripping out all test folders to prevent the 30,000 file hang
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples, Lib/test, Lib/unittest
source.exclude_patterns = license, README*, *.pyc, *.pyo, */test/*, */tests/*, site-packages/numpy/tests

version = 1.0
requirements = python3, kivy==2.3.0, cython==0.29.33, camera4kivy, gestures4kivy, pyjnius, android, sqlite3

# Android Configuration
android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.allow_backup = False

# THE "MAGIC" FIX: This stops the runner from attempting to compile 28,000+ internal files
android.no_byte_compile_python = True

# Essential libraries only
android.whitelist = sqlite3, libffi, openssl

[buildozer]
log_level = 1
warn_on_root = 1
