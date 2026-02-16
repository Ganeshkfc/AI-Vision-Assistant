[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# EXCLUSIONS: Critical to prevent the 30,000 file compilation loop
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples, Lib/test, Lib/unittest
source.exclude_patterns = license, README*, *.pyc, *.pyo, */test/*, */tests/*

version = 1.0

# REQUIREMENTS: Added 'hostpython3' to fix the missing jnius.c error
requirements = python3, hostpython3, kivy, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, sqlite3

android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.allow_backup = False

# SKIP COMPILATION: Keeps the build within GitHub's time limits
android.no_byte_compile_python = True

[buildozer]
log_level = 1
warn_on_root = 1
