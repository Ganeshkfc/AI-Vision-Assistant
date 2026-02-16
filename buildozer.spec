[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# AGGRESSIVE EXCLUSIONS: This stops the 30,000+ file processing loop
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples, Lib/test, Lib/unittest
source.exclude_patterns = license, README*, *.pyc, *.pyo, */test/*, */tests/*, */unittest/*, */lib-dynload/_test*

version = 1.0

# REQUIREMENTS: hostpython3 is mandatory here to handle the tflite/numpy compilation
requirements = python3, hostpython3, kivy, pyjnius, camera4kivy, gestures4kivy, android, numpy==1.26.4, pillow, sqlite3, tflite-runtime

android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.allow_backup = False

# THE CRITICAL FLAG: Stops the runner from trying to bytecode-compile the library
android.no_byte_compile_python = True

[buildozer]
log_level = 1
warn_on_root = 1
