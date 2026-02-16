[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# CRITICAL: Exclude all test directories to prevent runner timeouts
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples, kivy/tests, kivy/tools
source.exclude_patterns = license, README*, *test*, *Test*, tests/*, */tests/*, *.pyc, *.pyo

version = 1.0
requirements = python3, kivy==2.3.0, cython==0.29.33, camera4kivy, gestures4kivy, pyjnius, android, sqlite3

android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.allow_backup = False

# Optimized whitelist for essential libraries only
android.whitelist = sqlite3, libffi, openssl

[buildozer]
log_level = 1
warn_on_root = 1
