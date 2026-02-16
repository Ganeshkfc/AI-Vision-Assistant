[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
# Optimized extensions
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
# CRITICAL: Expanded exclusions to stop the runner from hanging on tests
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples, kivy/tests, kivy/tools, site-packages/numpy/tests
source.exclude_patterns = license, README*, *test*, *Test*, tests/*, */tests/*, *.pyc, *.pyo

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

# Speed up build by whitelisting only essential core libs
android.whitelist = sqlite3, libffi, openssl

[buildozer]
log_level = 1
warn_on_root = 1
