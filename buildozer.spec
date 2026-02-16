[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
source.exclude_dirs = tests, bin, venv, .venv, .git, .github

version = 1.0

# Pinning versions to prevent build failures
requirements = python3, hostpython3, kivy, pyjnius, camera4kivy, gestures4kivy, android, numpy==1.26.4, pillow, sqlite3, tflite-runtime, cython==0.29.37

android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True
android.archs = arm64-v8a
android.no_byte_compile_python = True

[buildozer]
log_level = 2
warn_on_root = 1
