[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
source.exclude_dirs = tests, bin, venv, .venv, .git, .github
version = 1.0

# Requirements: camera4kivy needs gestures4kivy and specific android libraries
requirements = python3, kivy==2.3.0, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, sqlite3, tflite-runtime

orientation = portrait
android.api = 33
android.minapi = 21
android.ndk = 25b
android.ndk_api = 21
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE
android.private_storage = True
android.accept_sdk_license = True

# REQUIRED for Camera4Kivy
android.enable_androidx = True
p4a.hook = camerax_provider/gradle_options.py

android.archs = arm64-v8a
android.no_byte_compile_python = True
p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1
