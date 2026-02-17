[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
source.exclude_dirs = tests, bin, venv, .venv, .git, .github
version = 1.0

# UPDATED REQUIREMENTS: Added specific Cython version and fixed dependencies
requirements = python3, kivy==2.3.0, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, tflite-runtime, cython==0.29.33

orientation = portrait

# Android specific
android.api = 33
android.minapi = 21
# Using NDK 25b is correct for API 33, but 23b is often more stable for Kivy
android.ndk = 25b
android.ndk_api = 21
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE
android.private_storage = True
android.accept_sdk_license = True

# CRITICAL FOR CAMERA4KIVY
android.enable_androidx = True

android.archs = arm64-v8a
android.no_byte_compile_python = True

# CHANGED: 'develop' branch is highly recommended over 'master' for recent Android fixes
p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 1
