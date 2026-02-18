[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.test
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite

# PINNED version of Cython is critical for Kivy 2.3.0
requirements = python3, kivy==2.3.0, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, tflite-runtime, cython==0.29.33, hostpython3

orientation = portrait
fullscreen = 0
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

android.api = 33
android.minapi = 21

# Automatically accept licenses to prevent the build from hanging
android.accept_sdk_license = True

# Build for modern 64-bit devices (most common now)
android.archs = arm64-v8a
android.allow_backup = True

# 'develop' branch is necessary for modern Android features and tflite
p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 1
