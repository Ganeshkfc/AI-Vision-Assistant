[app]

# (str) Title of your application
title = AI Vision Assistant

# (str) Package name
package.name = aivisionassistant

# (str) Package domain (needed for android packaging)
package.domain = org.test

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,tflite

# (list) Application requirements
# Added hostpython3 and cython to ensure smooth compilation in GitHub Actions
requirements = python3, kivy==2.3.0, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, tflite-runtime, cython, hostpython3

# (str) Supported orientation (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (list) Permissions
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK will support.
android.minapi = 21

# (str) Android NDK version to use
# android.ndk = 25b

# (bool) If True, then automatically accept SDK license agreements.
# THIS IS THE KEY FIX FOR AUTOMATION
android.accept_sdk_license = True

# (list) The Android archs to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
# Building for both modern 64-bit and older 32-bit devices
android.archs = arm64-v8a

# (bool) Allow backup
android.allow_backup = True

# (str) python-for-android branch to use
# 'develop' is highly recommended for API 33+ and tflite compatibility
p4a.branch = develop

# (list) List of service to declare
#services = NAME:ENTRYPOINT_PY

[buildozer]

# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = False, 1 = True)
warn_on_root = 1

# (str) Path to build artifact storage, cache, and prefix
# build_dir = ./.buildozer

# (str) Filename (without extension) of the bin build by buildozer
# bin_dir = ./bin
