[app]
# (str) Title of your application
title = AI Vision Assistant

# (str) Package name
package.name = aivisionassistant

# (str) Package domain (needed for android packaging)
package.domain = org.aivision

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# (list) List of directory to exclude
source.exclude_dirs = tests, bin, venv, .venv, .git, .github

# (str) Application versioning
version = 1.0

# (list) Application requirements
# Note: Cython is a build tool and is handled by the GitHub Action. 
# Added 'hostpython3' and ensured 'camera4kivy' dependencies are present.
requirements = python3, kivy==2.3.0, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, sqlite3, tflite-runtime

# (str) Supported orientations
orientation = portrait

# ----------------------------------
# Android specific
# ----------------------------------

# (int) Target Android API, should be as high as possible.
# API 34 is the current requirement for Google Play Store.
android.api = 34

# (int) Minimum API your APK will support.
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 25b

# (int) Android NDK API to use. This is the minimum API your app will support.
android.ndk_api = 21

# (list) Android permissions
# Vision apps require Camera; Internet is usually needed for AI features.
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

# (bool) Use --private data storage (True) or --dir public storage (False)
android.private_storage = True

# (bool) Accept SDK license
android.accept_sdk_license = True

# (bool) Enable AndroidX support (Required for Camera4Kivy)
android.enable_androidx = True

# (list) The Android architectures to build for.
android.archs = arm64-v8a

# (bool) skip byte compile for .py files
android.no_byte_compile_python = True

# (str) python-for-android branch to use
# The 'master' branch is recommended for API 34 compatibility
p4a.branch = master

# ----------------------------------
# Buildozer section
# ----------------------------------

[buildozer]
# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = off, 1 = on)
warn_on_root = 1
