[app]

# (str) Title of your application
title = AI Vision Assistant

# (str) Package name
package.name = aivisionassistant

# (str) Package domain (Updated to your requested domain)
package.domain = com.ganesh

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (Added tflite extension explicitly)
source.include_exts = py,png,jpg,kv,atlas,tflite

# (str) Application versioning
version = 0.1

# (list) Application requirements
# PINNED: kivy==2.3.0 is required for the latest Android fixes
requirements = python3, kivy==2.3.0, numpy, tflite-runtime, jnius, pillow

# (str) Custom pip dependencies 
# PINNED: Cython 0.29.33 prevents the 'boxshadow.c' missing file error
android.pip_dependencies = cython==0.29.33

# (list) Supported orientations
orientation = portrait

#
# Android specific
#

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (list) Permissions (Updated for your new main.py popup logic)
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

# (int) Target Android API (API 33 is standard for current Play Store)
android.api = 33

# (int) Minimum API your APK will support
android.minapi = 21

# (str) Android NDK version to use
android.ndk = 25b

# (bool) Indicate whether the screen should stay on
android.wakelock = True

# (list) The Android archs to build for (Single arch for faster building)
android.archs = arm64-v8a

# (bool) enables Android auto backup feature
android.allow_backup = True

# (str) The format used to package the app for debug mode
android.debug_artifact = apk

[buildozer]

# (int) Log level (2 = debug info to see error details)
log_level = 2

# (int) Display warning if buildozer is run as root
warn_on_root = 1
