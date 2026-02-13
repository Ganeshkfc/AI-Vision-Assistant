[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = com.ganesh
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

version = 0.1

# PINNED: Requirements for Camera4Kivy and TFLite
requirements = python3, kivy==2.3.0, hostpython3, numpy, tflite-runtime, camera4kivy, pillow, jnius, sh, gestures4kivy

# MANDATORY for Camera4Kivy to work on Android
p4a.branch = master
p4a.hook = camera4kivy

# PINNED: Stable Cython for Kivy 2.3.0
android.pip_dependencies = cython==0.29.33

orientation = portrait
fullscreen = 0

# Android specific (Targeting API 33/Android 13)
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33

# CRITICAL FIX: Automatically accept SDK licenses to prevent "Aidl not found" error
android.accept_sdk_license = True

# Updated permissions for Android 13+ (API 33)
# Note: WRITE_EXTERNAL_STORAGE is ignored on API 33; we use READ_MEDIA instead.
android.permissions = CAMERA, INTERNET, READ_MEDIA_IMAGES, READ_MEDIA_VIDEO, WAKE_LOCK

android.wakelock = True
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
