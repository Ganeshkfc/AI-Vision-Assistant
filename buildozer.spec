[app]
# (Section 1: Basic App Info)
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = com.ganesh
source.dir = .
# Optimized inclusions for vision apps
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
# EXCLUSION FIX: Prevents the build from stalling during bytecode compilation
android.exclude_src = bin/*, .google*, tests/*, **/test/*, **/tests/*, **/idle_test/*
version = 0.1
orientation = portrait
fullscreen = 0

# (Section 2: Requirements)
requirements = python3,kivy,camera4kivy,gestures4kivy,pillow,jnius,numpy,tflite-runtime,sh,android,requests

# (Section 3: Android / P4A Specifics)
p4a.branch = master
p4a.hook = camera4kivy
android.pip_dependencies = cython==0.29.33

# (Section 4: SDK & NDK Setup)
android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True

# (Section 5: Permissions & Hardware)
android.permissions = CAMERA, INTERNET, READ_MEDIA_IMAGES, READ_MEDIA_VIDEO, WAKE_LOCK, WRITE_EXTERNAL_STORAGE
android.wakelock = True
android.archs = arm64-v8a

# (Section 6: CRITICAL COMPILER FIX)
android.extra_cflags = "-Wno-error=implicit-function-declaration -Wno-implicit-function-declaration"

[buildozer]
log_level = 2
warn_on_root = 1
