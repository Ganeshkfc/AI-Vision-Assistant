[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# EXCLUSION FIX: 
# Removed specific numpy exclusions (**/numpy/core/tests/*) because they caused the 
# FileNotFoundError. We will rely on the 6-hour timeout to handle the extra size.
android.exclude_src = bin/*, .google*, tests/*, **/test/*, **/tests/*, **/idle_test/*, **/lib-tk/*, **/lib2to3/*, **/doc/*, **/docs/*, **/examples/*

version = 1.0
orientation = portrait
fullscreen = 0

requirements = python3, kivy==2.3.0, camera4kivy, gestures4kivy, pillow, jnius, numpy, tflite-runtime, android

p4a.hook = camera4kivy

android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True

android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, WAKE_LOCK
android.wakelock = True

android.archs = arm64-v8a
android.extra_cflags = -Wno-error=implicit-function-declaration -fno-lto

[buildozer]
log_level = 2
warn_on_root = 1
