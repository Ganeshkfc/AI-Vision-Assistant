[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = com.ganesh
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
# EXCLUSION FIX: Vital to avoid GitHub timeout during bytecode compilation
android.exclude_src = bin/*, .google*, tests/*, **/test/*, **/tests/*, **/idle_test/*, **/lib-tk/*, **/lib2to3/*
version = 0.1
orientation = portrait
fullscreen = 0

requirements = python3,kivy==2.3.0,camera4kivy,gestures4kivy,pillow,jnius,numpy,tflite-runtime,sh,android,requests

p4a.branch = master
p4a.hook = camera4kivy
android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.permissions = CAMERA, INTERNET, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, WAKE_LOCK
android.wakelock = True
android.archs = arm64-v8a
android.extra_cflags = "-Wno-error=implicit-function-declaration -Wno-implicit-function-declaration"

[buildozer]
log_level = 2
warn_on_root = 1
