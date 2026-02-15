[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.renpy
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# EXCLUSION FIX: Expanded to remove all test and doc folders from heavy libraries.
# Removed explicit numpy test exclusions to prevent FileNotFoundError during packaging
android.exclude_src = bin/*, .google*, tests/*, **/test/*, **/tests/*, **/idle_test/*, **/lib-tk/*, **/lib2to3/*, **/doc/*, **/docs/*, **/examples/*

version = 1.0
orientation = portrait
fullscreen = 0

# Requirements optimized for your VisionApp logic
requirements = python3, kivy==2.3.0, camera4kivy, gestures4kivy, pillow, jnius, numpy, tflite-runtime, android

# Camera4Kivy specific hook (DO NOT CHANGE)
p4a.hook = camera4kivy

android.api = 33
android.minapi = 24
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True

# Permissions required for your AI vision and speech logic
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, WAKE_LOCK
android.wakelock = True

# Targeting modern 64-bit devices
android.archs = arm64-v8a

# LTO fix to prevent linker errors and speed up compilation
android.extra_cflags = -Wno-error=implicit-function-declaration -fno-lto

[buildozer]
log_level = 2
warn_on_root = 1
