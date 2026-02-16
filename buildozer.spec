[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples
android.exclude_src = bin/*, .google*, tests/*, **/test/*, **/tests/*, **/idle_test/*, **/lib-tk/*, **/lib2to3/*, **/doc/*, **/docs/*, **/examples/*

version = 1.0
orientation = portrait
fullscreen = 0

# Requirements: Added cython pin and ensured camera requirements are present
requirements = python3, kivy==2.3.0, cython==3.0.11, camera4kivy, gestures4kivy, pillow, jnius, numpy, tflite-runtime, android

# Essential hook for CameraX provider
p4a.hook = camerax_provider/gradle_options.py

android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True

# Added RECORD_AUDIO as it is often a silent dependency for camera recipes
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, WAKE_LOCK, RECORD_AUDIO

android.wakelock = True
android.archs = arm64-v8a
android.extra_cflags = -Wno-error=implicit-function-declaration -fno-lto

[buildozer]
log_level = 2
warn_on_root = 1
