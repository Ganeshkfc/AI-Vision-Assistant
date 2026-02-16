[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples

version = 1.0
orientation = portrait
fullscreen = 0

# REDUCED REQUIREMENTS: Minimum set for successful first build
requirements = python3, kivy==2.3.0, cython==0.29.33, camera4kivy, gestures4kivy, pyjnius, android, sqlite3

# CameraX and Provider Setup
p4a.hook = camerax_provider/gradle_options.py

# Android API & NDK Configuration
android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True

# PERMISSIONS
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, WAKE_LOCK, RECORD_AUDIO
android.wakelock = True

# OPTIMIZATION: Single Architecture Only
android.archs = arm64-v8a

# Compilation Flags
android.extra_cflags = -Wno-error=implicit-function-declaration -fno-lto

[buildozer]
log_level = 1
warn_on_root = 1
