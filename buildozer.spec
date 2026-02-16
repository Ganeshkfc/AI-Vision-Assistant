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

# REQUIREMENTS: Pinned Cython and organized for AI/Camera stability
requirements = python3, kivy==2.3.0, cython==0.29.33, camera4kivy, gestures4kivy, pillow, pyjnius, numpy, tflite-runtime, android, sqlite3

# CameraX and Provider Setup
p4a.hook = camerax_provider/gradle_options.py

# Android API & NDK Configuration (Updated for API 34 stability)
android.api = 34
android.minapi = 21
android.ndk = 25b
android.ndk_api = 24
android.accept_sdk_license = True
android.enable_androidx = True

# PERMISSIONS: Corrected and expanded for CameraX and TFLite
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, WAKE_LOCK, RECORD_AUDIO
android.wakelock = True

# OPTIMIZATION: Build ONLY for 64-bit to prevent compilation timeouts
android.archs = arm64-v8a

# Compilation Flags to prevent "implicit function declaration" errors in NDK 25b
android.extra_cflags = -Wno-error=implicit-function-declaration -fno-lto

[buildozer]
# Level 1 prevents the GitHub log buffer from cutting off prematurely
log_level = 1
warn_on_root = 1
