[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
source.exclude_dirs = tests, bin, venv, .venv, .git, .github
version = 1.0

# Added 'hostpython3' to requirements for better build stability
requirements = python3, kivy==2.3.0, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, tflite-runtime, hostpython3, clippy

orientation = portrait
android.api = 33
android.minapi = 21
android.ndk = 25b
android.ndk_api = 21
android.permissions = CAMERA, INTERNET, READ_MEDIA_IMAGES, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE
android.private_storage = True
android.accept_sdk_license = True

# --- CRITICAL FIXES FOR CAMERA & TFLITE CONFLICTS ---
android.enable_androidx = True
# This hook connects your Python code to the Android Camera hardware
p4a.hook = camerax_provider/gradle_options.py

# FIX for duplicate library errors during APK packaging
android.gradle_options = "packagingOptions { pickFirst 'lib/*/librestrited_xyz.so'; exclude 'META-INF/INDEX.LIST' }"

# Required repository for TFLite dependencies
android.add_gradle_repositories = "https://maven.google.com"
# ----------------------------------------------------

android.archs = arm64-v8a
android.no_byte_compile_python = True
p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1
