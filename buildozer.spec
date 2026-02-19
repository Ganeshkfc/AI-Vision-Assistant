[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.aivision
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt,java
source.exclude_dirs = tests, bin, venv, .venv, .git, .github
version = 1.0

# UPDATED: Ensuring camera4kivy and gestures4kivy are called cleanly
requirements = python3, kivy==2.3.0, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, tflite-runtime, hostpython3

orientation = portrait
android.api = 33
android.minapi = 21
android.ndk = 25b
android.ndk_api = 21
# FIXED: Added permission for Camera and standard storage
android.permissions = CAMERA, INTERNET, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, READ_MEDIA_IMAGES
android.private_storage = True
android.accept_sdk_license = True

# --- CAMERA & TFLITE INTEGRATION ---
android.enable_androidx = True
p4a.hook = camerax_provider/gradle_options.py
android.add_src = camerax_provider/camerax_src

android.gradle_options = "packagingOptions { pickFirst 'lib/arm64-v8a/libc++_shared.so'; pickFirst 'lib/armeabi-v7a/libc++_shared.so'; pickFirst 'lib/x86/libc++_shared.so'; pickFirst 'lib/x86_64/libc++_shared.so'; exclude 'META-INF/INDEX.LIST' }"
android.add_gradle_repositories = "https://maven.google.com"

android.archs = arm64-v8a
android.no_byte_compile_python = True
p4a.branch = master

[buildozer]
log_level = 2
warn_on_root = 1
