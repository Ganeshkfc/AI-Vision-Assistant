[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = com.ganesh
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

version = 0.1

# PINNED: Using hostpython3 and specific Kivy version for stability
requirements = python3, kivy==2.3.0, hostpython3, numpy, tflite-runtime, camera4kivy, pillow, jnius

# PINNED: This version of Cython is the "sweet spot" for Kivy 2.3.0
android.pip_dependencies = cython==0.29.33

orientation = portrait
fullscreen = 0

# Android specific
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33
android.permissions = CAMERA, INTERNET, MANAGE_EXTERNAL_STORAGE
android.wakelock = True
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
