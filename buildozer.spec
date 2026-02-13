[app]
# (Section 1: Basic App Info)
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = com.ganesh
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite,txt
version = 0.1
orientation = portrait
fullscreen = 0

# (Section 2: Requirements)
# Pinned versions for stability with NDK 25b
requirements = python3==3.10.12, kivy==2.3.0, hostpython3, numpy, tflite-runtime, camera4kivy, pillow, jnius, sh, gestures4kivy

# (Section 3: Android / P4A Specifics)
p4a.branch = master
p4a.hook = camera4kivy
android.pip_dependencies = cython==0.29.33

# (Section 4: SDK & NDK Setup)
android.api = 33
android.minapi = 24
android.ndk = 25b
android.sdk = 33
# Forces the NDK to use headers compatible with Python 3.10
android.ndk_api = 24
android.accept_sdk_license = True

# (Section 5: Permissions & Hardware)
android.permissions = CAMERA, INTERNET, READ_MEDIA_IMAGES, READ_MEDIA_VIDEO, WAKE_LOCK
android.wakelock = True
android.archs = arm64-v8a

# (Section 6: CRITICAL COMPILER FIX)
# This prevents the 'endgrent' error from stopping the build
android.extra_cflags = "-Wno-error=implicit-function-declaration -Wno-implicit-function-declaration"

[buildozer]
log_level = 2
warn_on_root = 1
