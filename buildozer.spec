[app]
# (str) Title of your application
title = AI Vision Assistant

# (str) Package name
package.name = aivisionassistant

# (str) Package domain (needed for android/ios packaging)
package.domain = org.aivision

# (str) Source code where the main.py live
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas,tflite,txt

# (list) List of directory to exclude
source.exclude_dirs = tests, bin, venv, .venv, .git, .github, docs, examples

# (list) List of exclusions using pattern matching
android.exclude_src = bin/*, .google*, tests/*, **/test/*, **/tests/*, **/idle_test/*, **/lib-tk/*, **/lib2to3/*, **/doc/*, **/docs/*, **/examples/*

# (str) Application versioning
version = 1.0

# (str) Supported orientation (one of landscape, sensorLandscape, portrait or all)
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (list) Application requirements
# Added cython==3.0.11 to fix Kivy 2.3.0 build errors
requirements = python3, kivy==2.3.0, cython==3.0.11, camera4kivy, gestures4kivy, pillow, jnius, numpy, tflite-runtime, android

# (str) Custom source folders for requirements
# This hook is essential for Camera4Kivy to enable CameraX
p4a.hook = camerax_provider/gradle_options.py

# (int) Target Android API, should be as high as possible.
android.api = 33

# (int) Minimum API your APK will support.
android.minapi = 24

# (str) Android NDK version to use
android.ndk = 25b

# (int) Android NDK API to use.
android.ndk_api = 24

# (bool) Use --private data storage (True) or --dir public storage (False)
android.private_storage = True

# (bool) If True, then automatically accept SDK license
android.accept_sdk_license = True

# (list) Permissions
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, WAKE_LOCK, RECORD_AUDIO

# (bool) Enable AndroidX support. Required for CameraX (camera4kivy)
android.enable_androidx = True

# (list) The Android architectures to build for
android.archs = arm64-v8a

# (list) Extra CFLAGS for the build
android.extra_cflags = -Wno-error=implicit-function-declaration -fno-lto

# (str) Application icon
# icon.filename = %(source.dir)s/data/icon.png

# (str) Presplash of the application
# presplash.filename = %(source.dir)s/data/presplash.png

[buildozer]
# (int) Log level (0 = error only, 1 = info, 2 = debug (with command output))
log_level = 2

# (int) Display warning if buildozer is run as root (0 = no, 1 = yes)
warn_on_root = 1
