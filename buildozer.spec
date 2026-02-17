[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.test
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,tflite

# (list) Application requirements
# Added cython which is often needed for building numpy/pillow from source
requirements = python3, kivy==2.3.0, pyjnius, camera4kivy, gestures4kivy, android, numpy, pillow, tflite-runtime, cython

orientation = portrait
fullscreen = 0
android.archs = arm64-v8a
android.allow_backup = True

# (int) Target Android API
android.api = 33
# (int) Minimum Android API
android.minapi = 21

# (list) Permissions
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

# (str) The directory in which python-for-android should look for your own recipes
# p4a.local_recipes = ./recipes

# (str) python-for-android branch to use (develop is better for API 33+)
p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 1
