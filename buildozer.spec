[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = org.renpy
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
version = 1.0
requirements = python3,kivy,plyer

orientation = portrait
fullscreen = 0
android.archs = arm64-v8a

# CRITICAL FIX: Stops the build from stalling on Python test files
android.exclude_src = **/test/**, **/tests/**

# Permissions
android.permissions = CAMERA, INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE

[buildozer]
log_level = 2
warn_on_root = 1
