[app]
title = AI Vision Assistant
package.name = aivisionassistant
package.domain = com.ganesh

source.dir = .
source.include_exts = py,png,jpg,kv,atlas

# Requirements: Cleaned up to avoid conflicts. 
# Do NOT include python-for-android or pyjnius here.
requirements = python3, kivy==2.3.0, hostpython3, pillow, requests

orientation = portrait
fullscreen = 0

# Android specific
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33

# Modern Permissions (Replaces old WRITE_EXTERNAL_STORAGE)
android.permissions = INTERNET, MANAGE_EXTERNAL_STORAGE

# Build architecture
android.archs = arm64-v8a

[buildozer]
log_level = 2
warn_on_root = 1
