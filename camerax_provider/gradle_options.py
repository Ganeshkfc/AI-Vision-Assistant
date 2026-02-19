from os.path import join, dirname

def before_build(toolchain):
    # This script adds the Google CameraX dependencies to the Android build
    gradle_properties = join(toolchain.sdk_dir, '..', '..', 'build', 'python-installs', toolchain.app_name, 'camerax_provider', 'gradle.properties')
    
    with open(join(dirname(__file__), 'gradle.properties'), 'w') as f:
        f.write('android.useAndroidX=true\n')
        f.write('android.enableJetifier=true\n')

def update_gradle_dependencies(toolchain):
    # This adds the actual camera libraries
    return [
        "androidx.camera:camera-core:1.3.0",
        "androidx.camera:camera-camera2:1.3.0",
        "androidx.camera:camera-lifecycle:1.3.0",
        "androidx.camera:camera-video:1.3.0",
        "androidx.camera:camera-view:1.3.0",
        "androidx.camera:camera-extensions:1.3.0"
    ]
