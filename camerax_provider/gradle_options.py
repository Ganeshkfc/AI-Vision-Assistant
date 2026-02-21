#
# Add gradle options for CameraX
#
from pythonforandroid.logger import info # Modern logger
from os.path import dirname, join, exists

def before_apk_build(toolchain):
    unprocessed_args = toolchain.args.unknown_args

    # 1. Ensure AndroidX is enabled
    if '--enable-androidx' not in unprocessed_args:
        unprocessed_args.append('--enable-androidx')
        info('Camerax Hook: Add android.enable_androidx = True')

    # 2. Check for required permissions
    for perm in ['CAMERA', 'RECORD_AUDIO']:
        if perm not in unprocessed_args:
            unprocessed_args.append('--permission')
            unprocessed_args.append(perm)
            info(f'Camerax Hook: Auto-adding permission: {perm}')

    # 3. Define modern CameraX dependencies for API 33 compatibility
    # Optimized versions for stability
    required_depends = [
        'androidx.camera:camera-core:1.2.3',
        'androidx.camera:camera-camera2:1.2.3',
        'androidx.camera:camera-lifecycle:1.2.3',
        'androidx.camera:camera-view:1.2.3',
        'androidx.lifecycle:lifecycle-process:2.5.1',  
        'androidx.core:core:1.9.0'
    ]    

    existing_depends = []
    read_next = False
    for ua in unprocessed_args:
        if read_next:
            existing_depends.append(ua)
            read_next = False
        if ua == '--depend':
            read_next = True
            
    # Add missing dependencies
    for rd in required_depends:
        name, version = rd.rsplit(':', 1)
        found = False
        for ed in existing_depends:
            if name in ed:
                found = True
                break
        if not found:
            unprocessed_args.append('--depend')
            unprocessed_args.append(f'{name}:{version}')
            info(f'Camerax Hook: Adding dependency {name}:{version}')
        
    # 4. Add the Java source (Only if not already in buildozer.spec)
    # Note: If you have 'android.add_src' in buildozer.spec, we skip this to avoid duplicates
    camerax_java = join(dirname(__file__), 'camerax_src')
    
    source_already_added = False
    for i, arg in enumerate(unprocessed_args):
        if arg == '--add-source' and i + 1 < len(unprocessed_args):
            if 'camerax_src' in unprocessed_args[i+1]:
                source_already_added = True
                break

    if exists(camerax_java) and not source_already_added:
        unprocessed_args.append('--add-source')
        unprocessed_args.append(camerax_java)
        info('Camerax Hook: Added Java source from camerax_src folder')
    elif not exists(camerax_java):
        info('Camerax Hook WARNING: camerax_src folder not found!')
