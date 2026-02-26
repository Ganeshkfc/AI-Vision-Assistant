import os
import time
import numpy as np
from PIL import Image
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.clock import Clock
from kivy.utils import platform
from kivy.logger import Logger
from camera4kivy import Preview

if platform == 'android':
    from jnius import autoclass
    from android.permissions import request_permissions, Permission

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None
        Logger.error("TFLite module not found!")

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7, 'tv': 60}
        self.FOCAL_LENGTH = 2000 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 4 
        
        # Flashlight State
        self.flashlight_is_on = False
        # Increased to 100 so it turns on in "little bit dark" rooms
        self.brightness_threshold = 100  
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.vibrator = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        
        self.top_btn = Button(
            text="TAP TO CHANGE MODE\n(Mode 1: Direction Mode)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='18sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.middle_layout = BoxLayout(orientation='horizontal')
        
        self.slider_left = BoxLayout(orientation='vertical', size_hint_x=0.15)
        self.slider_label = Label(text='45%', size_hint_y=0.1, font_size='16sp')
        self.threshold_slider = Slider(orientation='vertical', min=1, max=100, value=45, size_hint_y=0.9)
        self.threshold_slider.bind(value=self.on_slider_value_change)
        self.slider_left.add_widget(self.slider_label)
        self.slider_left.add_widget(self.threshold_slider)

        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        
        self.slider_right = BoxLayout(orientation='vertical', size_hint_x=0.15)
        self.speed_label = Label(text='0.8x', size_hint_y=0.1, font_size='16sp')
        self.speed_slider = Slider(orientation='vertical', min=0.5, max=2.0, value=0.8, size_hint_y=0.9)
        self.speed_slider.bind(value=self.on_speed_change)
        self.slider_right.add_widget(self.speed_label)
        self.slider_right.add_widget(self.speed_slider)

        self.middle_layout.add_widget(self.slider_left)
        self.middle_layout.add_widget(self.preview)
        self.middle_layout.add_widget(self.slider_right)

        self.bottom_btn = Button(
            text="TAP HERE TO CLOSE APP",
            background_color=(0.8, 0.2, 0.2, 1), font_size='20sp', size_hint_y=0.15, halign='center'
        )
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.middle_layout)
        layout.add_widget(self.bottom_btn)
        return layout

    def set_flashlight(self, state):
        if platform == 'android' and self.flashlight_is_on != state:
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                Context = autoclass('android.content.Context')
                cameraManager = PythonActivity.mActivity.getSystemService(Context.CAMERA_SERVICE)
                cameraId = cameraManager.getCameraIdList()[0]
                cameraManager.setTorchMode(cameraId, state)
                self.flashlight_is_on = state
            except Exception as e:
                Logger.error(f"FLASHLIGHT: Error: {e}")

    def get_color_name(self, r, g, b):
        """Simple logic to identify the dominant color"""
        if r > 200 and g > 200 and b > 200: return "White"
        if r < 50 and g < 50 and b < 50: return "Black"
        
        colors = {"Red": (r), "Green": (g), "Blue": (b)}
        dominant = max(colors, key=colors.get)
        
        if dominant == "Red" and g > 100: return "Yellow or Orange"
        if dominant == "Blue" and g > 150: return "Cyan or Light Blue"
        if dominant == "Red" and b > 100: return "Purple or Pink"
        
        return dominant

    def on_slider_value_change(self, instance, value):
        self.slider_label.text = f"{int(value)}%"

    def on_speed_change(self, instance, value):
        self.speed_label.text = f"{value:.1f}x"
        if self.tts:
            try: self.tts.setSpeechRate(float(value))
            except: pass

    def on_start(self):
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                Context = autoclass('android.content.Context')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
                self.vibrator = PythonActivity.mActivity.getSystemService(Context.VIBRATOR_SERVICE)
                Clock.schedule_once(lambda dt: self.tts.setSpeechRate(self.speed_slider.value) if self.tts else None, 1.5)
            except Exception as e:
                Logger.error(f"Android Services Error: {e}")

        Clock.schedule_once(self.load_model, 0.5)
        if platform == 'android':
            perms = [Permission.CAMERA, Permission.VIBRATE]
            request_permissions(perms, self.on_permission_result)
        else:
            self.start_camera()

    def load_model(self, dt):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(cur_dir, "yolov8n_float32.tflite")
        if os.path.exists(model_path) and tflite:
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as e:
                Logger.error(f"MODEL: Load Error: {e}")

    def on_permission_result(self, permissions, grants):
        if all(grants):
            self.start_camera()

    def start_camera(self):
        Clock.schedule_once(self._connect_camera, 1)

    def _connect_camera(self, dt):
        try:
            self.preview.connect_camera(camera_id='back', enable_analyze_pixels=True)
            self.speak("AI Vision started. Mode 1 active.")
        except Exception as e:
            Logger.error(f"CAMERA: Error {e}")

    def toggle_mode(self, instance):
        self.current_mode += 1
        if self.current_mode > 3: self.current_mode = 1
        
        modes = {1: "Direction Mode", 2: "Distance Mode", 3: "Color Detection Mode"}
        msg = modes[self.current_mode]
        self.speak(msg + " active")
        self.top_btn.text = f"TAP TO CHANGE MODE\n({msg})"

    def check_close_app(self, instance):
        self.set_flashlight(False)
        self.speak("Closing application.")
        self.preview.disconnect_camera()
        Clock.schedule_once(lambda dt: self.stop(), 0.5)

    def speak(self, text):
        if self.tts:
            try: self.tts.speak(text, 1, None)
            except: pass

    def vibrate(self, duration_ms):
        if self.vibrator:
            try: self.vibrator.vibrate(duration_ms)
            except: pass

    def get_distance_cm(self, label, width_px, frame_w):
        real_w = self.KNOWN_WIDTHS.get(label, 30)
        return (real_w * self.FOCAL_LENGTH) / max(width_px, 1)

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return
        try:
            if isinstance(args[0], (list, tuple)):
                width, height = args[0]
            else:
                width, height = args[0], args[1]

            channels = len(pixels) // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            rgb = frame[:, :, :3] 

            # --- AUTO FLASHLIGHT ---
            avg_brightness = np.mean(rgb)
            if avg_brightness < self.brightness_threshold:
                self.set_flashlight(True)
            elif avg_brightness > (self.brightness_threshold + 30):
                self.set_flashlight(False)

            now = time.time()

            # --- MODE 3: COLOR DETECTION ---
            if self.current_mode == 3:
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    # Look at 50x50 center patch
                    center_x, center_y = width // 2, height // 2
                    patch = rgb[center_y-25:center_y+25, center_x-25:center_x+25]
                    avg_color = np.mean(patch, axis=(0, 1))
                    color_name = self.get_color_name(avg_color[0], avg_color[1], avg_color[2])
                    self.speak(f"Color is {color_name}")
                    self.last_speech_time = now
                return

            # --- MODES 1 & 2: OBJECT DETECTION ---
            img = Image.fromarray(rgb).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            if self.input_details[0]['shape'][1] == 3: 
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].transpose()
            
            scores = np.max(output[:, 4:], axis=1)
            current_threshold = self.threshold_slider.value / 100.0
            mask = scores > current_threshold 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                sort_idx = np.argsort(valid_scores)[::-1]
                top_k = min(3, len(sort_idx))
                
                speech_segments = []
                for i in range(top_k):
                    class_id = int(valid_class_ids[sort_idx[i]])
                    name = self.class_names[class_id]
                    xc, yc, w, h = valid_boxes[sort_idx[i]][:4]
                    
                    val = xc if xc <= 1.0 else xc / 640.0
                    if val < 0.35: direction = "on your left"
                    elif val > 0.65: direction = "on your right"
                    else: direction = "In front of you"

                    if self.current_mode == 1:
                        if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                            speech_segments.append(f"{name} {direction}")
                    elif self.current_mode == 2 and i == 0:
                        width_px = w if w > 1.0 else w * 640
                        dist_cm = self.get_distance_cm(name, width_px, width)
                        if dist_cm < 100: self.vibrate(200)
                        if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                            dist_str = f"{int(dist_cm)} cm" if dist_cm < 91 else f"{dist_cm/30.48:.1f} feet"
                            speech_segments.append(f"{name} {direction}, {dist_str}")

                if speech_segments and (now - self.last_speech_time > self.SPEECH_COOLDOWN):
                    self.speak(", ".join(speech_segments))
                    self.last_speech_time = now

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
