import os
import time
import numpy as np
from PIL import Image
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
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
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 5 # Reduced slightly
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        
        self.preview = Preview(aspect_ratio='16:9')
        # CRITICAL: enable pixel analysis
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame

        self.top_btn = Button(
            text="TAP HERE TO CHANGE MODE\n(Mode 1 Active: Multi-Object Detection)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='20sp', size_hint_y=0.2, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.bottom_btn = Button(
            text="TAP HERE TO CLOSE APP",
            background_color=(0.8, 0.2, 0.2, 1), font_size='20sp', size_hint_y=0.2, halign='center'
        )
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.preview)
        layout.add_widget(self.bottom_btn)
        return layout

    def on_start(self):
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
            except Exception as e:
                Logger.error(f"TTS Initialization Error: {e}")

        Clock.schedule_once(self.load_model, 0.5)

        if platform == 'android':
            perms = [Permission.CAMERA, Permission.RECORD_AUDIO] # Added Record Audio just in case
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
                Logger.info("MODEL: Loaded Successfully!")
            except Exception as e:
                Logger.error(f"MODEL: Load Error: {e}")
        else:
            Logger.error(f"MODEL: File not found at {model_path}")

    def on_permission_result(self, permissions, grants):
        if all(grants):
            Logger.info("PERMS: All granted.")
            self.start_camera()
        else:
            Logger.error("PERMS: Denied.")

    def start_camera(self):
        Clock.schedule_once(self._connect_camera, 1)

    def _connect_camera(self, dt):
        try:
            self.preview.connect_camera(camera_id='back', enable_analyze_pixels = True)
            Logger.info("CAMERA: Connected")
            Clock.schedule_once(lambda x: self.speak("Vision Activated"), 2)
        except Exception as e:
            Logger.error(f"CAMERA: Error {e}")

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Mode 2: Distance" if self.current_mode == 2 else "Mode 1: Detection"
        self.speak(msg)
        self.top_btn.text = msg

    def check_close_app(self, instance):
        self.speak("Closing")
        self.preview.disconnect_camera()
        Clock.schedule_once(lambda dt: self.stop(), 0.5)

    def speak(self, text):
        if self.tts:
            try:
                self.tts.speak(text, 0, None)
            except:
                pass

    def get_distance_cm(self, label, width_px):
        real_w = self.KNOWN_WIDTHS.get(label, 30)
        return (real_w * self.FOCAL_LENGTH) / max(width_px, 1)

    def analyze_frame(self, pixels, width, height, image_pos, image_size, texture):
        if not self.interpreter:
            return

        try:
            # OPTIMIZED: Use 3 or 4 channels dynamically
            channels = len(pixels) // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            rgb = frame[:, :, :3] # Take only RGB
            
            # 1. Resize for YOLOv8
            img = Image.fromarray(rgb).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            # Handle NCHW models
            if self.input_details[0]['shape'][1] == 3:
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            # 2. Run Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # 3. FAST VECTORIZED PROCESSING (No slow 'for' loop)
            output = output.transpose() # Shape: [8400, 84]
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > 0.45 # Threshold
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                # Take the top detection for speech
                top_idx = np.argmax(valid_scores)
                label = self.class_names[valid_class_ids[top_idx]]
                
                # Coordinate scaling
                xc, yc, w, h = valid_boxes[top_idx][:4]
                width_px = w * (width / 640)

                # 4. Handle Speech with Cooldown
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    if self.current_mode == 1:
                        self.speak(f"I see a {label}")
                    else:
                        dist = self.get_distance_cm(label, width_px)
                        self.speak(f"{label} at {int(dist)} centimeters")
                    self.last_speech_time = now
                    Logger.info(f"AI: Detected {label}")

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
