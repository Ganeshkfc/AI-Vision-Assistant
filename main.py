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
    import tensorflow.lite as tflite

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.METRIC_THRESHOLD_CM = 92 # 3 Feet threshold
        self.KNOWN_WIDTHS = {'person': 55, 'chair': 50, 'bottle': 9, 'cell phone': 8, 'tv': 75}
        
        # INCREASE THIS NUMBER if the AI says objects are CLOSER than they really are
        # DECREASE THIS NUMBER if the AI says objects are FURTHER than they really are
        self.FOCAL_LENGTH = 1600 
        
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 3.5 
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.vibrator = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        self.top_btn = Button(text="MODE 1: DETECTION", background_color=(0.1, 0.5, 0.8, 1), size_hint_y=0.15)
        self.top_btn.bind(on_release=self.toggle_mode)

        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame

        self.bottom_btn = Button(text="EXIT APP", background_color=(0.8, 0.2, 0.2, 1), size_hint_y=0.15)
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.preview)
        layout.add_widget(self.bottom_btn)
        return layout

    def on_start(self):
        if platform == 'android':
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
            self.vibrator = PythonActivity.mActivity.getSystemService(autoclass('android.content.Context').VIBRATOR_SERVICE)
            perms = [Permission.CAMERA, Permission.VIBRATE]
            request_permissions(perms, lambda p, g: self.preview.connect_camera(enable_analyze_pixels=True))
        
        Clock.schedule_once(self.load_model, 1)

    def load_model(self, dt):
        path = os.path.join(os.path.dirname(__file__), "yolov8n_float32.tflite")
        self.interpreter = tflite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Distance Mode" if self.current_mode == 2 else "Direction Mode"
        self.speak(msg)
        self.top_btn.text = f"ACTIVE: {msg}"

    def speak(self, text):
        if self.tts: self.tts.speak(text, 1, None)

    def get_distance_cm(self, label, width_px):
        real_w = self.KNOWN_WIDTHS.get(label, 35)
        return (real_w * self.FOCAL_LENGTH) / max(width_px, 1)

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return
        try:
            width, height = args[0] if isinstance(args[0], (list, tuple)) else (args[0], args[1])
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
            img = Image.fromarray(frame).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            if self.input_details[0]['shape'][1] == 3: input_data = np.transpose(input_data, (0, 3, 1, 2))
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].transpose()
            
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > 0.45
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                now = time.time()

                if self.current_mode == 2:
                    # --- YOUR LOGIC: FIND OBJECT CLOSEST TO CENTER ---
                    best_idx = -1
                    min_dist_to_center = float('inf')
                    
                    for i in range(len(valid_boxes)):
                        xc = valid_boxes[i][0] # Center X from model (0 to 640)
                        dist_to_center = abs(xc - 320) # 320 is middle of 640
                        if dist_to_center < min_dist_to_center:
                            min_dist_to_center = dist_to_center
                            best_idx = i

                    if best_idx != -1 and (now - self.last_speech_time > self.SPEECH_COOLDOWN):
                        box = valid_boxes[best_idx]
                        name = self.class_names[valid_ids[best_idx]]
                        
                        # Calculate Pixel Width (x2 - x1)
                        # In YOLO output, index 2 is width
                        pixel_w = box[2] 
                        d_cm = self.get_distance_cm(name, pixel_w)

                        # Determine Direction
                        xc = box[0]
                        if xc < 213: dir_s = "on your left"
                        elif xc < 426: dir_s = "in front of you"
                        else: dir_s = "on your right"

                        # --- YOUR UNIT CONVERSION LOGIC ---
                        if d_cm < self.METRIC_THRESHOLD_CM:
                            d_str = f"{int(d_cm)} centimeters"
                        else:
                            d_feet = round(d_cm / 30.48, 1)
                            d_str = f"{d_feet} feet"

                        self.speak(f"{name} {dir_s}, {d_str}")
                        self.last_speech_time = now
                
                elif self.current_mode == 1:
                    # Normal detection for top 2 items
                    if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                        top_names = [self.class_names[valid_ids[j]] for j in range(min(2, len(valid_ids)))]
                        self.speak(f"Objects: {', '.join(top_names)}")
                        self.last_speech_time = now

        except Exception as e:
            Logger.error(f"AI Error: {e}")

    def check_close_app(self, instance):
        self.stop()

if __name__ == "__main__":
    VisionApp().run()
