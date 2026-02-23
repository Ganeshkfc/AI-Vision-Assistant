import os
import time
import numpy as np
from PIL import Image
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.graphics import Color, Line
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

class BBoxOverlay(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels = []

    def draw_boxes(self, valid_boxes, valid_class_ids, class_names, preview_widget):
        self.canvas.clear()
        for lbl in self.labels:
            self.remove_widget(lbl)
        self.labels.clear()

        pw, ph = preview_widget.size
        px, py = preview_widget.pos

        with self.canvas:
            for i in range(len(valid_boxes)):
                box = valid_boxes[i]
                class_id = int(valid_class_ids[i])
                label_name = class_names[class_id]

                xc, yc, w, h = map(float, box[:4])
                scale_x, scale_y = pw / 640.0, ph / 640.0
                
                w_px, h_px = w * scale_x, h * scale_y
                x1_px = px + ((xc - w/2) * scale_x)
                y1_px = py + ph - ((yc + h/2) * scale_y) 

                Color(0, 1, 0, 1) # Bright green
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=4)

                lbl = Label(text=label_name.upper(), pos=(float(x1_px), float(y1_px + h_px)), 
                            size_hint=(None, None), size=(200, 50), color=(0,1,0,1), bold=True)
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 45, 'chair': 50, 'bottle': 7, 'cell phone': 7, 'cup': 9, 'laptop': 32, 'tv': 80}
        self.FOCAL_LENGTH = 720 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 5.0 
        self.frame_count = 0 # To track if the camera is working
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        self.top_btn = Button(
            text="MODE 1: MULTI-OBJECT\n(Tap to switch)",
            background_color=(0.1, 0.4, 0.8, 1), font_size='20sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.camera_container = FloatLayout()
        self.preview = Preview(aspect_ratio='16:9')
        self.overlay = BBoxOverlay()
        
        self.camera_container.add_widget(self.preview)
        self.camera_container.add_widget(self.overlay)

        self.bottom_btn = Button(text="EXIT APP", background_color=(0.8, 0.1, 0.1, 1), size_hint_y=0.10)
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.camera_container)
        layout.add_widget(self.bottom_btn)
        return layout

    def on_start(self):
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                Locale = autoclass('java.util.Locale')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
                # Slow down the rate to 0.5 (half speed)
                Clock.schedule_once(self.setup_tts, 2.0)
            except Exception as e:
                Logger.error(f"TTS_INIT_ERROR: {e}")

        Clock.schedule_once(self.load_model, 0.5)
        if platform == 'android':
            request_permissions([Permission.CAMERA], self.on_permission_result)
        else:
            self.start_camera()

    def setup_tts(self, dt):
        if self.tts:
            try:
                self.tts.setLanguage(autoclass('java.util.Locale').US)
                self.tts.setSpeechRate(0.5) # Very slow and clear
                self.speak("System ready.")
                Logger.info("TTS: Slow rate configured successfully.")
            except:
                pass

    def load_model(self, dt):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(cur_dir, "yolov8n_float32.tflite")
        if os.path.exists(model_path) and tflite:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            Logger.info("MODEL: Loaded Successfully!")

    def on_permission_result(self, permissions, grants):
        if all(grants): 
            Logger.info("PERMISSIONS: Camera granted.")
            self.start_camera()
        else:
            Logger.error("PERMISSIONS: Camera denied!")

    def start_camera(self):
        Logger.info("CAMERA: Starting back camera...")
        Clock.schedule_once(lambda dt: self.preview.connect_camera(camera_id='back', enable_analyze_pixels=True), 1)

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Mode 2. Distance Focus." if self.current_mode == 2 else "Mode 1. Multi Object."
        self.speak(msg)
        self.top_btn.text = f"{msg.upper()}\n(Tap to switch)"

    def check_close_app(self, instance):
        self.speak("Goodbye.")
        self.preview.disconnect_camera()
        Clock.schedule_once(lambda dt: self.stop(), 0.5)

    def speak(self, text):
        if self.tts:
            try: 
                self.tts.setSpeechRate(0.5) # Force slow speed
                self.tts.speak(text, 0, None)
            except: pass

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return
        self.frame_count += 1
        
        # HEARTBEAT: Log every 60 frames (~2 seconds) to confirm the camera is alive
        if self.frame_count % 60 == 0:
            Logger.info(f"AI_HEARTBEAT: Camera is active. Analyzed {self.frame_count} frames so far.")

        try:
            width, height = args[0] if isinstance(args[0], (list, tuple)) else (args[0], args[1])
            n_pixels = len(pixels)
            channels = n_pixels // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            
            # Input Prep
            img = Image.fromarray(frame[:, :, :3]).resize((640, 640))
            
            # Auto-handle Float vs Uint8 models
            if self.input_details[0]['dtype'] == np.uint8:
                input_data = np.expand_dims(np.array(img), axis=0).astype(np.uint8)
            else:
                input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            if self.input_details[0]['shape'][1] == 3: 
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].transpose()
            
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > 0.45 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                # Filter duplicates (NMS)
                selected = []
                indices = np.argsort(valid_scores)[::-1]
                for i in indices:
                    box_i = valid_boxes[i]
                    is_duplicate = False
                    for j in selected:
                        box_j = valid_boxes[j]
                        if np.linalg.norm(box_i[:2] - box_j[:2]) < 65: 
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        selected.append(i)
                        if len(selected) >= 3: break
                
                final_boxes = valid_boxes[selected]
                final_classes = valid_class_ids[selected]
                
                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(final_boxes, final_classes, self.class_names, self.preview), 0)
                
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    if self.current_mode == 1:
                        descriptions = []
                        for i in range(len(final_boxes)):
                            label = self.class_names[int(final_classes[i])]
                            xc = final_boxes[i][0]
                            if xc < 210: pos = "to your left"
                            elif xc > 430: pos = "to your right"
                            else: pos = "straight ahead"
                            descriptions.append(f"{label} {pos}")
                        self.speak("Detected: " + ". And ".join(descriptions))
                    else:
                        label = self.class_names[int(final_classes[0])]
                        w_px = final_boxes[0][2]
                        dist_cm = (self.KNOWN_WIDTHS.get(label, 30) * self.FOCAL_LENGTH) / max(w_px, 1)
                        if dist_cm < 31:
                            self.speak(f"{label}. {int(dist_cm)} centimeters.")
                        else:
                            feet = round(dist_cm / 30.48, 1)
                            self.speak(f"{label}. {feet} feet.")
                    self.last_speech_time = now
            else:
                # If nothing is detected, clear the boxes
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)

        except Exception as e:
            Logger.error(f"AI_ANALYSIS_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
