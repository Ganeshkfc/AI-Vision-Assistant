import os
import time
import numpy as np
from PIL import Image
from kivy.app import App
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
                
                # Normalize coordinates
                xc, yc, w, h = map(float, box[:4])
                scale_x, scale_y = pw / 640.0, ph / 640.0
                w_px, h_px = w * scale_x, h * scale_y
                x1_px = px + ((xc - w/2) * scale_x)
                y1_px = py + ph - ((yc + h/2) * scale_y) 

                Color(1, 0, 0, 1) # Red boxes for high visibility
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=4)

class VisionApp(App):
    def build(self):
        self.sensitivity = 0.45
        self.frame_count = 0
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 4.0
        self.interpreter = None
        self.tts = None
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.root = FloatLayout()
        self.preview = Preview() 
        self.overlay = BBoxOverlay()
        
        # Status label for sighted assistants/debugging
        self.status_label = Label(text="Starting Vision...", pos_hint={'center_y': 0.9}, size_hint=(1, 0.1))
        
        self.root.add_widget(self.preview)
        self.root.add_widget(self.overlay)
        self.root.add_widget(self.status_label)
        
        return self.root

    def on_start(self):
        # 1. Load the AI Model first
        Clock.schedule_once(self.load_model, 0.5)
        
        # 2. Setup TTS
        if platform == 'android':
            self.init_tts()
            # 3. Request permissions, then start camera automatically
            request_permissions([Permission.CAMERA], self.on_permission_result)
        else:
            Clock.schedule_once(lambda dt: self.start_camera(), 2.0)

    def init_tts(self):
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
        except: pass

    def speak(self, text):
        if self.tts:
            try: self.tts.speak(text, 0, None)
            except: pass
        Logger.info(f"TTS: {text}")

    def load_model(self, dt):
        try:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(cur_dir, "yolov8n_float32.tflite")
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            Logger.info("AI: Brain Ready")
        except Exception as e:
            Logger.error(f"MODEL_ERROR: {e}")

    def on_permission_result(self, permissions, grants):
        if all(grants):
            # Wait 2 seconds so the system is stable before firing the camera
            Clock.schedule_once(lambda dt: self.start_camera(), 2.0)
        else:
            self.speak("Camera permission is required for this app to work.")

    def start_camera(self):
        self.speak("Camera starting.")
        self.preview.analyze_callback = self.analyze_frame
        try:
            # We use no extra parameters here to let the Android driver pick the most stable defaults
            self.preview.connect_camera(enable_analyze_pixels=True)
            self.status_label.text = "Vision Active"
        except Exception as e:
            Logger.error(f"CAMERA_ERROR: {e}")
            self.speak("Camera failed to start.")

    def analyze_frame(self, pixels, width, height, rotation):
        if self.frame_count == 0:
            Logger.info(f"AI_FRAME_FLOW: First frame received {width}x{height}")
            self.speak("System ready.")
        
        self.frame_count += 1
        if not self.interpreter or self.frame_count % 2 != 0: return # Skip every 2nd frame for speed

        try:
            n_pixels = len(pixels)
            channels = n_pixels // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            
            # Prepare image for YOLO
            img = Image.fromarray(frame[:, :, :3]).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            if self.input_details[0]['shape'][1] == 3: 
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].transpose()
            
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > self.sensitivity 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                # Draw boxes
                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(valid_boxes[:3], valid_class_ids[:3], self.class_names, self.preview), 0)
                
                # Audio Feedback
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    label = self.class_names[int(valid_class_ids[0])]
                    self.speak(f"I see a {label}")
                    self.last_speech_time = now
            else:
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)
        except: pass

if __name__ == "__main__":
    VisionApp().run()
