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

                Color(0, 1, 0, 1) 
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=2)

                lbl = Label(text=label_name, pos=(float(x1_px), float(y1_px + h_px)), 
                            size_hint=(None, None), size=(150, 40), color=(0,1,0,1))
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7, 'cup': 10}
        self.FOCAL_LENGTH = 715 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 5 # Adjusted for longer sentences
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        self.top_btn = Button(
            text="MODE 1: MULTI-DETECTION\n(Tap to switch)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='18sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.camera_container = FloatLayout()
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        self.overlay = BBoxOverlay()
        
        self.camera_container.add_widget(self.preview)
        self.camera_container.add_widget(self.overlay)

        self.bottom_btn = Button(
            text="EXIT APPLICATION",
            background_color=(0.8, 0.2, 0.2, 1), font_size='18sp', size_hint_y=0.12
        )
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.camera_container)
        layout.add_widget(self.bottom_btn)
        return layout

    def on_start(self):
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
                Clock.schedule_once(lambda dt: self.tts.setSpeechRate(0.85) if self.tts else None, 1.5)
            except Exception as e:
                Logger.error(f"TTS Error: {e}")

        Clock.schedule_once(self.load_model, 0.5)
        if platform == 'android':
            request_permissions([Permission.CAMERA], self.on_permission_result)
        else:
            self.start_camera()

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
        if all(grants): self.start_camera()

    def start_camera(self):
        Clock.schedule_once(lambda dt: self.preview.connect_camera(camera_id='back'), 1)

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Mode 2: Distance" if self.current_mode == 2 else "Mode 1: Multi-Detection"
        self.speak(msg)
        self.top_btn.text = f"{msg.upper()}\n(Tap to switch)"

    def check_close_app(self, instance):
        self.speak("Closing application")
        self.preview.disconnect_camera()
        Clock.schedule_once(lambda dt: self.stop(), 0.5)

    def speak(self, text):
        if self.tts:
            try: self.tts.speak(text, 0, None)
            except: pass

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return
        try:
            width, height = args[0] if isinstance(args[0], (list, tuple)) else (args[0], args[1])
            channels = len(pixels) // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            img = Image.fromarray(frame[:, :, :3]).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            if self.input_details[0]['shape'][1] == 3: 
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].transpose()
            
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > 0.40
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                sort_idx = np.argsort(valid_scores)[::-1]
                num_to_process = 3 if self.current_mode == 1 else 1
                top_k = min(num_to_process, len(sort_idx))
                
                best_boxes = valid_boxes[sort_idx][:top_k]
                best_classes = valid_class_ids[sort_idx][:top_k]
                
                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(best_boxes, best_classes, self.class_names, self.preview), 0)
                
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    if self.current_mode == 1:
                        # Logic for Multi-Object and Direction
                        descriptions = []
                        for i in range(len(best_boxes)):
                            label = self.class_names[int(best_classes[i])]
                            xc = best_boxes[i][0]
                            # Determine Direction
                            if xc < 213: direction = "on your left"
                            elif xc > 426: direction = "on your right"
                            else: direction = "in front of you"
                            descriptions.append(f"{label} {direction}")
                        
                        full_sentence = "I see a " + ", and a ".join(descriptions)
                        self.speak(full_sentence)
                    else:
                        # Logic for Mode 2: Single Object + Smart Distance
                        label = self.class_names[int(best_classes[0])]
                        xc, yc, w, h = best_boxes[0][:4]
                        width_px = w * (width / 640)
                        dist_cm = (self.KNOWN_WIDTHS.get(label, 30) * self.FOCAL_LENGTH) / max(width_px, 1)
                        
                        if dist_cm < 30.48: # Less than 1 foot
                            dist_str = f"{int(dist_cm)} centimeters"
                        else: # More than 1 foot, convert to feet
                            dist_feet = round(dist_cm / 30.48, 1)
                            unit = "feet" if dist_feet != 1.0 else "foot"
                            dist_str = f"{dist_feet} {unit}"
                            
                        self.speak(f"{label} at {dist_str}")
                    
                    self.last_speech_time = now
            else:
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
