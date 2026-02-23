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
from kivy.uix.slider import Slider
from kivy.graphics import Color, Line, Rectangle
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

                # YOLOv8 format: [cx, cy, w, h]
                xc, yc, w, h = map(float, box[:4])
                
                # Convert 640-space to screen-space
                scale_x = pw / 640.0
                scale_y = ph / 640.0
                
                w_px = w * scale_x
                h_px = h * scale_y
                x1_px = px + ((xc - w/2) * scale_x)
                y1_px = py + ph - ((yc + h/2) * scale_y) 

                Color(0, 1, 0, 1) 
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=2)

                # Label Background
                Color(0, 0, 0, 0.7)
                Rectangle(pos=(x1_px, y1_px + h_px), size=(max(150, w_px), 40))

                lbl = Label(text=label_name.upper(), pos=(x1_px + 5, y1_px + h_px), 
                            size_hint=(None, None), size=(150, 40), color=(1,1,1,1), bold=True)
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {
            'person': 45, 'chair': 50, 'bottle': 8, 'cell phone': 7,
            'car': 180, 'laptop': 35, 'cup': 10, 'backpack': 30, 'book': 15
        }
        # Adjusted Focal Length for typical mobile sensors
        self.FOCAL_LENGTH = 550 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 3.8
        self.detection_threshold = 0.45 
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        self.top_btn = Button(text="MODE: DETECTION", background_color=(0.1, 0.5, 0.8, 1), size_hint_y=0.12)
        self.top_btn.bind(on_release=self.toggle_mode)

        self.camera_container = FloatLayout()
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        self.overlay = BBoxOverlay()
        
        # UI: Sensitivity Slider
        self.slider_box = BoxLayout(orientation='vertical', size_hint=(0.12, 0.6), pos_hint={'x': 0.02, 'center_y': 0.5})
        self.slider_label = Label(text=f"{int(self.detection_threshold * 100)}%", size_hint_y=0.1)
        self.slider = Slider(min=0.1, max=0.9, value=self.detection_threshold, orientation='vertical')
        self.slider.bind(value=self.on_slider_value)
        self.slider_box.add_widget(self.slider_label)
        self.slider_box.add_widget(self.slider)

        self.camera_container.add_widget(self.preview)
        self.camera_container.add_widget(self.overlay)
        self.camera_container.add_widget(self.slider_box)

        self.bottom_btn = Button(text="EXIT", background_color=(0.8, 0.2, 0.2, 1), size_hint_y=0.1)
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.camera_container)
        layout.add_widget(self.bottom_btn)
        return layout

    def on_slider_value(self, instance, value):
        self.detection_threshold = value
        self.slider_label.text = f"{int(value * 100)}%"

    def on_start(self):
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
            except: pass
        Clock.schedule_once(self.load_model, 0.5)
        if platform == 'android':
            request_permissions([Permission.CAMERA], self.on_permission_result)

    def load_model(self, dt):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(cur_dir, "yolov8n_float32.tflite")
        if os.path.exists(model_path):
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def on_permission_result(self, permissions, grants):
        if all(grants): self.preview.connect_camera(camera_id='back')

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "DISTANCE MODE" if self.current_mode == 2 else "DETECTION MODE"
        self.speak(msg)
        self.top_btn.text = f"MODE: {msg}"

    def check_close_app(self, instance):
        self.preview.disconnect_camera()
        self.stop()

    def speak(self, text):
        if self.tts:
            try: self.tts.speak(text, 0, None)
            except: pass

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return
        try:
            width, height = args[0] if isinstance(args[0], (list, tuple)) else (args[0], args[1])
            # Process Buffer
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
            rgb = frame[:, :, :3]
            img = Image.fromarray(rgb).resize((640, 640))
            
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            if self.input_details[0]['shape'][1] == 3:
                input_data = np.transpose(input_data, (0, 3, 1, 2))

            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].transpose()
            
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > self.detection_threshold
            
            if np.any(mask):
                v_boxes = output[mask]
                v_scores = scores[mask]
                v_classes = np.argmax(v_boxes[:, 4:], axis=1)
                
                # NMS logic
                indices = np.argsort(v_scores)[::-1][:5]
                best_boxes = v_boxes[indices]
                best_classes = v_classes[indices]

                # Visual Draw
                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(best_boxes, best_classes, self.class_names, self.preview), 0)

                # Voice Logic
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    if self.current_mode == 1:
                        # Grouping by normalized screen position
                        groups = {"left": [], "right": [], "front": []}
                        for i in range(len(best_boxes)):
                            lbl = self.class_names[int(best_classes[i])]
                            # Use relative X (0.0 to 1.0) to avoid pixel-scaling errors
                            rel_x = best_boxes[i][0] / 640.0 
                            
                            if rel_x < 0.35: groups["left"].append(lbl)
                            elif rel_x > 0.65: groups["right"].append(lbl)
                            else: groups["front"].append(lbl)

                        speech = []
                        for pos, items in groups.items():
                            if items:
                                unique_items = list(set(items))
                                speech.append(f"{' and '.join(unique_items)} on your {pos}")
                        
                        if speech:
                            self.speak("I see " + ", ".join(speech))
                            self.last_speech_time = now

                    elif self.current_mode == 2:
                        # Target closest/center object
                        box = best_boxes[0]
                        label = self.class_names[int(best_classes[0])]
                        # YOLO format: box[2] is width. Ensure it's not too small.
                        w_val = max(box[2], 10) 
                        real_w = self.KNOWN_WIDTHS.get(label, 30)
                        
                        dist_cm = (real_w * self.FOCAL_LENGTH) / w_val
                        
                        # Distance Sanity Check (Max 30 meters / 100 feet)
                        dist_cm = min(dist_cm, 3000)

                        if dist_cm < 30.48:
                            self.speak(f"{label}, {int(dist_cm)} centimeters")
                        else:
                            feet = round(dist_cm / 30.48, 1)
                            self.speak(f"{label}, {feet} feet")
                        self.last_speech_time = now
            else:
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)
        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
