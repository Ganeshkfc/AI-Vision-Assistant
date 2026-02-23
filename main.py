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

        # Force sizes to floats to avoid list-multiplication errors
        pw = float(preview_widget.width)
        ph = float(preview_widget.height)
        px = float(preview_widget.x)
        py = float(preview_widget.y)

        with self.canvas:
            for i in range(len(valid_boxes)):
                box = valid_boxes[i]
                class_id = int(valid_class_ids[i])
                label_name = class_names[class_id] if class_id < len(class_names) else "Object"

                # Map 640x640 model coordinates to screen coordinates
                xc, yc, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                scale_x = pw / 640.0
                scale_y = ph / 640.0
                
                w_px = w * scale_x
                h_px = h * scale_y
                x1_px = px + ((xc - w/2.0) * scale_x)
                y1_px = py + ph - ((yc + h/2.0) * scale_y) 

                Color(0, 1, 0, 1) # Green boxes
                Line(rectangle=(float(x1_px), float(y1_px), float(w_px), float(h_px)), width=2)

                lbl = Label(text=label_name.upper(), pos=(float(x1_px), float(y1_px + h_px)), 
                            size_hint=(None, None), size=(200, 40), color=(0,1,0,1), bold=True)
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.sensitivity = 0.50 
        self.KNOWN_WIDTHS = {'person': 45, 'chair': 50, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 720 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 4.0 
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        main_layout = BoxLayout(orientation='vertical')
        
        self.top_btn = Button(
            text="AI VISION: STARTING...",
            background_color=(0.1, 0.4, 0.8, 1), font_size='18sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.camera_area = FloatLayout()
        self.preview = Preview(aspect_ratio='16:9')
        self.overlay = BBoxOverlay()
        
        slider_layout = BoxLayout(orientation='vertical', size_hint=(0.15, 0.5), pos_hint={'right': 1, 'center_y': 0.5}, padding=5)
        self.sens_label = Label(text=f"SENS\n{int(self.sensitivity*100)}%", size_hint_y=0.2)
        self.slider = Slider(min=0.1, max=0.9, value=self.sensitivity, orientation='vertical')
        self.slider.bind(value=self.on_slider_change)
        slider_layout.add_widget(self.sens_label)
        slider_layout.add_widget(self.slider)

        self.camera_area.add_widget(self.preview)
        self.camera_area.add_widget(self.overlay)
        self.camera_area.add_widget(slider_layout)

        main_layout.add_widget(self.top_btn)
        main_layout.add_widget(self.camera_area)
        return main_layout

    def on_slider_change(self, instance, value):
        self.sensitivity = float(value)
        self.sens_label.text = f"SENS\n{int(value*100)}%"

    def on_start(self):
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
            except: pass
        Clock.schedule_once(self.load_model, 1.0)
        if platform == 'android':
            request_permissions([Permission.CAMERA], self.on_permission_result)
        else:
            self.start_camera()

    def load_model(self, dt):
        try:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(cur_dir, "yolov8n_float32.tflite")
            if os.path.exists(model_path):
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.top_btn.text = "MODE 1: MULTI-DETECTION\n(Tap to switch)"
                Logger.info("MODEL: Loaded Successfully!")
        except Exception as e:
            Logger.error(f"MODEL_LOAD_ERROR: {e}")

    def on_permission_result(self, permissions, grants):
        if all(grants): self.start_camera()

    def start_camera(self):
        self.preview.analyze_pixels_callback = self.analyze_frame
        Clock.schedule_once(lambda dt: self.preview.connect_camera(enable_analyze_pixels=True), 1.0)

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Distance Mode" if self.current_mode == 2 else "Multi Detection"
        self.top_btn.text = f"MODE {self.current_mode}: {msg.upper()}\n(Tap to switch)"

    def speak(self, text):
        if self.tts:
            try: self.tts.speak(text, 0, None)
            except: pass

    def analyze_frame(self, pixels, width, height, rotation, *args):
        if not self.interpreter: return
        try:
            # Prepare image
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
            img = Image.fromarray(frame[:, :, :3]).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            # YOLOv8 format check
            if self.input_details[0]['shape'][1] == 3: 
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get raw output and force to float32 NumPy array immediately
            raw_output = self.interpreter.get_tensor(self.output_details[0]['index'])
            output = np.asarray(raw_output, dtype=np.float32)[0]
            
            # YOLOv8 usually outputs (84, 8400). Transpose to (8400, 84)
            if output.shape[0] < output.shape[1]:
                output = output.transpose()
            
            # Extract scores and filter by sensitivity
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > self.sensitivity 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                indices = self.apply_simple_nms(valid_boxes, valid_scores, 0.45)
                if len(indices) > 0:
                    final_boxes = valid_boxes[indices]
                    final_classes = valid_class_ids[indices]
                    
                    Clock.schedule_once(lambda dt: self.overlay.draw_boxes(final_boxes, final_classes, self.class_names, self.preview), 0)
                    
                    # Logic for Speech (limited rate)
                    now = time.time()
                    if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                        label = self.class_names[int(final_classes[0])]
                        self.speak(f"I see a {label}")
                        self.last_speech_time = now
            else:
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

    def apply_simple_nms(self, boxes, scores, iou_thresh):
        if len(boxes) == 0: return []
        idxs = np.argsort(scores)[::-1]
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1: break
            rest = idxs[1:]
            ious = self.box_iou(boxes[i], boxes[rest])
            idxs = rest[ious < iou_thresh]
        return keep

    def box_iou(self, box1, boxes):
        # Convert all to strictly NumPy float arrays
        b1 = np.asarray(box1, dtype=np.float32)
        bs = np.asarray(boxes, dtype=np.float32)
        if bs.ndim == 1: bs = np.expand_dims(bs, axis=0)

        x1 = np.maximum(b1[0] - b1[2]/2.0, bs[:,0] - bs[:,2]/2.0)
        y1 = np.maximum(b1[1] - b1[3]/2.0, bs[:,1] - bs[:,3]/2.0)
        x2 = np.minimum(b1[0] + b1[2]/2.0, bs[:,0] + bs[:,2]/2.0)
        y2 = np.minimum(b1[1] + b1[3]/2.0, bs[:,1] + bs[:,3]/2.0)
        
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area1 = b1[2] * b1[3]
        area2 = bs[:,2] * bs[:,3]
        union = area1 + area2 - inter
        return inter / (union + 1e-6)

if __name__ == "__main__":
    VisionApp().run()
