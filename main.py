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

                xc, yc, w, h = map(float, box[:4])
                
                scale_x = pw / 640.0
                scale_y = ph / 640.0
                
                w_px = w * scale_x
                h_px = h * scale_y
                x1_px = px + ((xc - w/2) * scale_x)
                y1_px = py + ph - ((yc + h/2) * scale_y) 

                Color(0, 1, 0, 1) 
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=2)

                Color(0, 0, 0, 0.6)
                Rectangle(pos=(x1_px, y1_px + h_px), size=(180, 45))

                lbl = Label(text=label_name.upper(), pos=(float(x1_px + 5), float(y1_px + h_px + 2)), 
                            size_hint=(None, None), size=(170, 40), color=(1,1,1,1), font_size='14sp', bold=True)
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {
            'person': 45, 'chair': 50, 'bottle': 8, 'cell phone': 7,
            'car': 180, 'bicycle': 60, 'laptop': 35, 'tv': 80, 'cup': 10,
            'door': 90, 'table': 100, 'backpack': 30, 'book': 15
        }
        # Fixed Focal Length for 640px sensor width
        self.FOCAL_LENGTH_640 = 620 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 3.5 
        self.detection_threshold = 0.45 
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        
        self.top_btn = Button(
            text="TAP TO CHANGE MODE\n(Mode 1: Detection Active)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='18sp', size_hint_y=0.12, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.camera_container = FloatLayout()
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        
        self.overlay = BBoxOverlay()
        
        self.slider_box = BoxLayout(orientation='vertical', size_hint=(0.15, 0.7), pos_hint={'x': 0.02, 'center_y': 0.5})
        self.slider_label = Label(text=f"{int(self.detection_threshold * 100)}%", size_hint_y=0.1, bold=True)
        self.slider = Slider(min=0.1, max=0.9, value=self.detection_threshold, orientation='vertical', size_hint_y=0.9)
        self.slider.bind(value=self.on_slider_value)
        
        self.slider_box.add_widget(self.slider_label)
        self.slider_box.add_widget(self.slider)

        self.camera_container.add_widget(self.preview)
        self.camera_container.add_widget(self.overlay)
        self.camera_container.add_widget(self.slider_box)

        self.bottom_btn = Button(
            text="CLOSE APP",
            background_color=(0.8, 0.2, 0.2, 1), font_size='18sp', size_hint_y=0.1, halign='center'
        )
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
        if os.path.exists(model_path) and tflite:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

    def on_permission_result(self, permissions, grants):
        if all(grants):
            self.preview.connect_camera(camera_id='back')

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Distance Mode" if self.current_mode == 2 else "Detection Mode"
        self.speak(msg)
        self.top_btn.text = f"TAP TO CHANGE MODE\n({msg} Active)"

    def check_close_app(self, instance):
        self.preview.disconnect_camera()
        self.stop()

    def speak(self, text):
        if self.tts:
            try: self.tts.speak(text, 0, None)
            except: pass

    def non_max_suppression(self, boxes, scores, iou_threshold=0.45):
        if len(boxes) == 0: return []
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            iou = (w * h) / (areas[i] + areas[order[1:]] - (w * h))
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return keep

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return
        try:
            width, height = args[0] if isinstance(args[0], (list, tuple)) else (args[0], args[1])
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
                v_boxes, v_scores = output[mask], scores[mask]
                v_classes = np.argmax(v_boxes[:, 4:], axis=1)
                keep = self.non_max_suppression(v_boxes, v_scores)
                best_boxes, best_classes = v_boxes[keep][:5], v_classes[keep][:5]

                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(best_boxes, best_classes, self.class_names, self.preview), 0)
                
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    if self.current_mode == 1:
                        groups = {"on your left": [], "on your right": [], "in front of you": []}
                        for i in range(len(best_boxes)):
                            label = self.class_names[int(best_classes[i])]
                            xc = best_boxes[i][0]
                            # Direction Logic (Based on 640px width)
                            if xc < 180: dir_key = "on your left"
                            elif xc > 460: dir_key = "on your right"
                            else: dir_key = "in front of you"
                            if label not in groups[dir_key]: groups[dir_key].append(label)

                        speech_parts = []
                        for d_text, items in groups.items():
                            if items:
                                speech_parts.append(f"{' and '.join(items)} {d_text}")
                        
                        if speech_parts:
                            self.speak("I see " + ", ".join(speech_parts))
                            self.last_speech_time = now
                    
                    elif self.current_mode == 2:
                        # Target object closest to center
                        dists = [abs(b[0] - 320) for b in best_boxes]
                        idx = np.argmin(dists)
                        box, cls = best_boxes[idx], best_classes[idx]
                        label = self.class_names[int(cls)]
                        
                        # Distance Math
                        real_w = self.KNOWN_WIDTHS.get(label, 30)
                        # Avoid division by very small width (fixes the 800ft bug)
                        pix_w = max(box[2], 20) 
                        dist_cm = (real_w * self.FOCAL_LENGTH_640) / pix_w
                        
                        # Unit conversion
                        if dist_cm < 60:
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
