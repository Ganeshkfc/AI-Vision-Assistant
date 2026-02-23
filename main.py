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
            # Draw visual guides for horizontal positioning
            Color(1, 1, 1, 0.2)
            Line(points=[px + pw*0.33, py, px + pw*0.33, py + ph], width=1, dash_length=5)
            Line(points=[px + pw*0.66, py, px + pw*0.66, py + ph], width=1, dash_length=5)

            for i in range(len(valid_boxes)):
                box = valid_boxes[i]
                class_id = int(valid_class_ids[i])
                label_name = class_names[class_id]

                # Ensure coordinates are treated as floats
                xc, yc, w, h = map(float, box[:4])
                scale_x, scale_y = pw / 640.0, ph / 640.0
                
                w_px, h_px = w * scale_x, h * scale_y
                x1_px = px + ((xc - w/2) * scale_x)
                y1_px = py + ph - ((yc + h/2) * scale_y) 

                Color(0, 1, 0, 1) 
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=3)

                lbl = Label(text=label_name.upper(), pos=(float(x1_px), float(y1_px + h_px)), 
                            size_hint=(None, None), size=(200, 50), color=(0,1,0,1), bold=True)
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.sensitivity = 0.45 
        self.KNOWN_WIDTHS = {'person': 45, 'chair': 50, 'bottle': 8, 'cell phone': 7, 'cup': 9, 'laptop': 32, 'tv': 80}
        self.FOCAL_LENGTH = 720 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 5.0 
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        main_layout = BoxLayout(orientation='vertical')
        
        self.top_btn = Button(
            text="MODE 1: MULTI-DETECTION\n(Tap to switch)",
            background_color=(0.1, 0.4, 0.8, 1), font_size='18sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.camera_area = FloatLayout()
        self.preview = Preview(aspect_ratio='16:9')
        self.overlay = BBoxOverlay()
        
        slider_layout = BoxLayout(orientation='vertical', size_hint=(0.15, 0.6), pos_hint={'right': 1, 'center_y': 0.5}, padding=10)
        self.sens_label = Label(text=f"SENS\n{int(self.sensitivity*100)}%", size_hint_y=0.2, color=(1,1,1,1), bold=True)
        self.slider = Slider(min=0.1, max=0.9, value=self.sensitivity, orientation='vertical', value_track=True, value_track_color=[0, 1, 0, 1])
        self.slider.bind(value=self.on_slider_change)
        
        slider_layout.add_widget(self.sens_label)
        slider_layout.add_widget(self.slider)

        self.camera_area.add_widget(self.preview)
        self.camera_area.add_widget(self.overlay)
        self.camera_area.add_widget(slider_layout)

        self.bottom_btn = Button(
            text="EXIT APPLICATION",
            background_color=(0.8, 0.1, 0.1, 1), font_size='18sp', size_hint_y=0.12
        )
        self.bottom_btn.bind(on_release=self.check_close_app)

        main_layout.add_widget(self.top_btn)
        main_layout.add_widget(self.camera_area)
        main_layout.add_widget(self.bottom_btn)
        return main_layout

    def on_slider_change(self, instance, value):
        self.sensitivity = value
        self.sens_label.text = f"SENS\n{int(value*100)}%"

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
        self.preview.analyze_pixels_callback = self.analyze_frame
        Clock.schedule_once(lambda dt: self.preview.connect_camera(enable_analyze_pixels=True), 1.0)

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Distance Mode" if self.current_mode == 2 else "Multi Detection"
        self.speak(msg)
        self.top_btn.text = f"MODE {self.current_mode}: {msg.upper()}\n(Tap to switch)"

    def check_close_app(self, instance):
        self.speak("Closing")
        self.preview.disconnect_camera()
        Clock.schedule_once(lambda dt: self.stop(), 0.5)

    def speak(self, text):
        if self.tts:
            try: self.tts.speak(text, 0, None)
            except: pass

    def analyze_frame(self, pixels, width, height, rotation, *args):
        if not self.interpreter: return
        try:
            # 1. Image Pre-processing
            channels = len(pixels) // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            img = Image.fromarray(frame[:, :, :3]).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            # Check if model expects [batch, channels, height, width]
            if self.input_details[0]['shape'][1] == 3: 
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            # 2. Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # 3. Post-processing (Fixed to prevent "multiply sequence by non-int")
            output = np.array(self.interpreter.get_tensor(self.output_details[0]['index']))
            if len(output.shape) == 3:
                output = output[0]  # Remove batch dim
            
            # YOLOv8 standard: (Features, Proposals) -> (Proposals, Features)
            if output.shape[0] < output.shape[1]:
                output = output.transpose()
            
            # Calculate Scores and Mask
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > self.sensitivity 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                # Apply NMS
                indices = self.apply_simple_nms(valid_boxes, valid_scores, 0.45)
                final_boxes = valid_boxes[indices]
                final_classes = valid_class_ids[indices]
                
                # Update UI
                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(
                    final_boxes, final_classes, self.class_names, self.preview), 0)
                
                # 4. Voice Feedback Logic
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    descriptions = []
                    for i in range(min(len(final_boxes), 3)):
                        label = self.class_names[int(final_classes[i])]
                        xc = float(final_boxes[i][0])
                        
                        if xc < 213: pos = "on your left"
                        elif xc > 426: pos = "on your right"
                        else: pos = "straight ahead"

                        if self.current_mode == 1:
                            descriptions.append(f"{label} {pos}")
                        else:
                            w_px = float(final_boxes[i][2])
                            real_w = self.KNOWN_WIDTHS.get(label, 35)
                            dist_cm = (real_w * self.FOCAL_LENGTH) / max(w_px, 1)
                            
                            if dist_cm < 90:
                                d_str = f"{int(dist_cm)} centimeters"
                            else:
                                feet = round(dist_cm / 30.48, 1)
                                d_str = f"{feet} feet"
                            descriptions.append(f"{label} at {d_str}")
                            break 

                    if descriptions:
                        self.speak("I see " + " and ".join(descriptions))
                        self.last_speech_time = now
            else:
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

    def apply_simple_nms(self, boxes, scores, iou_thresh):
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
        # Force strict NumPy casting to prevent sequence errors
        box1 = np.array(box1, dtype=np.float32)
        boxes = np.array(boxes, dtype=np.float32)

        x1 = np.maximum(box1[0] - box1[2]/2, boxes[:,0] - boxes[:,2]/2)
        y1 = np.maximum(box1[1] - box1[3]/2, boxes[:,1] - boxes[:,3]/2)
        x2 = np.minimum(box1[0] + box1[2]/2, boxes[:,0] + boxes[:,2]/2)
        y2 = np.minimum(box1[1] + box1[3]/2, boxes[:,1] + boxes[:,3]/2)
        
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = boxes[:,2] * boxes[:,3]
        union = area1 + area2 - inter
        return inter / (union + 1e-6)

if __name__ == "__main__":
    VisionApp().run()
