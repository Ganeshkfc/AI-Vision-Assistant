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

                lbl = Label(text=label_name, pos=(float(x1_px), float(y1_px + h_px)), 
                            size_hint=(None, None), size=(150, 40), color=(0,1,0,1))
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        # Greatly expanded known widths (in cm) for accurate distance
        self.KNOWN_WIDTHS = {
            'person': 45, 'chair': 45, 'bottle': 8, 'cell phone': 7,
            'car': 180, 'bicycle': 60, 'motorcycle': 70, 'airplane': 3000,
            'bus': 250, 'train': 300, 'truck': 250, 'boat': 200,
            'traffic light': 25, 'fire hydrant': 20, 'stop sign': 60,
            'bench': 120, 'bird': 15, 'cat': 25, 'dog': 30, 'horse': 80,
            'sheep': 60, 'cow': 80, 'elephant': 200, 'bear': 100,
            'zebra': 120, 'giraffe': 100, 'backpack': 30, 'umbrella': 100,
            'handbag': 25, 'tie': 10, 'suitcase': 50, 'frisbee': 25,
            'skis': 10, 'snowboard': 25, 'sports ball': 22, 'kite': 50,
            'baseball bat': 5, 'baseball glove': 25, 'skateboard': 20,
            'surfboard': 50, 'tennis racket': 25, 'wine glass': 8,
            'cup': 8, 'fork': 3, 'knife': 3, 'spoon': 3, 'bowl': 15,
            'banana': 20, 'apple': 8, 'sandwich': 15, 'orange': 8,
            'broccoli': 10, 'carrot': 15, 'hot dog': 15, 'pizza': 30,
            'donut': 10, 'cake': 25, 'couch': 180, 'potted plant': 30,
            'bed': 150, 'dining table': 120, 'toilet': 40, 'tv': 80,
            'laptop': 35, 'mouse': 6, 'remote': 5, 'keyboard': 40,
            'microwave': 50, 'oven': 60, 'toaster': 25, 'sink': 50,
            'refrigerator': 70, 'book': 15, 'clock': 20, 'vase': 15,
            'scissors': 10, 'teddy bear': 30, 'hair drier': 15, 'toothbrush': 2
        }
        # Calibrated strictly for the AI's 640x640 matrix, ignoring screen size distortions
        self.FOCAL_LENGTH_640 = 550 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 4 
        self.detection_threshold = 0.35 
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        
        self.top_btn = Button(
            text="TAP HERE TO CHANGE MODE\n(Mode 1: Detection Active)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='20sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.camera_container = FloatLayout()
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        
        self.overlay = BBoxOverlay()
        
        # UI Container for Slider and Text
        self.slider_box = BoxLayout(orientation='vertical', size_hint=(0.15, 0.8), pos_hint={'x': 0.0, 'center_y': 0.5})
        self.slider_label = Label(text=f"{int(self.detection_threshold * 100)}%", size_hint_y=0.1, font_size='18sp', bold=True)
        self.slider = Slider(min=0.1, max=0.9, value=self.detection_threshold, orientation='vertical', size_hint_y=0.9)
        self.slider.bind(value=self.on_slider_value)
        
        self.slider_box.add_widget(self.slider_label)
        self.slider_box.add_widget(self.slider)

        self.camera_container.add_widget(self.preview)
        self.camera_container.add_widget(self.overlay)
        self.camera_container.add_widget(self.slider_box)

        self.bottom_btn = Button(
            text="TAP HERE TO CLOSE APP",
            background_color=(0.8, 0.2, 0.2, 1), font_size='20sp', size_hint_y=0.15, halign='center'
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
                Clock.schedule_once(lambda dt: self.tts.setSpeechRate(0.7) if self.tts else None, 1.5)
            except Exception as e:
                Logger.error(f"TTS Initialization Error: {e}")

        Clock.schedule_once(self.load_model, 0.5)
        if platform == 'android':
            perms = [Permission.CAMERA, Permission.RECORD_AUDIO]
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

    def on_permission_result(self, permissions, grants):
        if all(grants):
            self.start_camera()

    def start_camera(self):
        Clock.schedule_once(self._connect_camera, 1)

    def _connect_camera(self, dt):
        try:
            self.preview.connect_camera(camera_id='back', enable_analyze_pixels=True)
            Clock.schedule_once(lambda x: self.speak("Vision Activated"), 2)
        except Exception as e:
            Logger.error(f"CAMERA: Error {e}")

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Mode 2: Distance" if self.current_mode == 2 else "Mode 1: Detection"
        self.speak(msg)
        self.top_btn.text = f"TAP TO CHANGE MODE\n({msg} Active)"

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

    def get_distance_cm(self, label, box_w_640):
        # Uses the unscaled 640-space width for hyper-accurate distance
        real_w = self.KNOWN_WIDTHS.get(label, 30) 
        return (real_w * self.FOCAL_LENGTH_640) / max(box_w_640, 1)

    def non_max_suppression(self, boxes, scores, iou_threshold=0.4):
        # Cleans up overlapping duplicate bounding boxes mathematically
        if len(boxes) == 0:
            return []
        
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
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

    def analyze_frame(self, pixels, *args):
        if not self.interpreter:
            return

        try:
            if isinstance(args[0], (list, tuple)):
                width, height = args[0]
            else:
                width = args[0]
                height = args[1]

            channels = len(pixels) // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            rgb = frame[:, :, :3] 
            
            img = Image.fromarray(rgb).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            if self.input_details[0]['shape'][1] == 3: 
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            output = output.transpose() 
            scores = np.max(output[:, 4:], axis=1)
            
            mask = scores > self.detection_threshold 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                # Apply NMS to remove duplicate overlapping boxes
                keep_idx = self.non_max_suppression(valid_boxes, valid_scores, iou_threshold=0.4)
                best_boxes = valid_boxes[keep_idx]
                best_classes = valid_class_ids[keep_idx]
                
                # Limit to Top 5 distinct objects
                top_k = min(5, len(best_boxes))
                best_boxes = best_boxes[:top_k]
                best_classes = best_classes[:top_k]
                
                if self.current_mode == 2:
                    # Find strictly the most central object
                    center_x, center_y = 320.0, 320.0
                    min_dist_center = float('inf')
                    best_idx = -1
                    for i in range(len(best_boxes)):
                        xc, yc = best_boxes[i][0], best_boxes[i][1]
                        dist_to_center = ((xc - center_x) ** 2 + (yc - center_y) ** 2) ** 0.5
                        if dist_to_center < min_dist_center:
                            min_dist_center = dist_to_center
                            best_idx = i

                    if best_idx != -1:
                        best_boxes = np.array([best_boxes[best_idx]])
                        best_classes = np.array([best_classes[best_idx]])

                boxes_to_draw = np.copy(best_boxes)
                classes_to_draw = np.copy(best_classes)
                Clock.schedule_once(lambda dt, b=boxes_to_draw, c=classes_to_draw: self.overlay.draw_boxes(b, c, self.class_names, self.preview), 0)
                
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    
                    if self.current_mode == 1:
                        phrases = []
                        seen_objects = set()
                        
                        for i in range(min(3, len(best_boxes))):
                            box = best_boxes[i]
                            cls_id = int(best_classes[i])
                            label = self.class_names[cls_id]
                            xc = box[0]
                            
                            if xc < 640 / 3.0:
                                direction = "on your left"
                            elif xc > 2 * 640 / 3.0:
                                direction = "on your right"
                            else:
                                direction = "in front of you"
                                
                            phrase = f"a {label} {direction}"
                            
                            # Prevents repeating "chair on your left" multiple times
                            if phrase not in seen_objects:
                                phrases.append(phrase)
                                seen_objects.add(phrase)

                        if phrases:
                            speech_text = "I see " + " and ".join(phrases)
                            self.speak(speech_text)
                            self.last_speech_time = now
                    
                    elif self.current_mode == 2:
                        box = best_boxes[0]
                        top_label = self.class_names[int(best_classes[0])]
                        w = box[2] # Native 640 space width
                        
                        dist_cm = self.get_distance_cm(top_label, w)
                        
                        if dist_cm < 30.48:  
                            self.speak(f"I see a {top_label} {int(dist_cm)} centimeters in front of you")
                        else:
                            dist_ft = dist_cm / 30.48
                            # Uses 1 decimal place (e.g., 2.5 feet) for higher precision
                            self.speak(f"I see a {top_label} {round(dist_ft, 1)} feet in front of you")
                            
                        self.last_speech_time = now
            else:
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)
                def clear_labels(dt):
                    for l in self.overlay.labels:
                        self.overlay.remove_widget(l)
                    self.overlay.labels.clear()
                Clock.schedule_once(clear_labels, 0)

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
