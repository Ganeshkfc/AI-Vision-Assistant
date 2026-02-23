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
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=5)

                lbl = Label(text=label_name.upper(), pos=(float(x1_px), float(y1_px + h_px)), 
                            size_hint=(None, None), size=(200, 50), color=(0,1,0,1), bold=True, font_size='18sp')
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.sensitivity = 0.40 # Current Threshold
        self.KNOWN_WIDTHS = {'person': 45, 'chair': 50, 'bottle': 7, 'cell phone': 7, 'cup': 9, 'laptop': 32, 'tv': 80}
        self.FOCAL_LENGTH = 720 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 5.0 
        self.frame_count = 0
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        main_layout = BoxLayout(orientation='vertical')

        # --- TOP BUTTON: MODE SWITCHER ---
        self.top_btn = Button(
            text="AI VISION: STARTING...",
            background_color=(0.1, 0.4, 0.8, 1), font_size='18sp', size_hint_y=0.12, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)
        main_layout.add_widget(self.top_btn)

        # --- SENSITIVITY SLIDER ---
        slider_layout = BoxLayout(orientation='vertical', size_hint_y=0.12, padding=[15, 5])
        self.slider_label = Label(text=f"Detection Sensitivity: {int(self.sensitivity * 100)}%", font_size='14sp')
        self.sens_slider = Slider(min=0.05, max=0.95, value=0.40, step=0.05)
        self.sens_slider.bind(value=self.on_slider_change)
        
        slider_layout.add_widget(self.slider_label)
        slider_layout.add_widget(self.sens_slider)
        main_layout.add_widget(slider_layout)

        # --- CAMERA AREA ---
        self.camera_container = FloatLayout(size_hint_y=0.66)
        self.preview = Preview() 
        self.overlay = BBoxOverlay()
        self.camera_container.add_widget(self.preview)
        self.camera_container.add_widget(self.overlay)
        main_layout.add_widget(self.camera_container)

        # --- EXIT BUTTON ---
        self.bottom_btn = Button(text="EXIT APP", background_color=(0.8, 0.1, 0.1, 1), size_hint_y=0.10)
        self.bottom_btn.bind(on_release=self.check_close_app)
        main_layout.add_widget(self.bottom_btn)

        return main_layout

    # --- AUTO-START SEQUENCE ---
    def on_start(self):
        Clock.schedule_once(self.load_model, 0.5)
        if platform == 'android':
            self.init_tts()
            request_permissions([Permission.CAMERA], self.on_permission_result)
        else:
            self.start_camera()

    def init_tts(self):
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
            
            # Wait a moment for TTS engine to initialize before setting rate or speaking
            Clock.schedule_once(lambda dt: self.tts.setSpeechRate(0.8) if self.tts else None, 2.0)
            Clock.schedule_once(lambda dt: self.speak("System ready."), 3.0)
        except Exception as e:
            Logger.error(f"TTS_INIT_ERROR: {e}")

    def speak(self, text):
        if self.tts:
            try: 
                self.tts.speak(text, 0, None)
                Logger.info(f"TTS_SPOKE: {text}")
            except Exception as e: 
                Logger.error(f"TTS_SPEAK_ERROR: {e}")

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
            self.start_camera()
        else:
            self.speak("Camera permission required.")
            self.top_btn.text = "PERMISSION DENIED"

    def start_camera(self):
        Logger.info("CAMERA: Attempting connection...")
        self.preview.analyze_callback = self.analyze_frame
        # Using late connection for stability + filepath buffer trick
        Clock.schedule_once(lambda dt: self.preview.connect_camera(enable_analyze_pixels=True, filepath='test.png'), 1.5)
        self.top_btn.text = "MODE 1: MULTI-OBJECT\n(Tap to switch)"

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Distance Mode" if self.current_mode == 2 else "Object Mode"
        self.speak(msg)
        self.top_btn.text = f"{msg.upper()}\n(Tap to switch)"

    def on_slider_change(self, instance, value):
        self.sensitivity = value
        self.slider_label.text = f"Detection Sensitivity: {int(value * 100)}%"

    def analyze_frame(self, pixels, width, height, rotation):
        if self.frame_count == 0:
            Logger.info(f"AI_FIRST_FRAME: Received {width}x{height}")
            self.speak("Vision Active.")
        
        self.frame_count += 1
        
        # Performance Saver: Process only 1 out of every 3 frames to stop lag
        if self.frame_count % 3 != 0: return 
        
        if not self.interpreter: return

        try:
            n_pixels = len(pixels)
            channels = 4 if n_pixels == (width * height * 4) else 3
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            
            img_np = frame[:, :, :3]
            img = Image.fromarray(img_np).resize((640, 640))
            
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
            mask = scores > self.sensitivity 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                selected = []
                indices = np.argsort(valid_scores)[::-1]
                for i in indices:
                    box_i = valid_boxes[i]
                    is_duplicate = False
                    for j in selected:
                        if np.linalg.norm(box_i[:2] - valid_boxes[j][:2]) < 60: 
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
                    descriptions = []
                    for i in range(len(final_boxes)):
                        label = self.class_names[int(final_classes[i])]
                        xc = final_boxes[i][0]
                        pos = "left" if xc < 210 else "right" if xc > 430 else "center"
                        
                        if self.current_mode == 1:
                            descriptions.append(f"{label} {pos}")
                        else:
                            w_px = final_boxes[i][2]
                            dist_cm = (self.KNOWN_WIDTHS.get(label, 30) * self.FOCAL_LENGTH) / max(w_px, 1)
                            dist_str = f"{int(dist_cm)} centimeters" if dist_cm < 100 else f"{round(dist_cm/30.48, 1)} feet"
                            descriptions.append(f"{label} at {dist_str}")
                    
                    if descriptions:
                        self.speak("I see: " + ", ".join(descriptions))
                        self.last_speech_time = now
            else:
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)

        except Exception as e:
            pass

    def check_close_app(self, instance):
        self.speak("Exiting app.")
        self.preview.disconnect_camera()
        Clock.schedule_once(lambda dt: self.stop(), 0.5)

if __name__ == "__main__":
    VisionApp().run()
