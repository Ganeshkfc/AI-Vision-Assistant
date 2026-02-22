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

# --- New Overlay Class for Bounding Boxes ---
class BBoxOverlay(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels = [] # Keep track of text labels to clear them

    def draw_boxes(self, valid_boxes, valid_class_ids, class_names, preview_widget):
        self.canvas.clear()
        
        # Clear old text labels
        for lbl in self.labels:
            self.remove_widget(lbl)
        self.labels.clear()

        # Get preview dimensions to scale the 640x640 YOLO output to the screen
        pw, ph = preview_widget.size
        px, py = preview_widget.pos

        with self.canvas:
            for i in range(len(valid_boxes)):
                box = valid_boxes[i]
                class_id = valid_class_ids[i]
                label_name = class_names[class_id]

                # YOLOv8 outputs center_x, center_y, width, height relative to 640x640
                xc, yc, w, h = box[:4]
                
                # Scale coordinates to the Kivy widget size
                scale_x = pw / 640.0
                scale_y = ph / 640.0
                
                # Calculate pixel coordinates
                w_px = w * scale_x
                h_px = h * scale_y
                x1_px = px + ((xc - w/2) * scale_x)
                # Kivy y=0 is at the bottom, image y=0 is usually at the top
                y1_px = py + ph - ((yc + h/2) * scale_y) 

                # Draw Rectangle
                Color(0, 1, 0, 1) # Green Box
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=2)

                # Add Text Label
                lbl = Label(text=label_name, pos=(x1_px, y1_px + h_px), size_hint=(None, None), size=(100, 30), color=(0,1,0,1))
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 5 
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        
        # --- UI LAYOUT CHANGES ---
        self.top_btn = Button(
            text="TAP HERE TO CHANGE MODE\n(Mode 1 Active: Multi-Object Detection)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='20sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        # Create a FloatLayout to hold the Camera and the Overlay on top of each other
        self.camera_container = FloatLayout()
        
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        
        self.overlay = BBoxOverlay() # Initialize the overlay
        
        self.camera_container.add_widget(self.preview)
        self.camera_container.add_widget(self.overlay) # Overlay must be added AFTER preview

        self.bottom_btn = Button(
            text="TAP HERE TO CLOSE APP",
            background_color=(0.8, 0.2, 0.2, 1), font_size='20sp', size_hint_y=0.15, halign='center'
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
        else:
            Logger.error(f"MODEL: File not found at {model_path}")

    def on_permission_result(self, permissions, grants):
        if all(grants):
            Logger.info("PERMS: All granted.")
            self.start_camera()
        else:
            Logger.error("PERMS: Denied.")

    def start_camera(self):
        Clock.schedule_once(self._connect_camera, 1)

    def _connect_camera(self, dt):
        try:
            self.preview.connect_camera(camera_id='back', enable_analyze_pixels = True)
            Logger.info("CAMERA: Connected")
            Clock.schedule_once(lambda x: self.speak("Vision Activated"), 2)
        except Exception as e:
            Logger.error(f"CAMERA: Error {e}")

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Mode 2: Distance" if self.current_mode == 2 else "Mode 1: Detection"
        self.speak(msg)
        self.top_btn.text = msg

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

    def get_distance_cm(self, label, width_px):
        real_w = self.KNOWN_WIDTHS.get(label, 30)
        return (real_w * self.FOCAL_LENGTH) / max(width_px, 1)

    def analyze_frame(self, pixels, width, height, image_pos, image_size, texture):
        if not self.interpreter:
            return

        try:
            channels = len(pixels) // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            rgb = frame[:, :, :3] 
            
            # 1. Resize for YOLOv8
            img = Image.fromarray(rgb).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            if self.input_details[0]['shape'][1] == 3:
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            # 2. Run Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # 3. FAST VECTORIZED PROCESSING
            output = output.transpose() 
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > 0.45 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                # Sort by confidence to draw the best boxes
                sort_idx = np.argsort(valid_scores)[::-1]
                
                # Cap at top 5 boxes to prevent screen clutter/lag
                top_k = min(5, len(sort_idx))
                best_boxes = valid_boxes[sort_idx][:top_k]
                best_classes = valid_class_ids[sort_idx][:top_k]
                
                # --- UPDATE THE GUI OVERLAY ---
                # We use Clock.schedule_once because Kivy UI updates MUST happen on the main thread, 
                # but analyze_frame runs in a background thread.
                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(best_boxes, best_classes, self.class_names, self.preview), 0)
                
                # 4. Handle Speech
                top_label = self.class_names[best_classes[0]]
                xc, yc, w, h = best_boxes[0][:4]
                width_px = w * (width / 640)

                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    if self.current_mode == 1:
                        self.speak(f"I see a {top_label}")
                    else:
                        dist = self.get_distance_cm(top_label, width_px)
                        self.speak(f"{top_label} at {int(dist)} centimeters")
                    self.last_speech_time = now
            else:
                # Clear boxes if nothing is detected
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)
                Clock.schedule_once(lambda dt: [self.overlay.remove_widget(l) for l in self.overlay.labels], 0)
                self.overlay.labels.clear()

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
