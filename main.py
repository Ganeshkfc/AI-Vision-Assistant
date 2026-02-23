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
        Logger.error("TFLite: Module NOT found!")

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
                Color(0, 0, 0, 0.6)
                Rectangle(pos=(x1_px, y1_px + h_px), size=(180, 45))
                lbl = Label(text=label_name.upper(), pos=(float(x1_px + 5), float(y1_px + h_px + 2)), 
                            size_hint=(None, None), size=(170, 40), color=(1,1,1,1), bold=True)
                self.add_widget(lbl)
                self.labels.append(lbl)

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        self.METRIC_THRESHOLD_CM = 91.44
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 6 
        self.detection_threshold = 0.40 
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        self.top_btn = Button(text="MODE 1 ACTIVE", background_color=(0.1, 0.5, 0.8, 1), font_size='20sp', size_hint_y=0.15)
        self.top_btn.bind(on_release=self.toggle_mode)

        self.camera_container = FloatLayout()
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        self.overlay = BBoxOverlay()
        
        self.camera_container.add_widget(self.preview)
        self.camera_container.add_widget(self.overlay)

        self.bottom_btn = Button(text="CLOSE APP", background_color=(0.8, 0.2, 0.2, 1), font_size='20sp', size_hint_y=0.15)
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.camera_container)
        layout.add_widget(self.bottom_btn)
        return layout

    def on_start(self):
        Logger.info("AI_HEARTBEAT: App Started")
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
            except Exception as e: Logger.error(f"TTS_ERROR: {e}")
        
        Clock.schedule_once(self.load_model, 1.0)
        if platform == 'android':
            request_permissions([Permission.CAMERA], self.on_permission_result)

    def load_model(self, dt):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # Try finding any .tflite file in the directory
        model_files = [f for f in os.listdir(cur_dir) if f.endswith('.tflite')]
        Logger.info(f"AI_HEARTBEAT: Files found in root: {os.listdir(cur_dir)}")
        
        if model_files and tflite:
            model_path = os.path.join(cur_dir, model_files[0])
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                Logger.info(f"AI_HEARTBEAT: Model {model_files[0]} loaded successfully!")
            except Exception as e:
                Logger.error(f"AI_ERROR: Model failed to load: {e}")
        else:
            Logger.error("AI_HEARTBEAT: No .tflite file found in app directory!")

    def on_permission_result(self, permissions, grants):
        if all(grants):
            Logger.info("AI_HEARTBEAT: Camera permission granted")
            self.preview.connect_camera(camera_id='back')

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Mode 2" if self.current_mode == 2 else "Mode 1"
        self.top_btn.text = f"{msg} ACTIVE"

    def check_close_app(self, instance):
        self.preview.disconnect_camera()
        self.stop()

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return
        try:
            # Get frame data
            width, height = args[0] if isinstance(args[0], (list, tuple)) else (args[0], args[1])
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
            rgb = frame[:, :, :3]
            img = Image.fromarray(rgb).resize((640, 640))
            
            # Prepare Input
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            # FIX: Automatic check for NCHW vs NHWC
            if self.input_details[0]['shape'][1] == 3: # If model wants [1, 3, 640, 640]
                input_data = np.transpose(input_data, (0, 3, 1, 2))
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get Output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            if output.shape[0] < output.shape[1]: output = output.transpose() # Ensure [8400, 84]
            
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > self.detection_threshold
            
            if np.any(mask):
                v_boxes, v_scores = output[mask], scores[mask]
                v_classes = np.argmax(v_boxes[:, 4:], axis=1)
                
                # Simple NMS/Filter for top 5
                indices = np.argsort(v_scores)[::-1][:5]
                best_boxes, best_classes = v_boxes[indices], v_classes[indices]

                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(best_boxes, best_classes, self.class_names, self.preview), 0)
                
                # Voice Logic
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    label = self.class_names[int(best_classes[0])]
                    if self.tts: self.tts.speak(f"I see {label}", 0, None)
                    self.last_speech_time = now
            else:
                Clock.schedule_once(lambda dt: self.overlay.canvas.clear(), 0)
        except Exception as e:
            Logger.error(f"AI_PROCESS_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
