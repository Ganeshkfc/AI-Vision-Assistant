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
except:
    tflite = None

class BBoxOverlay(Widget):
    def draw_boxes(self, valid_boxes, valid_class_ids, class_names, preview_widget):
        self.canvas.clear()
        pw, ph = preview_widget.size
        px, py = preview_widget.pos
        with self.canvas:
            for i in range(len(valid_boxes)):
                box = valid_boxes[i]
                xc, yc, w, h = map(float, box[:4])
                scale_x, scale_y = pw / 640.0, ph / 640.0
                w_px, h_px = w * scale_x, h * scale_y
                x1_px = px + ((xc - w/2) * scale_x)
                y1_px = py + ph - ((yc + h/2) * scale_y) 
                Color(0, 1, 0, 1) 
                Line(rectangle=(x1_px, y1_px, w_px, h_px), width=2)

class VisionApp(App):
    def build(self):
        self.interpreter = None
        self.tts = None
        self.last_speech = 0
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        layout = BoxLayout(orientation='vertical')
        # Diagnostic Label
        self.status_lbl = Label(text="Initializing...", size_hint_y=0.1)
        
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        
        self.overlay = BBoxOverlay()
        cam_layout = FloatLayout()
        cam_layout.add_widget(self.preview)
        cam_layout.add_widget(self.overlay)

        # Force Start Button (In case permissions fail)
        self.start_btn = Button(text="START CAMERA", size_hint_y=0.15, background_color=(0,1,0,1))
        self.start_btn.bind(on_release=lambda x: self.preview.connect_camera(camera_id='back'))

        layout.add_widget(self.status_lbl)
        layout.add_widget(cam_layout)
        layout.add_widget(self.start_btn)
        return layout

    def on_start(self):
        Logger.info("DIAGNOSTIC: App Started")
        if platform == 'android':
            request_permissions([Permission.CAMERA], self.init_all)
        else:
            self.init_all(None, [True])

    def init_all(self, permissions, grants):
        if any(grants):
            self.status_lbl.text = "Camera Permission OK. Loading TTS..."
            self.init_tts()
            self.load_model()
            self.preview.connect_camera(camera_id='back')
        else:
            self.status_lbl.text = "Camera Permission DENIED!"

    def init_tts(self):
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                TextToSpeech = autoclass('android.speech.tts.TextToSpeech')
                self.tts = TextToSpeech(PythonActivity.mActivity, None)
                Clock.schedule_once(lambda dt: self.speak("Vision app ready"), 2)
            except Exception as e:
                Logger.error(f"DIAGNOSTIC: TTS Init Failed: {e}")

    def load_model(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        files = os.listdir(cur_dir)
        Logger.info(f"DIAGNOSTIC: Files in app: {files}")
        
        tflite_files = [f for f in files if f.endswith('.tflite')]
        if not tflite_files:
            self.status_lbl.text = "ERROR: No .tflite file found!"
            return

        try:
            self.interpreter = tflite.Interpreter(model_path=os.path.join(cur_dir, tflite_files[0]))
            self.interpreter.allocate_tensors()
            self.input_det = self.interpreter.get_input_details()
            self.output_det = self.interpreter.get_output_details()
            self.status_lbl.text = f"Model Loaded: {tflite_files[0]}"
            Logger.info("DIAGNOSTIC: Model Loaded Successfully")
        except Exception as e:
            self.status_lbl.text = f"Model Load Failed: {str(e)[:20]}"

    def speak(self, text):
        if self.tts:
            self.tts.speak(text, 0, None)

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return
        try:
            width, height = args[0] if isinstance(args[0], (list, tuple)) else (args[0], args[1])
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
            rgb = frame[:, :, :3]
            img = Image.fromarray(rgb).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            self.interpreter.set_tensor(self.input_det[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_det[0]['index'])[0].transpose()
            
            # Simple Confidence Filter
            scores = np.max(output[:, 4:], axis=1)
            mask = scores > 0.45
            if np.any(mask):
                v_boxes = output[mask]
                v_classes = np.argmax(v_boxes[:, 4:], axis=1)
                
                Clock.schedule_once(lambda dt: self.overlay.draw_boxes(v_boxes[:3], v_classes[:3], self.class_names, self.preview), 0)
                
                if time.time() - self.last_speech > 6:
                    label = self.class_names[int(v_classes[0])]
                    self.speak(f"I see {label}")
                    self.last_speech = time.time()
        except Exception as e:
            pass

if __name__ == "__main__":
    VisionApp().run()
