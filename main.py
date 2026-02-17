__version__ = "1.0"
import os
import time
import numpy as np
from PIL import Image
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.utils import platform
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

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 6 
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
            except Exception as e:
                print(f"TTS Initialization Error: {e}")

        # Improved model path finding for Android
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(cur_dir, "yolov8n_float32.tflite")
        
        self.interpreter = None
        if os.path.exists(model_path) and tflite:
            try:
                self.interpreter = tflite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as e:
                print(f"Interpreter Error: {e}")

        layout = BoxLayout(orientation='vertical')
        self.preview = Preview(aspect_ratio='16:9', enable_analyze_pixels=True)
        self.preview.analyze_pixels_callback = self.analyze_frame

        self.top_btn = Button(
            text="TAP TO CHANGE MODE\n(Mode 1: Multi-Object)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='18sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.bottom_btn = Button(
            text="EXIT APPLICATION",
            background_color=(0.8, 0.2, 0.2, 1), font_size='18sp', size_hint_y=0.15, halign='center'
        )
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.preview)
        layout.add_widget(self.bottom_btn)
        return layout

    def on_start(self):
        if platform == 'android':
            from android.os import Build
            perms = [Permission.CAMERA]
            if Build.VERSION.SDK_INT >= 33:
                perms.append(Permission.READ_MEDIA_IMAGES)
            else:
                perms.append(Permission.READ_EXTERNAL_STORAGE)
                perms.append(Permission.WRITE_EXTERNAL_STORAGE)
            request_permissions(perms, self.on_permission_result)
        else:
            self.start_camera()

    def on_permission_result(self, permissions, grants):
        if all(grants):
            self.start_camera()
        else:
            self.speak("Camera permission required.")

    def start_camera(self):
        # Delayed connection to ensure UI is ready
        Clock.schedule_once(lambda dt: self.preview.connect_camera(enable_analyze_pixels=True), 1)
        Clock.schedule_once(lambda dt: self.speak("Vision System Ready."), 2)

    def toggle_mode(self, instance):
        if self.current_mode == 1:
            self.current_mode = 2
            self.speak("Mode 2: Distance sensing.")
            self.top_btn.text = "MODE 2: DISTANCE"
        else:
            self.current_mode = 1
            self.speak("Mode 1: Object detection.")
            self.top_btn.text = "MODE 1: OBJECTS"

    def check_close_app(self, instance):
        self.speak("Closing.")
        try:
            self.preview.disconnect_camera()
        except:
            pass
        Clock.schedule_once(lambda dt: self.stop(), 0.5)

    def speak(self, text):
        if self.tts:
            try:
                self.tts.speak(text, 0, None)
            except:
                pass

    def analyze_frame(self, pixels, width, height, image_pos, image_size, texture):
        if not self.interpreter: return
        try:
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 4))
            rgb = frame[:, :, :3]
            img = Image.fromarray(rgb).resize((640, 640), Image.BILINEAR)
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            # Fix for different model input shapes
            if self.input_details[0]['shape'][1] == 3:
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            self.process_results(output, width, height)
        except Exception as e:
            print(f"Analysis Error: {e}")

    def process_results(self, output, f_w, f_h):
        output = output.transpose() 
        processed_boxes = []
        for i in range(output.shape[0]):
            row = output[i]
            scores = row[4:]
            class_id = np.argmax(scores)
            if scores[class_id] > 0.5: # Higher confidence threshold
                xc, yc, w, h = row[:4]
                x1 = (xc - w/2) * (f_w / 640)
                x2 = (xc + w/2) * (f_w / 640)
                processed_boxes.append({'x1': x1, 'x2': x2, 'label': self.class_names[class_id]})

        if processed_boxes:
            now = time.time()
            if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                if self.current_mode == 1:
                    items = list(set([b['label'] for b in processed_boxes[:2]]))
                    self.speak("I see " + " and ".join(items))
                else:
                    box = processed_boxes[0]
                    # Simple distance calc
                    real_w = self.KNOWN_WIDTHS.get(box['label'], 30)
                    dist = (real_w * self.FOCAL_LENGTH) / max((box['x2'] - box['x1']), 1)
                    self.speak(f"{box['label']} at {int(dist)} centimeters")
                self.last_speech_time = now

if __name__ == "__main__":
    VisionApp().run()
