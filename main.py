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

class VisionApp(App):
    def build(self):
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 4 
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        
        self.top_btn = Button(
            text="TAP HERE TO CHANGE MODE\n(Mode 1: Detection Active)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='20sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        # Middle section containing Slider (left) and Camera (right)
        self.middle_layout = BoxLayout(orientation='horizontal')
        
        # Left side: Slider for adjusting confidence threshold
        self.slider_layout = BoxLayout(orientation='vertical', size_hint_x=0.15)
        self.slider_label = Label(text='35%', size_hint_y=0.1, font_size='18sp')
        self.threshold_slider = Slider(orientation='vertical', min=1, max=100, value=35, size_hint_y=0.9)
        self.threshold_slider.bind(value=self.on_slider_value_change)
        
        self.slider_layout.add_widget(self.slider_label)
        self.slider_layout.add_widget(self.threshold_slider)

        # Right side: Camera preview
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        
        self.middle_layout.add_widget(self.slider_layout)
        self.middle_layout.add_widget(self.preview)

        self.bottom_btn = Button(
            text="TAP HERE TO CLOSE APP",
            background_color=(0.8, 0.2, 0.2, 1), font_size='20sp', size_hint_y=0.15, halign='center'
        )
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.middle_layout)
        layout.add_widget(self.bottom_btn)
        return layout

    def on_slider_value_change(self, instance, value):
        self.slider_label.text = f"{int(value)}%"

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

    def get_distance_cm(self, label, width_px, frame_w):
        real_w = self.KNOWN_WIDTHS.get(label, 30)
        return (real_w * self.FOCAL_LENGTH) / max(width_px, 1)

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
            
            # Use dynamic threshold from the UI slider
            current_threshold = self.threshold_slider.value / 100.0
            mask = scores > current_threshold 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                sort_idx = np.argsort(valid_scores)[::-1]
                top_k = min(5, len(sort_idx))
                best_boxes = valid_boxes[sort_idx][:top_k]
                best_classes = valid_class_ids[sort_idx][:top_k]
                
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    if self.current_mode == 1:
                        # Mode 1: Announce multiple unique objects
                        detected_names = []
                        for class_id in best_classes:
                            name = self.class_names[int(class_id)]
                            if name not in detected_names:
                                detected_names.append(name)
                        
                        if len(detected_names) == 1:
                            speech_text = f"I see a {detected_names[0]}"
                        elif len(detected_names) == 2:
                            speech_text = f"I see a {detected_names[0]} and a {detected_names[1]}"
                        else:
                            speech_text = "I see a " + ", a ".join(detected_names[:-1]) + f", and a {detected_names[-1]}"
                        
                        self.speak(speech_text)
                    else:
                        # Mode 2: Distance estimation for top object
                        top_label = self.class_names[int(best_classes[0])]
                        xc, yc, w, h = best_boxes[0][:4]
                        width_px = w * (width / 640)
                        dist = self.get_distance_cm(top_label, width_px, width)
                        self.speak(f"{top_label} at {int(dist)} centimeters")
                    self.last_speech_time = now

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
