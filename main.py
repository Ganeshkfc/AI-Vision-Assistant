import os
import time
import numpy as np
from PIL import Image
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
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
        self.FOCAL_LENGTH = 300 
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 3 
        
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.tts = None
        self.vibrator = None
        self.interpreter = None

        layout = BoxLayout(orientation='vertical')
        
        # --- TOP BUTTON ---
        self.top_btn = Button(
            text="TAP TO CHANGE MODE\n(Mode 1: Object Detection)",
            background_color=(0.1, 0.5, 0.8, 1), font_size='18sp', size_hint_y=0.15, halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        # --- MIDDLE SECTION (Slider - Camera - Slider) ---
        self.middle_layout = BoxLayout(orientation='horizontal')
        
        # Left Slider: Confidence Threshold
        self.slider_left = BoxLayout(orientation='vertical', size_hint_x=0.15)
        self.slider_label = Label(text='35%', size_hint_y=0.1, font_size='16sp')
        self.threshold_slider = Slider(orientation='vertical', min=1, max=100, value=35, size_hint_y=0.9)
        self.threshold_slider.bind(value=self.on_slider_value_change)
        self.slider_left.add_widget(self.slider_label)
        self.slider_left.add_widget(self.threshold_slider)

        # Center: Camera Preview
        self.preview = Preview(aspect_ratio='16:9')
        self.preview.enable_analyze_pixels = True
        self.preview.analyze_pixels_callback = self.analyze_frame
        
        # Right Slider: Voice Speed
        self.slider_right = BoxLayout(orientation='vertical', size_hint_x=0.15)
        self.speed_label = Label(text='0.8x', size_hint_y=0.1, font_size='16sp')
        self.speed_slider = Slider(orientation='vertical', min=0.5, max=2.0, value=0.8, size_hint_y=0.9)
        self.speed_slider.bind(value=self.on_speed_change)
        self.slider_right.add_widget(self.speed_label)
        self.slider_right.add_widget(self.speed_slider)

        self.middle_layout.add_widget(self.slider_left)
        self.middle_layout.add_widget(self.preview)
        self.middle_layout.add_widget(self.slider_right)

        # --- BOTTOM BUTTON ---
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

    def on_speed_change(self, instance, value):
        self.speed_label.text = f"{value:.1f}x"
        if self.tts:
            try: self.tts.setSpeechRate(float(value))
            except: pass

    def on_start(self):
        if platform == 'android':
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                Context = autoclass('android.content.Context')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
                self.vibrator = PythonActivity.mActivity.getSystemService(Context.VIBRATOR_SERVICE)
                # Set initial speed after a short delay to ensure TTS is ready
                Clock.schedule_once(lambda dt: self.tts.setSpeechRate(self.speed_slider.value) if self.tts else None, 1.5)
            except Exception as e:
                Logger.error(f"Android Services Error: {e}")

        Clock.schedule_once(self.load_model, 0.5)
        if platform == 'android':
            perms = [Permission.CAMERA, Permission.VIBRATE]
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
            Clock.schedule_once(lambda x: self.speak("AI Vision Activated."), 2)
        except Exception as e:
            Logger.error(f"CAMERA: Error {e}")

    def toggle_mode(self, instance):
        self.current_mode = 2 if self.current_mode == 1 else 1
        msg = "Distance Mode" if self.current_mode == 2 else "Object Mode"
        self.speak(msg)
        self.top_btn.text = f"TAP TO CHANGE MODE\n({msg} Active)"

    def check_close_app(self, instance):
        self.speak("Closing")
        self.preview.disconnect_camera()
        Clock.schedule_once(lambda dt: self.stop(), 0.5)

    def speak(self, text):
        if self.tts:
            try: self.tts.speak(text, 0, None)
            except: pass

    def vibrate(self, duration_ms):
        if self.vibrator:
            try: self.vibrator.vibrate(duration_ms)
            except: pass

    def get_distance_cm(self, label, width_px, frame_w):
        real_w = self.KNOWN_WIDTHS.get(label, 30)
        return (real_w * self.FOCAL_LENGTH) / max(width_px, 1)

    def analyze_frame(self, pixels, *args):
        if not self.interpreter: return

        try:
            if isinstance(args[0], (list, tuple)):
                width, height = args[0]
            else:
                width, height = args[0], args[1]

            channels = len(pixels) // (width * height)
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, channels))
            rgb = frame[:, :, :3] 
            
            img = Image.fromarray(rgb).resize((640, 640))
            input_data = np.expand_dims(np.array(img), axis=0).astype(np.float32) / 255.0
            
            if self.input_details[0]['shape'][1] == 3: 
                input_data = np.transpose(input_data, (0, 3, 1, 2))
                
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0].transpose()
            
            scores = np.max(output[:, 4:], axis=1)
            current_threshold = self.threshold_slider.value / 100.0
            mask = scores > current_threshold 
            
            if np.any(mask):
                valid_boxes = output[mask]
                valid_scores = scores[mask]
                valid_class_ids = np.argmax(valid_boxes[:, 4:], axis=1)
                
                sort_idx = np.argsort(valid_scores)[::-1]
                top_k = min(3, len(sort_idx))
                best_boxes = valid_boxes[sort_idx][:top_k]
                best_classes = valid_class_ids[sort_idx][:top_k]
                
                now = time.time()
                speech_segments = []

                for i in range(len(best_classes)):
                    class_id = int(best_classes[i])
                    name = self.class_names[class_id]
                    xc, yc, w, h = best_boxes[i][:4]
                    
                    # --- FIXED SPATIAL LOGIC (yc = Horizontal for Portrait) ---
                    # The formula for relative position is:
                    # $$relative\_pos = \frac{yc}{640}$$
                    relative_pos = yc / 640.0 
                    
                    if relative_pos < 0.35:
                        direction = "on your left"
                    elif relative_pos > 0.65:
                        direction = "on your right"
                    else:
                        direction = "ahead"
                    # -----------------------------------------------------------

                    if self.current_mode == 1:
                        if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                            segment = f"{name} {direction}"
                            if segment not in speech_segments:
                                speech_segments.append(segment)
                    else:
                        if i == 0:
                            # Use 'h' (height) as width when camera is rotated
                            width_px = h * (width / 640) 
                            dist = self.get_distance_cm(name, width_px, width)
                            
                            if dist < 100:
                                self.vibrate(150 if dist > 50 else 400)

                            if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                                speech_segments.append(f"{name} {direction}, {int(dist)} centimeters")

                if speech_segments:
                    self.speak(", ".join(speech_segments))
                    self.last_speech_time = now

        except Exception as e:
            Logger.error(f"AI_ERROR: {e}")

if __name__ == "__main__":
    VisionApp().run()
