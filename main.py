import os
import time
import threading
import numpy as np
from datetime import datetime

# --- KIVY COMPONENTS ---
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera  
from kivy.clock import Clock
from jnius import autoclass

# --- TFLITE RUNTIME ---
import tflite_runtime.interpreter as tflite

class VisionApp(App):
    def build(self):
        # --- APP SETTINGS (UNTOUCHED) ---
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        self.METRIC_THRESHOLD_CM = 91.44
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 6 

        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # --- SETUP TTS ---
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
        except: self.tts = None

        # --- LOAD TFLITE MODEL ---
        # Ensure the model file is in the same folder as main.py
        self.interpreter = tflite.Interpreter(model_path="yolov8n_float32.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # --- UI LAYOUT ---
        layout = BoxLayout(orientation='vertical')

        # Native Camera widget
        self.camera = Camera(play=True, resolution=(640, 480))
        
        self.top_btn = Button(
            text="TAP HERE TO CHANGE MODE\n(Mode 1: Multi-Object Active)",
            background_color=(0.1, 0.5, 0.8, 1),
            font_size='20sp',
            halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.bottom_btn = Button(
            text="TAP HERE TO CLOSE APP",
            background_color=(0.8, 0.2, 0.2, 1),
            font_size='20sp',
            halign='center'
        )
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.bottom_btn)

        # Startup Announcement
        Clock.schedule_once(lambda dt: self.speak("AI vision Activated. Mode 1 active. Detecting multiple objects. To change mode .Tap on your phone's Top screen . To close the application. tap on bottom screen"), 2)
        
        # Start AI thread
        threading.Thread(target=self.ai_engine, daemon=True).start()
        return layout

    def toggle_mode(self, instance):
        if self.current_mode == 1:
            self.current_mode = 2
            self.speak("Mode 2 activated.Distance detection enabled.")
            self.top_btn.text = "MODE 2 ACTIVE\n(Single Object + Distance)"
        else:
            self.current_mode = 1
            self.speak("Mode 1 activated.Multiple object detection enabled")
            self.top_btn.text = "MODE 1 ACTIVE\n(Multiple Objects)"

    def check_close_app(self, instance):
        self.speak("Closing application. Thank you.")
        Clock.schedule_once(lambda dt: self.stop(), 1)

    def speak(self, text):
        if self.tts:
            self.tts.setSpeechRate(0.8) 
            self.tts.speak(text, 0, None) 

    def get_distance_cm(self, label, width_px):
        real_w = self.KNOWN_WIDTHS.get(label, 30)
        return (real_w * self.FOCAL_LENGTH) / width_px

    def ai_engine(self):
        while True:
            # SAFETY CHECK: Only process if camera is ready and has a texture
            if self.camera.play and self.camera.texture:
                try:
                    # Get raw pixel data
                    pixels = self.camera.texture.pixels
                    f_w, f_h = self.camera.texture.size
                    
                    # Convert to numpy (RGBA to RGB)
                    frame = np.frombuffer(pixels, dtype=np.uint8)
                    frame = frame.reshape((f_h, f_w, 4))[:, :, :3]
                    
                    # Run AI Preprocessing and Inference
                    input_data = self.preprocess(frame)
                    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                    self.interpreter.invoke()
                    output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                    
                    # Process results
                    self.process_results(output, f_w, f_h)
                except Exception as e:
                    print(f"AI Engine Error: {e}")
            
            time.sleep(0.1) # Prevents CPU overheating

    def preprocess(self, frame):
        # Faster numpy-only resizing for mobile
        h, w = frame.shape[:2]
        img = np.array(frame, dtype=np.float32)
        img = img[::max(1, h//640), ::max(1, w//640)][:640, :640]
        # Padding to exactly 640x640
        img = np.pad(img, ((0, max(0, 640-img.shape[0])), (0, max(0, 640-img.shape[1])), (0,0)), mode='constant')
        img = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
        return img

    def process_results(self, output, f_w, f_h):
        processed_boxes = []
        # YOLOv8 output processing
        for i in range(8400):
            scores = output[4:, i]
            class_id = np.argmax(scores)
            if scores[class_id] > 0.4:
                xc, yc, w, h = output[:4, i]
                x1 = (xc - w/2) * (f_w / 640)
                x2 = (xc + w/2) * (f_w / 640)
                processed_boxes.append({'x1': x1, 'x2': x2, 'label': self.class_names[class_id]})

        if processed_boxes:
            now = time.time()
            if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                if self.current_mode == 1:
                    items = [f"a {b['label']}" for b in processed_boxes[:3]]
                    self.speak("I see " + " and ".join(items))
                else:
                    box = processed_boxes[0]
                    dist = self.get_distance_cm(box['label'], box['x2'] - box['x1'])
                    self.speak(f"{box['label']} at {int(dist)} centimeters")
                self.last_speech_time = now

if __name__ == "__main__":
    VisionApp().run()
