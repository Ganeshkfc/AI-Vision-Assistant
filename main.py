import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime

# --- KIVY UI COMPONENTS ---
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from jnius import autoclass

# REPLACE ULTRALYTICS WITH TFLITE-RUNTIME
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

class VisionApp(App):
    def build(self):
        # Your EXACT Settings from main.py
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        self.METRIC_THRESHOLD_CM = 91.44
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 6 

        # Standard COCO Class Names (Matching YOLOv8n)
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # Android TTS Setup
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
        except: self.tts = None

        # INITIALIZE TFLITE INTERPRETER
        self.interpreter = tflite.Interpreter(model_path="yolov8n_float32.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # --- GESTURE BASED UI (KEEPING YOUR EXACT DESIGN) ---
        layout = BoxLayout(orientation='vertical')
        self.top_btn = Button(text="TAP HERE TO CHANGE MODE\n(Mode 1: Multi-Object Active)", background_color=(0.1, 0.5, 0.8, 1), font_size='20sp', halign='center')
        self.top_btn.bind(on_release=self.toggle_mode)
        self.bottom_btn = Button(text="TAP HERE TO CLOSE APP", background_color=(0.8, 0.2, 0.2, 1), font_size='20sp', halign='center')
        self.bottom_btn.bind(on_release=self.check_close_app)
        layout.add_widget(self.top_btn)
        layout.add_widget(self.bottom_btn)

        Clock.schedule_once(lambda dt: self.speak("Ai vision activated. Mode 1 active. Detecting multiple objects. To change mode. Tap on your phone's top screen. To close the application. Tap the bottom screen."), 2)
        threading.Thread(target=self.ai_engine, daemon=True).start()
        return layout

    def toggle_mode(self, instance):
        if self.current_mode == 1:
            self.current_mode = 2
            self.speak("Mode 2 activated. Distance detection enabled.")
            self.top_btn.text = "MODE 2 ACTIVE\n(Single Object + Distance)"
        else:
            self.current_mode = 1
            self.speak("Mode 1 activated. Multi-object detection enabled.")
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
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: continue
            f_h, f_w, _ = frame.shape

            # PREPROCESS FOR TFLITE (640x640)
            img = cv2.resize(frame, (640, 640))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # RUN TFLITE INFERENCE
            self.interpreter.set_tensor(self.input_details[0]['index'], img)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Shape [84, 8400]
            
            # PARSE YOLOv8 TFLITE DATA
            processed_boxes = []
            for i in range(8400):
                scores = output[4:, i]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    xc, yc, w, h = output[:4, i]
                    x1 = (xc - w/2) * (f_w / 640)
                    x2 = (xc + w/2) * (f_w / 640)
                    processed_boxes.append({'x1': x1, 'x2': x2, 'label': self.class_names[class_id]})

            if processed_boxes:
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    if self.current_mode == 1:
                        detected_items = []
                        seen_labels = set()
                        for box in processed_boxes:
                            label = box['label']
                            if label not in seen_labels:
                                x_center = (box['x1'] + box['x2']) / 2
                                dir_s = "on your left" if x_center < (f_w/3) else ("in front of you" if x_center < (2*f_w/3) else "on your right")
                                detected_items.append(f"a {label} {dir_s}")
                                seen_labels.add(label)
                        if detected_items:
                            self.speak("I see " + " and ".join(detected_items))
                            self.last_speech_time = now
                    else:
                        # MODE 2: Same logic using the new 'boxes' list
                        best_box = min(processed_boxes, key=lambda b: abs(((b['x1']+b['x2'])/2) - (f_w/2)))
                        label = best_box['label']
                        x1, x2 = best_box['x1'], best_box['x2']
                        d_cm = self.get_distance_cm(label, x2 - x1)
                        d_str = f"{int(d_cm)} centimeters" if d_cm < self.METRIC_THRESHOLD_CM else f"{round(d_cm/30.48, 1)} feet"
                        self.speak(f"I see a {label}, {d_str}, in front of you")
                        self.last_speech_time = now
            time.sleep(0.1)

if __name__ == "__main__":
    VisionApp().run()
