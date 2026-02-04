import cv2
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import numpy as np
from typing import Dict, List

class ViolenceDetector:    
    def __init__(self, confidence_threshold: float = 0.55):
        print("Initializing Violence Detector...")
        self.model = ViTForImageClassification.from_pretrained('jaranohaal/vit-base-violence-detection')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('jaranohaal/vit-base-violence-detection')
        self.confidence_threshold = confidence_threshold
        
        # Print model configuration
        print("Model configuration:")
        print(f"Labels: {self.model.config.id2label}")
        print(f"Confidence threshold: {confidence_threshold}")
        print("Violence Detector initialized\n")

    def detect_frame(self, image: Image.Image) -> bool:
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            
            # Get probabilities for both classes
            violence_prob = probs[1].item()  # LABEL_1 is violence
            non_violence_prob = probs[0].item()  # LABEL_0 is non-violence
            
            predicted_class_idx = logits.argmax(-1).item()
            label = self.model.config.id2label[predicted_class_idx]
            
            # Consider it violence if LABEL_1 probability exceeds threshold
            is_violence = violence_prob >= self.confidence_threshold
            print(f"Frame analysis - Label: {label}, Violence prob: {violence_prob:.2%}, Non-violence prob: {non_violence_prob:.2%}")
            if is_violence:
                print(f"Violence detected with confidence: {violence_prob:.2%}")
            return is_violence

class BloodDetector:
    def __init__(self, blood_threshold: float = 0.02):
        self.blood_threshold = blood_threshold
        print(f"Blood Detector initialized with threshold: {blood_threshold}")
        
    def detect_frame(self, frame: np.ndarray) -> bool:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = (0, 50, 50)
        upper_red1 = (10, 255, 255)
        lower_red2 = (160, 50, 50)
        upper_red2 = (180, 255, 255)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        red_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_ratio = red_pixels / total_pixels
        
        is_blood = red_ratio > self.blood_threshold
        if is_blood:
            print(f"Blood detected with ratio: {red_ratio:.2%}")
        return is_blood

class VideoDetector:
    def __init__(self, violence_confidence: float = 0.6, blood_threshold: float = 0.02):
        print("\nInitializing Video Detection System...")
        self.violence_detector = ViolenceDetector(confidence_threshold=violence_confidence)
        self.blood_detector = BloodDetector(blood_threshold=blood_threshold)
        print("Video Detection System Ready\n")

    def process_video(self, video_path: str, sample_rate: float = 0.5) -> Dict[str, List[float]]:
        """
        Process video for violence and blood detection
        Args:
            video_path: Path to video file
            sample_rate: Process every N seconds (default: 0.5 seconds for more frequent checks)
        Returns:
            Dictionary with timestamps for violence and blood detection
        """
        print(f"\nProcessing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * sample_rate))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video stats - FPS: {fps}, Total frames: {total_frames}")
        print(f"Processing every {frame_interval} frames ({sample_rate} seconds)\n")
        
        violence_timestamps = []
        blood_timestamps = []
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                timestamp = frame_number / fps
                print(f"\nAnalyzing frame {frame_number}/{total_frames} at {timestamp:.2f} seconds:")
                
                # Check for violence
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.violence_detector.detect_frame(pil_image):
                    violence_timestamps.append(timestamp)
                    print(f"Violence timestamp added: {timestamp:.2f}s")

                # Check for blood
                if self.blood_detector.detect_frame(frame):
                    blood_timestamps.append(timestamp)
                    print(f"Blood timestamp added: {timestamp:.2f}s")

            frame_number += 1

        cap.release()
        print("\nVideo processing completed")

        violence_timestamps = self._filter_timestamps(violence_timestamps)
        blood_timestamps = self._filter_timestamps(blood_timestamps)

        print(f"\nFinal Results:")
        print(f"Violence detected at: {[f'{t:.2f}s' for t in violence_timestamps]}")
        print(f"Blood detected at: {[f'{t:.2f}s' for t in blood_timestamps]}\n")

        return {
            "violence_timestamps": violence_timestamps,
            "blood_timestamps": blood_timestamps
        }

    def _filter_timestamps(self, timestamps: List[float], threshold: float = 2.0) -> List[float]:
        if not timestamps:
            return []
        
        filtered = [timestamps[0]]
        for t in timestamps[1:]:
            if t - filtered[-1] > threshold:
                filtered.append(t)
        return filtered
