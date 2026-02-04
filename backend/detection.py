import cv2
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import numpy as np
import tempfile
from typing import Dict, List, Tuple
import torch.nn as nn

class ViolenceDetector:
    def __init__(self, confidence_threshold: float = 0.6):
        print("Initializing Violence Detector...")
        self.model = ViTForImageClassification.from_pretrained('jaranohaal/vit-base-violence-detection')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('jaranohaal/vit-base-violence-detection')
        self.confidence_threshold = confidence_threshold
        print(f"Violence Detector initialized with confidence threshold: {confidence_threshold}")

    def detect_frame(self, image: Image.Image) -> bool:
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            predicted_class_idx = logits.argmax(-1).item()
            label = self.model.config.id2label[predicted_class_idx]
            confidence = probs[predicted_class_idx].item()
            
            is_violence = (label.lower() == "violence" and confidence >= self.confidence_threshold)
            print(f"Frame analysis - Label: {label}, Confidence: {confidence:.2%}")
            if is_violence:
                print(f"Violence detected with confidence: {confidence:.2%}")
            return is_violence

class BloodDetector:
    def __init__(self, blood_threshold: float = 1.00):  # Increased threshold for less sensitivity
        self.blood_threshold = blood_threshold
        
    def detect_frame(self, frame: np.ndarray) -> bool:
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red color range for blood detection
        lower_red1 = (0, 50, 50)
        upper_red1 = (10, 255, 255)
        lower_red2 = (160, 50, 50)
        upper_red2 = (180, 255, 255)

        # Threshold the image
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Count red pixels
        red_pixels = cv2.countNonZero(mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        red_ratio = red_pixels / total_pixels

        return red_ratio > self.blood_threshold

class VideoDetector:
    def __init__(self, violence_confidence: float = 0.6, blood_threshold: float = 0.05):
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
        frame_interval = max(1, int(fps * sample_rate))  # At least process 1 frame
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

        # Filter out near-duplicate detections (within 2 seconds)
        violence_timestamps = self._filter_timestamps(violence_timestamps)
        blood_timestamps = self._filter_timestamps(blood_timestamps)

        return {
            "violence_timestamps": violence_timestamps,
            "blood_timestamps": blood_timestamps
        }

    def _filter_timestamps(self, timestamps: List[float], threshold: float = 2.0) -> List[float]:
        """Filter out timestamps that are too close to each other"""
        if not timestamps:
            return []
            
        filtered = [timestamps[0]]
        for t in timestamps[1:]:
            if t - filtered[-1] > threshold:
                filtered.append(t)
        return filtered

class ViolenceRNNModel(nn.Module):
    """
     RNN model for sequence data (e.g., audio event detection for violence).
    """
    def __init__(self, input_size=128, hidden_size=128, num_layers=3, num_classes=2, dropout=0.3):
        super(ViolenceRNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Last time step
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class ViolenceCNNModel(nn.Module):
    """
     CNN model for image/frame data (e.g., violence/blood detection).
    """
    def __init__(self, num_classes=2):
        super(ViolenceCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Assuming input 224x224
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, 3, 224, 224)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
