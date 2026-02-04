import librosa
import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import ffmpeg
import os
from typing import Dict, List, Tuple
import tempfile
import torchcrepe

class AudioEventDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        print("Initializing Audio Event Detector...")
        
        # Check if ffmpeg is available
        ffmpeg_status = self._check_ffmpeg()
        if not ffmpeg_status['available']:
            raise RuntimeError(
                f"FFmpeg is not available: {ffmpeg_status['error']}\n\n"
                "To install FFmpeg on Windows:\n"
                "1. Download the latest release from https://github.com/BtbN/FFmpeg-Builds/releases\n"
                "   (Download ffmpeg-master-latest-win64-gpl.zip)\n"
                "2. Extract the ZIP file to a folder (e.g., C:\\ffmpeg)\n"
                "3. Add the bin folder (e.g., C:\\ffmpeg\\bin) to your system PATH:\n"
                "   a. Open System Properties (Win + X -> System)\n"
                "   b. Click 'Environment Variables'\n"
                "   c. Under 'System variables', find and select 'Path'\n"
                "   d. Click 'Edit' and add the bin folder path\n"
                "   e. Click 'OK' on all windows\n"
                "4. Restart your terminal/IDE\n\n"
                "Alternative: You can also install FFmpeg using winget:\n"
                "   Open PowerShell and run: winget install ffmpeg"
            )
            
        # Load the model and feature extractor for audio event detection
        model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.confidence_threshold = confidence_threshold
        
        # Labels we're interested in
        self.target_labels = {'Gunshot, gunfire', 'Screaming', 'Explosion', 'Crying, sobbing'}
        
        print(f"Audio Event Detector initialized with confidence threshold: {confidence_threshold}\n")
        
    def _check_ffmpeg(self) -> Dict[str, any]:
        """
        Check if ffmpeg is available in the system
        Returns:
            Dict with 'available' boolean and 'error' string if not available
        """
        import subprocess
        import shutil
        
        result = {
            'available': False,
            'error': None
        }
        
        # First check if ffmpeg is in PATH
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is None:
            result['error'] = "FFmpeg not found in system PATH"
            return result
            
        # Try to run ffmpeg -version to verify it works
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         check=True, 
                         capture_output=True,
                         text=True)
            result['available'] = True
            return result
        except subprocess.CalledProcessError as e:
            result['error'] = f"FFmpeg found but failed to run: {str(e)}"
            return result
        except Exception as e:
            result['error'] = f"Error checking FFmpeg: {str(e)}"
            return result
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio from video file and save as WAV"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
            
        # Extract audio using ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, temp_audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        return temp_audio_path

    def detect_events(self, audio_path: str, window_size: float = 1.0, hop_size: float = 0.5) -> Dict[str, List[float]]:
        """
        Detect audio events in windows of audio
        Args:
            audio_path: Path to audio file
            window_size: Size of analysis window in seconds
            hop_size: Time between windows in seconds
        Returns:
            Dictionary with timestamps for each detected event type
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Initialize timestamp lists for each event type
        timestamps = {label: [] for label in self.target_labels}
        
        # Convert sizes to samples
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        for i in range(0, len(y) - window_samples, hop_samples):
            # Extract window
            window = y[i:i + window_samples]
            
            # Get current timestamp
            timestamp = i / sr
            
            # Process audio through the model
            inputs = self.feature_extractor(
                window, 
                sampling_rate=sr, 
                return_tensors="pt", 
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs.logits)[0]
            
            # Check each label
            for idx, label in enumerate(self.model.config.id2label.values()):
                if label in self.target_labels and probs[idx] >= self.confidence_threshold:
                    timestamps[label].append(timestamp)
                    print(f"{label} detected at {timestamp:.2f}s with confidence: {probs[idx]:.2%}")
        
        return timestamps

    def detect_pitch_events(self, audio_path: str, window_size: float = 1.0, hop_size: float = 0.1, pitch_threshold: float = 400, periodicity_threshold: float = 0.8) -> List[float]:
        """
        Detect scream-like events using torchcrepe pitch tracking.
        Returns a list of timestamps where high-pitch, high-periodicity events occur.
        """
        y, sr = librosa.load(audio_path, sr=16000)
        # torchcrepe expects (batch, time)
        audio = torch.tensor(y).unsqueeze(0)
        # Predict pitch and periodicity
        f0, periodicity = torchcrepe.predict(
            audio,
            sample_rate=sr,
            hop_length=int(hop_size * sr),
            fmin=50,
            fmax=2000,
            model='full',
            batch_size=128,
            device='cpu',
            return_periodicity=True
        )
        f0 = f0.squeeze().numpy()
        periodicity = periodicity.squeeze().numpy()
        # Find frames with high pitch and high periodicity
        high_pitch = f0 > pitch_threshold
        high_period = periodicity > periodicity_threshold
        scream_frames = (high_pitch & high_period)
        # Convert frame indices to times
        times = np.arange(len(f0)) * hop_size
        scream_times = times[scream_frames]
        # Filter out events that are too close
        filtered = []
        for t in scream_times:
            if not filtered or t - filtered[-1] > window_size:
                filtered.append(t)
        return filtered

    def detect_events_with_pitch(self, audio_path: str, window_size: float = 1.0, hop_size: float = 0.5) -> Dict[str, List[float]]:
        """
        Detect audio events using both the transformer model and torchcrepe pitch tracking.
        Returns a dictionary with timestamps for each detected event type, including 'Scream (pitch)'.
        """
        # Standard transformer-based detection
        transformer_events = self.detect_events(audio_path, window_size, hop_size)
        # Pitch-based scream detection
        scream_pitch_times = self.detect_pitch_events(audio_path)
        # Merge results
        result = dict(transformer_events)
        if 'Scream (pitch)' not in result:
            result['Scream (pitch)'] = []
        result['Scream (pitch)'].extend(scream_pitch_times)
        return result


class VideoAudioDetector:
    def __init__(self, audio_confidence: float = 0.5):
        print("\nInitializing Video Audio Detection System...")
        self.audio_detector = AudioEventDetector(confidence_threshold=audio_confidence)
        print("Video Audio Detection System Ready\n")

    def process_video(self, video_path: str) -> Dict[str, List[float]]:
        """
        Process video for audio events
        Args:
            video_path: Path to video file
        Returns:
            Dictionary with timestamps for detected audio events
        """
        print(f"\nProcessing video audio: {video_path}")
        
        # Extract audio from video
        print("Extracting audio from video...")
        audio_path = self.audio_detector.extract_audio_from_video(video_path)
        
        try:
            # Detect events (use combined transformer + pitch method)
            print("Detecting audio events...")
            timestamps = self.audio_detector.detect_events_with_pitch(audio_path)
            
            # Filter out near-duplicate detections
            filtered_timestamps = {}
            for event_type, times in timestamps.items():
                filtered_timestamps[event_type] = self._filter_timestamps(times)
            
            # Print results
            print("\nFinal Audio Results:")
            for event_type, times in filtered_timestamps.items():
                if times:
                    print(f"{event_type} detected at: {[f'{t:.2f}s' for t in times]}")
            print()
            
            return filtered_timestamps
            
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _filter_timestamps(self, timestamps: List[float], threshold: float = 2.0) -> List[float]:
        """Filter out timestamps that are too close to each other"""
        if not timestamps:
            return []
        
        filtered = [timestamps[0]]
        for t in sorted(timestamps):
            if t - filtered[-1] > threshold:
                filtered.append(t)
        return filtered
