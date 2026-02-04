import cv2
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import numpy as np
from typing import Dict, List
import os
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.fft import fft, fftfreq
import librosa
try:
    from pydub import AudioSegment
except ImportError:
    print("pydub not found. Please install with: pip install pydub")
    AudioSegment = None
import soundfile as sf
from scipy.stats import zscore
import tempfile

class ViolenceDetector:
    def __init__(self, confidence_threshold: float = 0.55):  # Restored to original threshold
        print("Initializing Violence Detector...")
        self.model = ViTForImageClassification.from_pretrained('jaranohaal/vit-base-violence-detection')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('jaranohaal/vit-base-violence-detection')
        self.confidence_threshold = confidence_threshold
        
        # Print model configuration
        print("\nModel configuration:")
        print(f"Labels: {self.model.config.id2label}")
        print(f"Confidence threshold: {confidence_threshold}")
        print("Violence Detector initialized\n")
    
    def detect_frame(self, image: Image.Image) -> bool:
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits[0], dim=0)
            
            # LABEL_1 is violence, LABEL_0 is non-violence
            violence_prob = probs[1].item()  # Probability of violence
            non_violence_prob = probs[0].item()  # Probability of non-violence
            
            predicted_class_idx = logits.argmax(-1).item()
            label = self.model.config.id2label[predicted_class_idx]
            
            # Consider it violence if LABEL_1 probability exceeds threshold
            is_violence = violence_prob >= self.confidence_threshold
            print(f"Frame analysis - Label: {label}, Violence prob: {violence_prob:.2%}, Non-violence prob: {non_violence_prob:.2%}")
            if is_violence:
                print(f"Violence detected with confidence: {violence_prob:.2%}")
            return is_violence


class BloodDetector:
    def __init__(self, blood_threshold: float = 0.02):  # Restored to original threshold
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


class AudioDetector:
    def __init__(self, 
                 threshold_db: float = 30,  # Increased from 20
                 min_duration: float = 0.01,
                 max_duration: float = 0.5,
                 scream_frequency_threshold: float = 3000,  # Increased from 2000
                 scream_amplitude_threshold: float = 0.15):  # Increased from 0.08
        """
        Initialize the audio detector with both gunshot and scream detection.
        
        Args:
            threshold_db: Amplitude threshold above ambient noise (in dB)
            min_duration: Minimum duration of a gunshot (seconds)
            max_duration: Maximum duration of a gunshot (seconds)
            scream_frequency_threshold: Frequency threshold for scream detection (Hz)
            scream_amplitude_threshold: Amplitude threshold for scream detection
        """
        self.threshold_db = threshold_db
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.scream_frequency_threshold = scream_frequency_threshold
        self.scream_amplitude_threshold = scream_amplitude_threshold

    def extract_audio(self, video_path: str) -> tuple:
        """Extract and load audio from video file"""
        try:
            # Extract audio using PyDub
            audio = AudioSegment.from_file(video_path)
            temp_path = os.path.join(os.path.dirname(video_path), "temp_audio.wav")
            audio.export(temp_path, format="wav")
            
            # Load audio data
            sample_rate, audio_data = wavfile.read(temp_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize to [-1, 1]
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Clean up temp file
            os.remove(temp_path)
            
            return sample_rate, audio_data
            
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return None, None

    def detect_gunshots(self, audio_data: np.ndarray, sample_rate: int) -> List[float]:
        """Detect gunshots in audio data"""
        if audio_data is None:
            return []

        # Calculate envelope using Hilbert transform
        analytic_signal = signal.hilbert(audio_data)
        envelope = np.abs(analytic_signal)
        
        # Smooth the envelope
        window_size = int(0.01 * sample_rate)  # 10ms window
        envelope_smooth = signal.medfilt(envelope, kernel_size=window_size)
        
        # Calculate dynamic threshold
        window_length = int(0.5 * sample_rate)
        local_mean = signal.convolve(envelope_smooth, 
                                   np.ones(window_length)/window_length, 
                                   mode='same')
        
        threshold_linear = 10 ** (self.threshold_db / 20)
        threshold = local_mean * threshold_linear
        
        # Find peaks above threshold
        peaks, _ = signal.find_peaks(envelope_smooth, 
                                   height=threshold,
                                   distance=int(0.1 * sample_rate))
        
        # Analyze each peak
        gunshots = []
        for peak_idx in peaks:
            # Find start and end of transient
            start_idx = peak_idx
            while start_idx > 0 and envelope_smooth[start_idx] > threshold[peak_idx] * 0.1:
                start_idx -= 1
            
            end_idx = peak_idx
            while end_idx < len(envelope_smooth) - 1 and envelope_smooth[end_idx] > threshold[peak_idx] * 0.1:
                end_idx += 1
            
            duration = (end_idx - start_idx) / sample_rate
            
            if self.min_duration <= duration <= self.max_duration:
                # Extract segment for frequency analysis
                segment = audio_data[start_idx:end_idx]
                
                if len(segment) >= 10:
                    # Perform FFT
                    n = len(segment)
                    yf = fft(segment)
                    xf = fftfreq(n, 1/sample_rate)[:n//2]
                    power_spectrum = 2.0/n * np.abs(yf[0:n//2])
                    
                    # Analyze frequency distribution
                    freq_mask = (xf >= 500) & (xf <= 5000)
                    if np.any(freq_mask):
                        gunshot_band_power = np.sum(power_spectrum[freq_mask])
                        total_power = np.sum(power_spectrum)
                        
                        if gunshot_band_power / total_power > 0.3:
                            peak_freqs = xf[signal.find_peaks(power_spectrum, 
                                                            height=np.max(power_spectrum)*0.5)[0]]
                            if len(peak_freqs) >= 2:
                                gunshots.append(start_idx / sample_rate)
        
        return gunshots

    def detect_screams(self, audio_data: np.ndarray, sample_rate: int) -> List[float]:
        """Detect high-pitched, high-amplitude sounds that might be screams"""
        if audio_data is None:
            return []

        hop_length = 256
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, 
                                                              sr=sample_rate, 
                                                              hop_length=hop_length)[0]
        rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, 
                                                           sr=sample_rate, 
                                                           hop_length=hop_length)[0]
        
        # Find potential scream regions
        scream_frames = np.where(
            (spectral_centroids > self.scream_frequency_threshold) & 
            (rms > self.scream_amplitude_threshold) &
            (spectral_rolloff > 0.8 * sample_rate/2)
        )[0]
        
        # Convert frames to timestamps
        scream_timestamps = []
        min_duration = 0.1
        current_scream = None
        
        for frame in scream_frames:
            timestamp = frame * hop_length / sample_rate
            
            if current_scream is None:
                current_scream = timestamp
            elif frame * hop_length / sample_rate - current_scream >= min_duration:
                if np.mean(rms[frame-5:frame+5]) > self.scream_amplitude_threshold:
                    scream_timestamps.append(current_scream)
                current_scream = None
        
        return scream_timestamps

    def process_video(self, video_path: str) -> Dict[str, List[tuple]]:
        """Process video for audio events"""
        print(f"\nProcessing audio from video: {video_path}")
        
        try:
            # Extract and load audio
            sample_rate, audio_data = self.extract_audio(video_path)
            if audio_data is None:
                print("Failed to extract audio")
                return {'audio_events': []}

            # Detect gunshots and screams
            gunshot_timestamps = self.detect_gunshots(audio_data, sample_rate)
            scream_timestamps = self.detect_screams(audio_data, sample_rate)
            
            # Combine and label events
            audio_events = []
            for timestamp in gunshot_timestamps:
                audio_events.append((timestamp, "Gunshot"))
            for timestamp in scream_timestamps:
                audio_events.append((timestamp, "Scream"))
            
            # Sort by timestamp
            audio_events.sort(key=lambda x: x[0])
            
            return {
                'audio_events': audio_events
            }
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return {'audio_events': []}


class VideoDetector:
    def __init__(self, 
                 violence_confidence: float = 0.55, 
                 blood_threshold: float = 0.02,
                 threshold_db: float = 30,  # Unused, kept for compatibility
                 min_duration: float = 0.01,
                 max_duration: float = 0.5,
                 scream_frequency_threshold: float = 3000,  # Unused, kept for compatibility
                 scream_amplitude_threshold: float = 0.15):  # Unused, kept for compatibility
        print("\nInitializing Video Detection System...")
        self.violence_detector = ViolenceDetector(confidence_threshold=violence_confidence)
        self.blood_detector = BloodDetector(blood_threshold=blood_threshold)
        self.audio_detector = AudioEventDetector  # Use the best/most accurate audio detector
        print("Video Detection System Ready\n")

    def process_video(self, video_path: str, sample_rate: float = 0.5) -> Dict[str, list]:
        """
        Process video for violence, blood, and audio events
        Args:
            video_path: Path to video file
            sample_rate: Process every N seconds (default: 0.5 seconds)
        Returns:
            Dictionary with timestamps for all detected events
        """
        print(f"\nProcessing video: {video_path}")
        # Process visual events (violence and blood)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * sample_rate))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video stats - FPS: {fps}, Total frames: {total_frames}")
        print(f"Processing every {frame_interval} frames ({sample_rate} seconds)\n")
        violence_timestamps = []
        blood_timestamps = []
        frame_number = 0
        last_violence_time = -9999
        min_violence_gap = 1.0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % frame_interval == 0:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamp = frame_number / fps
                if self.violence_detector.detect_frame(pil_image):
                    if (timestamp - last_violence_time) >= min_violence_gap:
                        violence_timestamps.append(timestamp)
                        last_violence_time = timestamp
                if self.blood_detector.detect_frame(frame):
                    blood_timestamps.append(timestamp)
            frame_number += 1
        cap.release()
        # Use the best/most accurate audio detector from test.py
        audio_detector = self.audio_detector(video_path)
        gunshots, screams = audio_detector.analyze_audio()
        audio_events = []
        if gunshots:
            for t in gunshots:
                if isinstance(t, dict) and 'time' in t:
                    audio_events.append((t['time'], 'Gunshot'))
                elif isinstance(t, (float, int)):
                    audio_events.append((t, 'Gunshot'))
        if screams:
            for s in screams:
                if isinstance(s, dict) and 'start_time' in s:
                    audio_events.append((s['start_time'], 'Scream'))
                elif isinstance(s, (float, int)):
                    audio_events.append((s, 'Scream'))
        audio_events.sort(key=lambda x: x[0])
        return {
            'violence_timestamps': violence_timestamps,
            'blood_timestamps': blood_timestamps,
            'audio_events': audio_events
        }
# --- Begin AudioEventDetector from test.py ---

class AudioEventDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_data = None
        self.sr = None
        self.load_audio()
    def load_audio(self):
        """Extract audio from MP4 file using pydub"""
        try:
            audio = AudioSegment.from_file(self.video_path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_path = temp_file.name
            audio.export(temp_audio_path, format="wav")
            self.audio_data, self.sr = librosa.load(temp_audio_path)
            os.unlink(temp_audio_path)
            print(f"Audio loaded successfully: {len(self.audio_data)/self.sr:.2f} seconds")
        except Exception as e:
            print(f"Error loading audio: {e}")
            print("Make sure you have ffmpeg installed on your system")
            return None
    def remove_silence(self, top_db=30):
        intervals = librosa.effects.split(self.audio_data, top_db=top_db)
        non_silent_audio = np.concatenate([self.audio_data[start:end] for start, end in intervals])
        return non_silent_audio, intervals
    def detect_gunshots(self, threshold_factor=3.0, min_duration=0.01, max_duration=0.5):
        onset_envelope = librosa.onset.onset_strength(y=self.audio_data, sr=self.sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=self.sr,
            units='time',
            delta=0.1,
            wait=10
        )
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=self.audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=hop_length)
        rms_threshold = np.mean(rms) + threshold_factor * np.std(rms)
        high_energy_frames = rms > rms_threshold
        gunshot_candidates = []
        for onset_time in onsets:
            frame_idx = np.argmin(np.abs(times - onset_time))
            window = 5
            start_idx = max(0, frame_idx - window)
            end_idx = min(len(high_energy_frames), frame_idx + window)
            if np.any(high_energy_frames[start_idx:end_idx]):
                start_sample = max(0, int((onset_time - 0.1) * self.sr))
                end_sample = min(len(self.audio_data), int((onset_time + 0.5) * self.sr))
                segment = self.audio_data[start_sample:end_sample]
                if len(segment) > 0:
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=self.sr))
                    zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
                    gunshot_candidates.append({
                        'time': onset_time,
                        'rms_peak': np.max(rms[start_idx:end_idx]),
                        'spectral_centroid': spectral_centroid,
                        'zcr': zcr,
                        'confidence': rms[frame_idx] / np.mean(rms)
                    })
        return gunshot_candidates
    def detect_screams(self, min_pitch=300, max_pitch=2000, min_duration=0.5):
        non_silent_audio, intervals = self.remove_silence()
        scream_candidates = []
        for start_frame, end_frame in intervals:
            segment = self.audio_data[start_frame:end_frame]
            segment_duration = len(segment) / self.sr
            if segment_duration < min_duration:
                continue
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            pitch = f0[voiced_flag]
            pitch = pitch[pitch > 0]
            if len(pitch) > 0:
                avg_pitch = np.mean(pitch)
                pitch_std = np.std(pitch)
                if min_pitch <= avg_pitch <= max_pitch:
                    segment_rms = np.sqrt(np.mean(segment**2))
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=self.sr))
                    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=self.sr))
                    start_time = start_frame / self.sr
                    end_time = end_frame / self.sr
                    scream_candidates.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': segment_duration,
                        'avg_pitch': avg_pitch,
                        'pitch_std': pitch_std,
                        'rms': segment_rms,
                        'spectral_centroid': spectral_centroid,
                        'spectral_rolloff': spectral_rolloff,
                        'confidence': avg_pitch / max_pitch
                    })
        return scream_candidates
    def analyze_audio(self):
        if self.audio_data is None:
            print("No audio data loaded")
            return None, None
        print("Detecting gunshots...")
        gunshots = self.detect_gunshots()
        print("Detecting screams...")
        screams = self.detect_screams()
        return gunshots, screams
    def print_results(self, gunshots, screams):
        print(f"\n=== GUNSHOT TIMESTAMPS ===")
        if gunshots:
            for gunshot in gunshots:
                print(f"{gunshot['time']:.2f}s")
        else:
            print("No gunshots detected")
        print(f"\n=== SCREAM TIMESTAMPS ===")
        if screams:
            for scream in screams:
                print(f"{scream['start_time']:.2f}s")
        else:
            print("No screams detected")
# --- End AudioEventDetector from test.py ---
