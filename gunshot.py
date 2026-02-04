import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy import signal
from scipy.stats import zscore
import tempfile
import os

class AudioEventDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_data = None
        self.sr = None
        self.load_audio()
    
    def load_audio(self):
        """Extract audio from MP4 file using pydub"""
        try:
            # Load video and extract audio
            audio = AudioSegment.from_file(self.video_path)
            
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_path = temp_file.name
            
            # Export audio as wav
            audio.export(temp_audio_path, format="wav")
            
            # Load with librosa for analysis
            self.audio_data, self.sr = librosa.load(temp_audio_path)
            
            # Cleanup temp file
            os.unlink(temp_audio_path)
            
            print(f"Audio loaded successfully: {len(self.audio_data)/self.sr:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading audio: {e}")
            print("Make sure you have ffmpeg installed on your system")
            return None
    
    def remove_silence(self, top_db=30):
        """Remove silence from audio using your original approach"""
        intervals = librosa.effects.split(self.audio_data, top_db=top_db)
        non_silent_audio = np.concatenate([self.audio_data[start:end] for start, end in intervals])
        return non_silent_audio, intervals
    
    def detect_gunshots(self, threshold_factor=3.0, min_duration=0.01, max_duration=0.5):
        """
        Detect gunshots based on:
        - Sharp onset detection
        - High energy bursts
        - Short duration
        - Broadband frequency content
        """
        # Calculate onset strength
        onset_envelope = librosa.onset.onset_strength(y=self.audio_data, sr=self.sr)
        
        # Detect sharp onsets
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=self.sr,
            units='time',
            delta=0.1,  # Minimum time between onsets
            wait=10     # Minimum frames between onsets
        )
        
        # Calculate RMS energy
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=self.audio_data, 
                                 frame_length=frame_length, 
                                 hop_length=hop_length)[0]
        
        # Convert to time
        times = librosa.frames_to_time(np.arange(len(rms)), 
                                      sr=self.sr, 
                                      hop_length=hop_length)
        
        # Find high energy events
        rms_threshold = np.mean(rms) + threshold_factor * np.std(rms)
        high_energy_frames = rms > rms_threshold
        
        gunshot_candidates = []
        
        for onset_time in onsets:
            # Find the closest RMS frame
            frame_idx = np.argmin(np.abs(times - onset_time))
            
            # Check if there's high energy around this onset
            window = 5  # frames to check around onset
            start_idx = max(0, frame_idx - window)
            end_idx = min(len(high_energy_frames), frame_idx + window)
            
            if np.any(high_energy_frames[start_idx:end_idx]):
                # Calculate spectral characteristics
                start_sample = max(0, int((onset_time - 0.1) * self.sr))
                end_sample = min(len(self.audio_data), int((onset_time + 0.5) * self.sr))
                
                segment = self.audio_data[start_sample:end_sample]
                
                if len(segment) > 0:
                    # Calculate spectral centroid (higher for gunshots)
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=self.sr))
                    
                    # Calculate zero crossing rate (higher for gunshots)
                    zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
                    
                    # Store candidate with features
                    gunshot_candidates.append({
                        'time': onset_time,
                        'rms_peak': np.max(rms[start_idx:end_idx]),
                        'spectral_centroid': spectral_centroid,
                        'zcr': zcr,
                        'confidence': rms[frame_idx] / np.mean(rms)
                    })
        
        return gunshot_candidates
    
    def detect_screams(self, min_pitch=300, max_pitch=2000, min_duration=0.5):
        """
        Detect screams based on:
        - High pitch (your original approach)
        - Sustained duration
        - High energy
        """
        # Remove silence first
        non_silent_audio, intervals = self.remove_silence()
        
        scream_candidates = []
        
        # Analyze each non-silent segment
        for start_frame, end_frame in intervals:
            segment = self.audio_data[start_frame:end_frame]
            segment_duration = len(segment) / self.sr
            
            # Skip very short segments
            if segment_duration < min_duration:
                continue
            
            # Extract pitch using your original method
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7')   # ~2093 Hz
            )
            
            # Filter out unvoiced sections and zeros
            pitch = f0[voiced_flag]
            pitch = pitch[pitch > 0]
            
            if len(pitch) > 0:
                avg_pitch = np.mean(pitch)
                pitch_std = np.std(pitch)
                
                # Check if pitch is in scream range
                if min_pitch <= avg_pitch <= max_pitch:
                    # Calculate additional features
                    segment_rms = np.sqrt(np.mean(segment**2))
                    
                    # Calculate spectral features
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=self.sr))
                    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=self.sr))
                    
                    # Time in original audio
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
                        'confidence': avg_pitch / max_pitch  # Higher pitch = higher confidence
                    })
        
        return scream_candidates
    
    def analyze_audio(self):
        """Complete analysis for both gunshots and screams"""
        if self.audio_data is None:
            print("No audio data loaded")
            return None, None
        
        print("Detecting gunshots...")
        gunshots = self.detect_gunshots()
        
        print("Detecting screams...")
        screams = self.detect_screams()
        
        return gunshots, screams
    
    def print_results(self, gunshots, screams):
        """Print detection results - timestamps only"""
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

def analyze_video(video_path):
    """Main function to analyze a video file"""
    detector = AudioEventDetector(video_path)
    gunshots, screams = detector.analyze_audio()
    
    if gunshots is not None and screams is not None:
        detector.print_results(gunshots, screams)
        return gunshots, screams
    else:
        print("Analysis failed")
        return None, None

if __name__ == "__main__":
    video_file = "gun.mp4"  # Replace with your MP4 file path
    
    # Analyze the video
    gunshots, screams = analyze_video(video_file)
    
    # You can also create the detector object directly for more control
    # detector = AudioEventDetector(video_file)
    # gunshots, screams = detector.analyze_audio()
    # detector.print_results(gunshots, screams)

# Installation requirements:
# pip install librosa numpy soundfile pydub scipy
# 
# Note: pydub requires ffmpeg to be installed on your system
# Windows: Download from https://ffmpeg.org/download.html
# Or install via chocolatey: choco install ffmpeg
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
