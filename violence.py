import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os

class ViolenceEventDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_data = None
        self.sr = None
        self.load_audio()
    
    def load_audio(self):
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
    
    def detect_violence(self, min_energy=0.05, min_duration=1.0):
        """
        Detect violence based on sustained high energy and chaotic audio patterns.
        """
        if self.audio_data is None:
            print("No audio data loaded")
            return []
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=self.audio_data, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=hop_length)
        threshold = np.mean(rms) + 2 * np.std(rms)
        violence_segments = []
        in_segment = False
        start_time = 0
        for i, energy in enumerate(rms):
            if energy > threshold and not in_segment:
                in_segment = True
                start_time = times[i]
            elif energy <= threshold and in_segment:
                end_time = times[i]
                if end_time - start_time >= min_duration:
                    violence_segments.append({'start_time': start_time, 'end_time': end_time, 'duration': end_time - start_time})
                in_segment = False
        return violence_segments
    
    def print_results(self, violence_segments):
        print(f"\n=== VIOLENCE SEGMENTS ===")
        if violence_segments:
            for seg in violence_segments:
                print(f"{seg['start_time']:.2f}s - {seg['end_time']:.2f}s (Duration: {seg['duration']:.2f}s)")
        else:
            print("No violence detected")

def analyze_video(video_path):
    detector = ViolenceEventDetector(video_path)
    violence_segments = detector.detect_violence()
    detector.print_results(violence_segments)
    return violence_segments

if __name__ == "__main__":
    video_file = "violence.mp4"  # Replace with your MP4 file path
    analyze_video(video_file)

# Requirements:
# pip install librosa numpy soundfile pydub
# ffmpeg required for pydub
