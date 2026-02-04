import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os

class BloodEventDetector:
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
    
    def detect_bloody_scenes(self, min_low_freq_energy=0.02, min_duration=0.5):
        """
        Detect scenes with high low-frequency energy (e.g., for blood/gore effects).
        """
        if self.audio_data is None:
            print("No audio data loaded")
            return []
        S = np.abs(librosa.stft(self.audio_data))
        freqs = librosa.fft_frequencies(sr=self.sr)
        low_freq_band = (freqs < 250)  # 0-250 Hz
        low_freq_energy = S[low_freq_band, :].mean(axis=0)
        times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr)
        threshold = np.mean(low_freq_energy) + 2 * np.std(low_freq_energy)
        bloody_segments = []
        in_segment = False
        start_time = 0
        for i, energy in enumerate(low_freq_energy):
            if energy > threshold and not in_segment:
                in_segment = True
                start_time = times[i]
            elif energy <= threshold and in_segment:
                end_time = times[i]
                if end_time - start_time >= min_duration:
                    bloody_segments.append({'start_time': start_time, 'end_time': end_time, 'duration': end_time - start_time})
                in_segment = False
        return bloody_segments
    
    def print_results(self, bloody_segments):
        print(f"\n=== BLOODY SCENES ===")
        if bloody_segments:
            for seg in bloody_segments:
                print(f"{seg['start_time']:.2f}s - {seg['end_time']:.2f}s (Duration: {seg['duration']:.2f}s)")
        else:
            print("No bloody scenes detected")

def analyze_video(video_path):
    detector = BloodEventDetector(video_path)
    bloody_segments = detector.detect_bloody_scenes()
    detector.print_results(bloody_segments)
    return bloody_segments

if __name__ == "__main__":
    video_file = "blood.mp4"  # Replace with your MP4 file path
    analyze_video(video_file)

# Requirements:
# pip install librosa numpy soundfile pydub
# ffmpeg required for pydub
