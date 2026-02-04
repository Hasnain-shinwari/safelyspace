# Violence Detection Backend Documentation

## Overview
This backend provides detection of violence, blood, and gunshot/scream events in video files. It uses deep learning models and audio processing libraries to extract, analyze, and classify visual and audio events related to violence.

---

## Violence Detection

- **Model Used:**
  - Name: `jaranohaal/vit-base-violence-detection`
  - Type: Vision Transformer (ViT) for Image Classification
  - Source: HuggingFace Model Hub ([link](https://huggingface.co/jaranohaal/vit-base-violence-detection))
  - Task: Frame-level violence detection in videos
  - Usage: Each video frame is processed through the ViT model. If the probability of violence (LABEL_1) exceeds the confidence threshold (default 0.55), the frame is marked as violent.
  - Output: List of timestamps (in seconds) where violence is detected.

---

## Blood Detection

- **Method:**
  - Approach: Color-based detection using HSV color space in OpenCV.
  - Details: The frame is converted to HSV, and red color regions are detected using two red hue ranges. If the ratio of red pixels to total pixels exceeds the threshold (default 0.02), blood is detected in the frame.
  - Output: List of timestamps (in seconds) where blood is detected.

---

## Gunshot/Scream Detection

- **Gunshot Detection:**
  - Method: Onset and energy analysis using librosa (from test.py)
  - Details: Detects sharp onsets in the audio and checks for high energy around these onsets. Returns timestamps (in seconds) for detected gunshots.
  - Output: List of timestamps (in seconds) where gunshots are detected.

- **Scream Detection:**
  - Method: Spectral feature analysis using librosa.
  - Details: Detects high-pitch, high-amplitude regions in the audio using spectral centroid, RMS, and rolloff features. Classifies as scream if features exceed set thresholds.
  - Output: List of timestamps (in seconds) where screams are detected.

---

## Libraries Used

| Library         | Purpose                                                      |
|-----------------|-------------------------------------------------------------|
| `torch`         | PyTorch, for deep learning model inference                  |
| `transformers`  | HuggingFace Transformers, for loading pre-trained models    |
| `opencv-python` | Video frame extraction and color processing                 |
| `Pillow`        | Image processing                                            |
| `numpy`         | Numerical operations                                        |
| `scipy`         | Signal processing and FFT                                   |
| `librosa`       | Audio feature extraction                                    |
| `pydub`         | Audio extraction from video                                 |

---

## System Requirements

- **Python Libraries:**  
  - `torch`, `transformers`, `opencv-python`, `Pillow`, `numpy`, `scipy`, `librosa`, `pydub`
- **External Dependencies:**  
  - FFmpeg (required by pydub for audio extraction)

---

## Usage Flow

1. **Initialization:**  
   - Loads the violence detection model and feature extractor.
   - Sets up blood and audio detectors with configurable thresholds.

2. **Processing a Video:**  
   - Extracts frames and processes each for violence and blood detection.
   - Extracts audio and processes for gunshot and scream detection.
   - Returns all detected events with their timestamps.

3. **Output:**  
   - Dictionary with lists of timestamps for violence, blood, and audio events (gunshot/scream).
   - Example output:
     ```python
     {
         'violence_timestamps': [12.5, 24.0, ...],
         'blood_timestamps': [13.0, 25.0, ...],
         'audio_events': [(14.2, 'Gunshot'), (15.8, 'Scream'), ...]
     }
     ```

---

## Setup Guide

### 1. Clone the Repository

Clone or download the project to your local machine:
```sh
# Using git
git clone <your-repo-url>
cd <project-folder>
```

### 2. Install Python (if not already installed)
- Download and install Python 3.8 or newer from [python.org](https://www.python.org/downloads/).
- Make sure to check "Add Python to PATH" during installation.

### 3. Create and Activate a Virtual Environment (Recommended)
```sh
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies
Install all required libraries using pip:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers opencv-python pillow numpy scipy librosa pydub
```

### 5. Install FFmpeg
FFmpeg is required for audio extraction. Install it as follows:
- **Windows:**
  1. Download from [FFmpeg official site](https://ffmpeg.org/download.html) or [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases).
  2. Extract the ZIP and add the `bin` folder to your system PATH.
  3. Or use Chocolatey: `choco install ffmpeg`
  4. Or use PowerShell (winget): `winget install ffmpeg`
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

To verify installation, run:
```sh
ffmpeg -version
```

### 6. Download Pretrained Models (Automatic)
The required models (`jaranohaal/vit-base-violence-detection`) will be downloaded automatically the first time you run the backend.

### 7. Running the Backend
To start the backend server, use the following command from the `backend` directory:
```sh
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```
This will launch the FastAPI backend at `http://localhost:8000`.

### 8. Running the Frontend
Navigate to the `frontend` directory and start the frontend server:
```sh
npm install  # Only needed the first time or when dependencies change
npm run dev
```
This will launch the frontend, usually at `http://localhost:5173` (or as shown in your terminal).

### 9. (Optional) Frontend Integration
Make sure the backend output matches the expected format for your frontend:
```python
{
    'violence_timestamps': [12.5, 24.0, ...],
    'blood_timestamps': [13.0, 25.0, ...],
    'audio_events': [(14.2, 'Gunshot'), (15.8, 'Scream'), ...]
}
```

---

### Alternative FFmpeg Installation (PowerShell)
You can also install FFmpeg on Windows using PowerShell:
```powershell
winget install ffmpeg
```
Or using Chocolatey (if installed):
```powershell
choco install ffmpeg
```

---

### Troubleshooting
- If you get errors about missing libraries, double-check your virtual environment and pip install commands.
- If audio extraction fails, ensure FFmpeg is installed and available in your system PATH.
- For GPU acceleration, install the appropriate PyTorch version for your CUDA setup (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).

---
