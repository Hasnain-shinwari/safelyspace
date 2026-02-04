from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from detection_new_tmp import VideoDetector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector with custom thresholds if needed
video_detector = VideoDetector(
    violence_confidence=0.55,
    blood_threshold=0.02,
    threshold_db=20,
    min_duration=0.01,
    max_duration=0.5,
    scream_frequency_threshold=2000,
    scream_amplitude_threshold=0.08
)

@app.post("/detect-fights/")
async def detect_fights(file: UploadFile = File(...)):
    # Create a fixed temp directory in our project
    temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize temp paths
    temp_video_path = os.path.join(temp_dir, "input.mp4")
    
    try:
        # Save the file content
        content = await file.read()
        with open(temp_video_path, "wb") as f:
            f.write(content)
        
        print(f"Temp directory: {temp_dir}")
        print(f"Video saved to: {temp_video_path}")
        print(f"File exists: {os.path.exists(temp_video_path)}")
        print(f"File size: {os.path.getsize(temp_video_path)} bytes")
        print(f"Directory is writable: {os.access(temp_dir, os.W_OK)}")
        
        # Process video for all events
        results = video_detector.process_video(temp_video_path)
        print(f"Backend: violence_timestamps: {results['violence_timestamps']}")
        print(f"Backend: blood_timestamps: {results['blood_timestamps']}")
        print(f"Backend: audio_events: {results['audio_events']}")
        
        # Separate audio events by type and convert to float
        gunshot_timestamps = []
        scream_timestamps = []
        for timestamp, event_type in results['audio_events']:
            ts = float(timestamp)
            if event_type == "Gunshot":
                gunshot_timestamps.append(ts)
            elif event_type == "Scream":
                scream_timestamps.append(ts)
        # Combine gunshot and scream timestamps for frontend as a flat list of floats
        scream_gunshot_timestamps = [float(ts) for ts in gunshot_timestamps + scream_timestamps]
        response = {
            'violence_timestamps': [float(t) for t in results['violence_timestamps']],
            'blood_timestamps': [float(t) for t in results['blood_timestamps']],
            'scream_gunshot_timestamps': scream_gunshot_timestamps
        }
        print(f"Backend: FINAL RESPONSE: {response}")
        return response    
    finally:
        # Clean up temp files but keep the directory
        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
                print(f"Cleaned up video file: {temp_video_path}")
        except Exception as e:
            print(f"Error during cleanup: {e}")
