from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import ffmpeg
import yt_dlp
from typing import Union
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)

def extract_local_video_metadata(file_path: str):
    print(file_path)
    try:
        probe = ffmpeg.probe(file_path, v='error', select_streams='v:0', show_entries='stream=duration,width,height,codec_name,avg_frame_rate,bit_rate')
        video_stream = probe['streams'][0]

        metadata = {
            "size": os.path.getsize(file_path),
            "duration": float(video_stream['duration']),
            "resolution": f"{video_stream['width']}x{video_stream['height']}",
            "codec": video_stream['codec_name'],
            "frame_rate": eval(video_stream['avg_frame_rate']),
            "bitrate": int(video_stream.get('bit_rate', 0)), 
            "title": os.path.basename(file_path),
        }

        return metadata
    except Exception as e:
        raise Exception(f"Error processing local video metadata: {str(e)}")


def extract_youtube_video_metadata(url: str):
    try:
        ydl_opts = {
            'quiet': True,
            'extract_audio': False,
            'force_generic_extractor': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            metadata = {
                "title": info_dict.get('title', 'Unknown'),
                "duration": info_dict.get('duration', 0),
                "resolution": f"{info_dict['width']}x{info_dict['height']}",
                "codec": info_dict.get('ext', 'Unknown'),
                "bitrate": info_dict.get('bitrate', 0),
                "frame_rate": info_dict.get('fps', 'Unknown'),
                "size": info_dict.get('filesize', 0),
                "creator": info_dict.get('uploader', None)
            }

        return metadata
    except Exception as e:
        raise Exception(f"Error processing YouTube video metadata: {str(e)}")


@app.post("/api/upload-video")
async def upload_video(
    videoLink: Union[str, None] = Form(None),
    videoFile: Union[UploadFile, None] = File(None)
):
    if not videoLink and not videoFile:
        return JSONResponse(status_code=400, content={"error": "No video link or file provided."})

    metadata = {}

    if videoLink:
        try:
            metadata = extract_youtube_video_metadata(videoLink)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    elif videoFile:
        file_location = os.path.join(tempfile.gettempdir(), f"temp_{videoFile.filename}")
        try:
            with open(file_location, "wb") as file:
                content = await videoFile.read()
                file.write(content)

            metadata = extract_local_video_metadata(file_location)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
        finally:
            if os.path.exists(file_location):
                os.remove(file_location)

    return metadata
