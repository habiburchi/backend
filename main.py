from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from pathlib import Path
from typing import Dict
import shutil
import uuid
import cv2
import os
import numpy as np
from ultralytics import YOLO
from supervision.geometry.core import Point
import json
from datetime import datetime
from tracking.sort import Sort
from supervision import (
    VideoInfo,
    get_video_frames_generator,
    LineZone,
    LineZoneAnnotator,
    Detections,
    BoxAnnotator,
    ColorLookup
)

app = FastAPI()

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # sesuaikan dengan FE
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directories ---
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("result")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model ---
model = YOLO("yolov8x.pt")
model.fuse()

# --- Progress Store ---
progress_store: Dict[str, float] = {}

# --- Cancel ---
cancel_flags: Dict[str, bool] = {}


# --- API ---

#History
HISTORY_FILE = "history.json"

def save_history(entry):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

@app.get("/history")
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []



@app.post("/process-video/")
async def process_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    video_id = uuid.uuid4().hex
    temp_video_path = UPLOAD_DIR / f"{video_id}_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def run_task():
        result_path, count_in, count_out = run_detection(temp_video_path, video_id)
        # Simpan hasil akhir
        progress_store[video_id] = {
            "status": "done",
            "masuk": count_in,
            "keluar": count_out,
            "video_result": f"http://localhost:8000/result/{result_path.name}"
        }
        save_history({
            "video_id": video_id,
            "filename": file.filename,  # nama asli file
            "masuk": count_in,
            "keluar": count_out,
            "video_result": f"http://localhost:8000/result/{result_path.name}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    background_tasks.add_task(run_task)

    return {"video_id": video_id, "status": "processing"}


@app.get("/progress/{video_id}")
def get_progress(video_id: str):
    if video_id not in progress_store:
        return {"progress": 0, "status": "processing"}

    if isinstance(progress_store[video_id], dict):
        return progress_store[video_id]  # sudah selesai

    return {"progress": progress_store[video_id], "status": "processing"}


@app.get("/result/{video_name:path}")
def get_result_video(video_name: str):
    path = RESULT_DIR / video_name
    print("Looking for file:", path)  # Debug log
    if path.exists():
        return FileResponse(path, media_type="video/mp4")
    return JSONResponse(content={"error": f"Video {video_name} not found"}, status_code=404)


@app.post("/cancel/{video_id}")
def cancel_video(video_id: str):
    if video_id not in progress_store:
        raise HTTPException(status_code=404, detail="Video not found or not started")
    cancel_flags[video_id] = True
    print("Proceses Canceled")
    return {"status": "canceled"}

# --- Detection Logic ---
def run_detection(video_path: Path, video_id: str):
    video_info = VideoInfo.from_video_path(video_path)
    generator = get_video_frames_generator(video_path)
    sink_path = RESULT_DIR / f"output_{video_id}.mp4"

    
    tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.2)

    line_start = Point(0, video_info.height // 2)
    line_end = Point(video_info.width, video_info.height // 2)
    line_counter = LineZone(start=line_start, end=line_end)
    line_annotator = LineZoneAnnotator(thickness=4, text_thickness=2, text_scale=1)
    box_annotator = BoxAnnotator(color_lookup=ColorLookup.TRACK)

    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        str(sink_path),
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height)
    )

    total_frames = video_info.total_frames
    for idx, frame in enumerate(generator, start=1):
        if cancel_flags.get(video_id):
            out.release()
            progress_store[video_id] = {"status": "canceled"}
            return sink_path, 0, 0

        
        results = model(frame, classes=[2, 3, 5, 7])[0]

        sort_inputs = []
        class_ids = []
        for box, conf, cls in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy()
        ):
            sort_inputs.append([*box, conf])
            class_ids.append(int(cls))

        
        if len(sort_inputs) == 0:
            sort_outputs = tracker.update(np.empty((0, 5)))
            class_ids_tracked = []
        else:
            sort_outputs = tracker.update(np.array(sort_inputs))
            
            class_ids_tracked = class_ids[:len(sort_outputs)]

        if len(sort_outputs) == 0:
            out.write(frame)
        else:
            track_xyxy = sort_outputs[:, :4]
            track_ids = sort_outputs[:, 4].astype(int)

            
            if sort_outputs.shape[1] > 4:
                confidence = sort_outputs[:, 4]
            else:
                confidence = np.ones(len(track_ids), dtype=np.float32)

            
            class_id = np.array(class_ids_tracked, dtype=int)

            detections = Detections(
                xyxy=track_xyxy,
                confidence=confidence,
                class_id=class_id,
                tracker_id=track_ids
            )

           
            line_counter.trigger(detections)

            
            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections
            )
            annotated_frame = line_annotator.annotate(annotated_frame, line_counter)

            out.write(annotated_frame)

        
        progress_store[video_id] = round((idx / total_frames) * 100, 2)

    out.release()
    return sink_path, line_counter.in_count, line_counter.out_count

