from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import threading
import random
import io

from channel_simulator import ChannelSimulator
from satellite_files import generate_file, generate_batch
from decision_engine import DecisionEngine
from mission_configs import MISSION_CONFIGS

app = FastAPI(title="MarsAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
engine: Optional[DecisionEngine] = None
channel = ChannelSimulator("mars")
current_mission = "mars"
file_queue = []
sent_files = []
stats = {
    "total_collected_mb": 0.0,
    "total_transmitted_mb": 0.0,
    "total_dropped_mb": 0.0,
    "files_sent": 0,
    "files_pending": 0,
    "anomalies_detected": 0,
    "bandwidth_saved_mb": 0.0,
    "ticks": 0,
}

def load_models():
    global engine
    engine = DecisionEngine()
    # Pre-fill queue
    global file_queue
    batch = generate_batch(8, current_mission)
    for f in batch:
        f["mission"] = current_mission
        stats["total_collected_mb"] += f["size_mb"]
    file_queue = batch

threading.Thread(target=load_models, daemon=True).start()


@app.get("/status")
def get_status():
    return {
        "status": "online",
        "models_ready": engine is not None,
        "mission": current_mission,
        "channel": channel.get_state(),
        "stats": stats,
        "queue_size": len(file_queue),
        "sent_count": len(sent_files),
    }


@app.get("/files")
def get_files():
    return {
        "queue": file_queue[-20:],
        "sent": sent_files[-10:],
    }


@app.post("/tick")
def tick():
    global file_queue, sent_files, stats

    if engine is None:
        return {"status": "loading", "message": "AI models still initializing..."}

    stats["ticks"] += 1
    channel_state = channel.update()
    cfg = MISSION_CONFIGS[current_mission]
    bw_min, bw_max = cfg["bandwidth_range"]

    # Add new files every 2 ticks
    if stats["ticks"] % 2 == 0:
        new_file = generate_file(current_mission)
        new_file["mission"] = current_mission
        stats["total_collected_mb"] += new_file["size_mb"]
        file_queue.append(new_file)

    # Run AI decision on all queued files
    for f in file_queue:
        if f["status"] == "pending":
            f["mission"] = current_mission
            result = engine.decide(f, channel_state)
            f.update(result)
            if result["is_anomaly"]:
                stats["anomalies_detected"] += 1

    # Sort: critical first, then by ai_score
    status_order = {"critical": 0, "sending": 1, "queued": 2, "pending": 3}
    file_queue.sort(key=lambda x: (status_order.get(x["status"], 4), -x.get("ai_score", 0)))

    # Available bandwidth per tick (3 sec)
    available_mb = channel_state["bandwidth_mbps"] * 3 * 0.125 * 8  # Mbps * seconds * conversion
    sent_this_tick = []

    # Transmission policy depends on mission:
    # Mars/DeepSpace: only critical/sending/queued (every byte precious)
    # Lunar/Satellite: also pending files (good bandwidth, generous window)
    if current_mission in ["satellite", "lunar"]:
        transmit_statuses = ["critical", "sending", "queued", "pending"]
        size_limit_multiplier = 5  # can send bigger files
    else:
        transmit_statuses = ["critical", "sending", "queued"]
        size_limit_multiplier = 2

    for f in file_queue[:]:
        if available_mb <= 0:
            break
        if f["status"] in transmit_statuses and f["size_mb"] <= available_mb * size_limit_multiplier:
            available_mb -= f["size_mb"]
            f["status"] = "sent"
            stats["total_transmitted_mb"] += f["size_mb"]
            stats["files_sent"] += 1
            sent_this_tick.append(f)
            file_queue.remove(f)
            sent_files.append(f)
            if len(sent_files) > 50:
                sent_files.pop(0)

    # Drop low priority if queue too large
    # Mars/DeepSpace: aggressive dropping, Lunar/Satellite: more lenient
    max_queue = 12 if current_mission in ["mars", "deepspace"] else 25
    if len(file_queue) > max_queue:
        to_drop = [f for f in file_queue if f["status"] == "pending"]
        drop_count = 5 if current_mission in ["mars", "deepspace"] else 2
        for f in to_drop[:drop_count]:
            stats["total_dropped_mb"] += f["size_mb"]
            file_queue.remove(f)

    stats["files_pending"] = len(file_queue)
    predicted_bw = engine.predict_channel(bw_min, bw_max)

    return {
        "status": "ok",
        "mission": current_mission,
        "channel": channel_state,
        "queue": file_queue[-20:],
        "sent_this_tick": sent_this_tick,
        "stats": stats,
        "predicted_bandwidth": float(predicted_bw),
    }


class MissionSwitch(BaseModel):
    mission: str

@app.post("/mission")
def switch_mission(body: MissionSwitch):
    global current_mission, file_queue, sent_files, stats
    mission = body.mission
    if mission not in MISSION_CONFIGS:
        return {"error": f"Unknown mission: {mission}"}

    current_mission = mission
    channel.set_mission(mission)
    file_queue = []
    sent_files = []
    stats = {
        "total_collected_mb": 0.0,
        "total_transmitted_mb": 0.0,
        "total_dropped_mb": 0.0,
        "files_sent": 0,
        "files_pending": 0,
        "anomalies_detected": 0,
        "bandwidth_saved_mb": 0.0,
        "ticks": 0,
    }

    batch = generate_batch(8, mission)
    for f in batch:
        f["mission"] = mission
        stats["total_collected_mb"] += f["size_mb"]
    file_queue = batch

    return {"status": "ok", "mission": mission, "channel": channel.get_state()}


@app.post("/reset")
def reset():
    global file_queue, sent_files, stats
    file_queue = []
    sent_files = []
    stats = {
        "total_collected_mb": 0.0,
        "total_transmitted_mb": 0.0,
        "total_dropped_mb": 0.0,
        "files_sent": 0,
        "files_pending": 0,
        "anomalies_detected": 0,
        "bandwidth_saved_mb": 0.0,
        "ticks": 0,
    }
    batch = generate_batch(8, current_mission)
    for f in batch:
        f["mission"] = current_mission
        stats["total_collected_mb"] += f["size_mb"]
    file_queue = batch
    return {"status": "reset", "mission": current_mission}


class AnalyzeRequest(BaseModel):
    file: Dict[str, Any]
    mission: str = "mars"

@app.post("/analyze")
def analyze_file(body: AnalyzeRequest):
    if engine is None:
        return {"error": "Models still loading, please wait"}

    file = body.file
    file["mission"] = body.mission

    if "sensor_data" not in file:
        file["sensor_data"] = {
            "temperature": -25.0,
            "pressure": 729.0,
            "chemical_index": 0.15,
            "radiation_level": 0.3,
            "humidity": 0.01,
        }

    result = engine.decide(file, channel.get_state())
    return result


@app.get("/channel/history")
def channel_history():
    return {"history": channel.get_history()}


@app.get("/missions")
def get_missions():
    return {"missions": list(MISSION_CONFIGS.keys()), "current": current_mission}


@app.post("/analyze_pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    mission: str = Form(default="mars")
):
    """
    Upload a PDF file — extract text, run Sentence Transformer,
    return scientific value score and AI decision.
    """
    if engine is None:
        return {"error": "Models still loading, please wait"}

    # Read PDF bytes
    pdf_bytes = await file.read()
    
    # Extract text from PDF
    text = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        text = " ".join(pages_text)
        # Limit to first 2000 chars for speed
        text = text[:2000].strip()
    except Exception as e:
        return {"error": f"PDF parsing failed: {str(e)}. Make sure it is a valid PDF."}

    if not text or len(text) < 20:
        return {"error": "Could not extract text from PDF. Try a text-based PDF (not scanned image)."}

    # Run semantic analysis
    semantic_score = engine.analyze_semantic(text, mission)

    # Determine file size estimate from bytes
    size_mb = round(len(pdf_bytes) / (1024 * 1024), 2)

    # Build file object for full AI pipeline
    pdf_file = {
        "id": "pdf_upload",
        "name": file.filename or "uploaded.pdf",
        "type": "PDF",
        "size_mb": max(size_mb, 0.1),
        "description": text[:500],
        "mission": mission,
        "sensor_data": {
            "temperature": -25.0,
            "pressure": 729.0,
            "chemical_index": min(1.0, semantic_score * 0.8),
            "radiation_level": 0.3,
            "humidity": 0.01,
        },
    }

    result = engine.decide(pdf_file, channel.get_state())
    result["extracted_text_preview"] = text[:300]
    result["text_length"] = len(text)
    result["filename"] = file.filename
    result["semantic_score"] = semantic_score

    return result