from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import threading

from satellite_files import generate_batch, generate_file, advance_sol
from channel_simulator import ChannelSimulator
from decision_engine import DecisionEngine

app = FastAPI(title="MarsAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────
channel = ChannelSimulator()
engine = None  # Lazy init (model loading takes ~10s)
engine_lock = threading.Lock()

file_queue: List[Dict] = []
sent_files: List[Dict] = []
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

def get_engine() -> DecisionEngine:
    global engine
    with engine_lock:
        if engine is None:
            engine = DecisionEngine()
    return engine


def _init_queue():
    global file_queue, stats
    files = generate_batch(8)
    channel_state = channel.get_state()
    eng = get_engine()

    for f in files:
        result = eng.decide(f, channel_state)
        f.update(result)
        stats["total_collected_mb"] += f["size_mb"]

    file_queue = files
    stats["files_pending"] = len([f for f in file_queue if f["status"] == "pending"])


# ── Startup ───────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    # Load models in background thread so server starts fast
    def load():
        get_engine()
        _init_queue()
        print("[API] Ready ✓")
    t = threading.Thread(target=load, daemon=True)
    t.start()


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/status")
def get_status():
    return {
        "status": "online",
        "models_ready": engine is not None,
        "channel": channel.get_state(),
        "stats": stats,
        "queue_size": len(file_queue),
        "sent_count": len(sent_files),
    }


@app.get("/files")
def get_files():
    return {
        "queue": file_queue,
        "sent": sent_files[-20:],  # last 20 sent
        "channel": channel.get_state(),
    }


@app.get("/channel/history")
def get_channel_history():
    return {"history": channel.get_history()}


@app.post("/tick")
def tick():
    """
    One simulation cycle:
    1. Update channel
    2. Add new file
    3. Process queue — send critical/high-value files
    4. Return updated state
    """
    global file_queue, sent_files, stats

    if engine is None:
        return {"status": "loading", "message": "AI models still loading, please wait..."}

    # Update channel
    channel_state = channel.update()
    stats["ticks"] += 1

    # Add new file every 2 ticks
    if stats["ticks"] % 2 == 0:
        new_file = generate_file()
        result = engine.decide(new_file, channel_state)
        new_file.update(result)
        stats["total_collected_mb"] += new_file["size_mb"]
        if new_file["is_anomaly"]:
            stats["anomalies_detected"] += 1
        file_queue.append(new_file)

    # Re-evaluate queue with updated channel
    for f in file_queue:
        if f["status"] not in ("critical", "sending"):
            result = engine.decide(f, channel_state)
            f.update(result)

    # Transmit files based on available bandwidth
    available_mb = channel_state["bandwidth_mbps"] * 5  # per tick budget
    transmitted_this_tick = 0.0
    newly_sent = []

    # Priority order: critical → sending → queued
    priority_order = ["critical", "sending", "queued"]
    for priority_status in priority_order:
        for f in list(file_queue):
            if f["status"] == priority_status:
                if transmitted_this_tick + f["size_mb"] <= available_mb:
                    f["status"] = "sent"
                    transmitted_this_tick += f["size_mb"]
                    stats["total_transmitted_mb"] += f["size_mb"]
                    stats["files_sent"] += 1
                    newly_sent.append(f)
                    file_queue.remove(f)

    sent_files.extend(newly_sent)

    # Drop very low priority files if queue too large
    if len(file_queue) > 15:
        to_drop = [f for f in file_queue if f["status"] == "pending" and f["priority"] <= 1]
        for f in to_drop[:3]:
            stats["total_dropped_mb"] += f["size_mb"]
            stats["bandwidth_saved_mb"] += f["size_mb"]
            file_queue.remove(f)

    stats["files_pending"] = len(file_queue)

    return {
        "channel": channel_state,
        "queue": file_queue,
        "sent_this_tick": newly_sent,
        "stats": stats,
        "predicted_bandwidth": engine.predict_channel(),
    }


@app.post("/reset")
def reset():
    global file_queue, sent_files, stats
    file_queue = []
    sent_files = []
    stats.update({
        "total_collected_mb": 0.0,
        "total_transmitted_mb": 0.0,
        "total_dropped_mb": 0.0,
        "files_sent": 0,
        "files_pending": 0,
        "anomalies_detected": 0,
        "bandwidth_saved_mb": 0.0,
        "ticks": 0,
    })
    _init_queue()
    return {"status": "reset", "message": "Simulation restarted"}


class MarsDelayUpdate(BaseModel):
    minutes: float

@app.get("/mars-delay")
def get_mars_delay():
    return {"mars_delay_minutes": channel.mars_delay_minutes}

@app.post("/mars-delay")
def set_mars_delay(body: MarsDelayUpdate):
    channel.set_mars_delay(body.minutes)
    return {"mars_delay_minutes": channel.mars_delay_minutes}
