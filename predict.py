import os
import tempfile
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

# 🔥 YOUR FUNCTIONS
from predict_video import raahat_predict_video
from predict_audio import raahat_predict_audio


# ══════════ CONFIG ══════════
DEBUG_VIDEO_DIR = "debug_videos"
os.makedirs(DEBUG_VIDEO_DIR, exist_ok=True)
# ════════════════════════════


app = FastAPI(
    title="Raahat Local Testing API",
    version="1.0.0",
)


# ══════════ RESPONSE MODEL ══════════

class PredictResponse(BaseModel):
    line: str
    vehicle_count: int
    density: str
    avg_speed: float
    emergency: bool
    audio_used: bool

    video_score: float
    audio_score: float
    final_score: float


# ══════════ HEALTH CHECK ══════════

@app.get("/health")
async def health():
    return {"status": "ok"}


# ══════════ MAIN API (UPLOAD VIDEO) ══════════

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    lane_id: str = Form(...),
    line_type: Optional[str] = Form(None),
):
    tmp_video_path = None

    try:
        # ── 1. SAVE UPLOADED VIDEO ──
        tmp_fd, tmp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)

        with open(tmp_video_path, "wb") as f:
            f.write(await file.read())

        print(f"✅ Video uploaded → {tmp_video_path}")

        # ── 2. LINE DETECTION ──
        line = _derive_line(lane_id, line_type)

        # ── 3. OUTPUT VIDEO ──
        output_video_path = os.path.join(
            DEBUG_VIDEO_DIR,
            f"{lane_id}_{int(time.time())}.mp4",
        )

        # ── 4. VIDEO MODEL ──
        print("🚀 Running video model...")
        video_result = raahat_predict_video(
            input_video_path=tmp_video_path,
            output_video_path=output_video_path,
            line=line
        )

        # ── 5. AUDIO MODEL ──
        print("🎧 Running audio model...")
        audio_used = True
        audio_emergency = False
        audio_confidence = 0.0

        try:
            audio_result = raahat_predict_audio(tmp_video_path)

            if "error" in audio_result:
                audio_used = False
            else:
                audio_emergency = audio_result["emergency_audio"]
                audio_confidence = audio_result["confidence"]

        except Exception as e:
            print(f"⚠️ Audio failed: {e}")
            audio_used = False

        # ── 6. 🔥 FUSION LOGIC ──

        video_emergency = video_result["emergency_video"]

        video_score = 0.8 if video_emergency else 0.2

        if audio_used and audio_emergency:
            audio_score = audio_confidence
        else:
            audio_score = 0.2

        final_score = 0.4 * video_score + 0.6 * audio_score
        final_emergency = final_score >= 0.65

        # ── 7. RESPONSE ──
        return {
            "line": video_result["line"],
            "vehicle_count": video_result["vehicle_count"],
            "density": video_result["density"],
            "avg_speed": video_result["avg_speed"],
            "emergency": final_emergency,
            "audio_used": audio_used,
            "video_score": round(video_score, 3),
            "audio_score": round(audio_score, 3),
            "final_score": round(final_score, 3),
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)


# ══════════ HELPER ══════════

def _derive_line(lane_id: str, line_type: Optional[str]) -> str:
    if line_type:
        return line_type

    lane = lane_id.upper()

    if lane in ("A", "C"):
        return "horizontal"
    elif lane in ("B", "D"):
        return "vertical"

    return "horizontal"


# ══════════ RUN ══════════

if __name__ == "__main__":
    uvicorn.run(
        "predict:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )