import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agent.agent import run_agent
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Soil & Crop Recommendation API")

# CORS — frontend se connect hone ke liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

# ── Health Check ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "message": "Soil & Crop API is live"}

# ── Main Endpoint ─────────────────────────────────────────────
@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    N:           float = Form(...),
    P:           float = Form(...),
    K:           float = Form(...),
    temperature: float = Form(...),
    humidity:    float = Form(...),
    ph:          float = Form(...),
    rainfall:    float = Form(...)
):
    try:
        # Image save karo
        ext = os.path.splitext(image.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        image_path = os.path.join(UPLOAD_DIR, filename)

        with open(image_path, "wb") as f:
            f.write(await image.read())

        # Agent run karo
        parameters = {
            "N": N, "P": P, "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        result = run_agent(image_path, parameters)

        # Image delete karo baad mein
        os.remove(image_path)

        return JSONResponse(content={
            "status": "success",
            "result": result
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )