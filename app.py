import os
import shutil
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from decision_layer import predict_from_file
from config import ORGANS


class UserType(str, Enum):
    donor = "donor"
    patient = "patient"


app = FastAPI(
    title="STODS NLP AI Service",
    description="Medical Report Analysis — Organ Status Detection using ClinicalBERT",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@app.get("/")
async def root():
    return {
        "service": "STODS NLP AI Service",
        "status": "running",
        "version": "1.0.0",
        "endpoint": "/analyze-report"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }


@app.post("/analyze-report")
async def analyze_report(
    user_type: UserType = Form(...),
    user_id: str = Form(...),
    report: UploadFile = File(...)
):
    # Validate file extension
    filename = report.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .txt and .pdf are allowed."
        )

    file_path = os.path.join(
        UPLOAD_DIR,
        f"{user_type}_{user_id}_{filename}"
    )

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(report.file, buffer)

        # Run NLP Model
        pid, final_predictions, alerts = predict_from_file(file_path)

        return {
            "status": "success",
            "user_type": user_type,
            "user_id": user_id,
            "nlp_patient_id": pid,
            "organs": {
                organ: final_predictions.get(organ, "unknown")
                for organ in ORGANS
            },
            "alerts": alerts
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
```
