import os
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.decision_layer import predict_from_file
from src.config import ORGANS

app = FastAPI(
    title="STODS NLP AI Service",
    description="Medical Report Analysis — Organ Status Detection using ClinicalBERT + AI Orchestrator",
    version="2.0.0"
)

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

# ── URLs للـ Services ──
CV_AI_URL  = os.getenv("CV_AI_URL",  "https://your-cv-service.up.railway.app/predict")
NLP_AI_URL = os.getenv("NLP_AI_URL", "https://your-nlp-service.up.railway.app/analyze-report")

# ── Organ Name Mapping بين CV و NLP ──
CV_TO_NLP_ORGAN_MAP = {
    "Liver":    "liver",
    "R_Kidney": "right_kidney",
    "L_Kidney": "left_kidney",
    "Spleen":   "spleen",
}


# ─────────────────────────────────────────────
# Health & Root
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "STODS NLP AI Service",
        "status": "running",
        "version": "2.0.0",
        "endpoints": ["/analyze-report", "/compare-ai"]
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ─────────────────────────────────────────────
# NLP Endpoint
# ─────────────────────────────────────────────

@app.post("/analyze-report")
async def analyze_report(report: UploadFile = File(...)):
    filename = report.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .txt and .pdf are allowed."
        )

    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(report.file, buffer)

        pid, final_predictions, alerts = predict_from_file(file_path)

        return {
            "status": "success",
            "patient_id": pid,
            "organs": {
                organ: final_predictions.get(organ, "unknown")
                for organ in ORGANS
            },
            "alerts": alerts
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ─────────────────────────────────────────────
# Compare Helper
# ─────────────────────────────────────────────

def compare_cv_nlp(cv_organs: dict, nlp_organs: dict):
    comparison = {}
    mismatch_alert = False

    for cv_name, nlp_name in CV_TO_NLP_ORGAN_MAP.items():
        cv_status  = cv_organs.get(cv_name,  "unknown")
        nlp_status = nlp_organs.get(nlp_name, "unknown")
        matched    = cv_status == nlp_status

        if not matched:
            mismatch_alert = True

        comparison[nlp_name] = {
            "cv_status":  cv_status,
            "nlp_status": nlp_status,
            "matched":    matched
        }

    return comparison, mismatch_alert


# ─────────────────────────────────────────────
# AI Orchestrator Endpoint
# ─────────────────────────────────────────────

@app.post("/compare-ai")
async def compare_ai(
    patient_id:  str        = Form(...),
    before_scan: UploadFile = File(...),
    after_scan:  UploadFile = File(...),
    report:      UploadFile = File(...)
):
    try:
        before_bytes = await before_scan.read()
        after_bytes  = await after_scan.read()
        report_bytes = await report.read()

        # ── 1. CV Service ──
        cv_response = requests.post(
            CV_AI_URL,
            data={"patient_id": patient_id},
            files={
                "before_scan": (before_scan.filename, before_bytes, before_scan.content_type),
                "after_scan":  (after_scan.filename,  after_bytes,  after_scan.content_type),
            },
            timeout=300
        )

        if cv_response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail={"service": "CV", "error": cv_response.text}
            )

        cv_result = cv_response.json()

        # ── 2. NLP Service ──
        nlp_response = requests.post(
            NLP_AI_URL,
            files={"report": (report.filename, report_bytes, report.content_type)},
            timeout=120
        )

        if nlp_response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail={"service": "NLP", "error": nlp_response.text}
            )

        nlp_result = nlp_response.json()

        # ── 3. Compare ──
        comparison, mismatch_alert = compare_cv_nlp(
            cv_result.get("organs",  {}),
            nlp_result.get("organs", {})
        )

        return {
            "status":         "success",
            "patient_id":     patient_id,
            "mismatch_alert": mismatch_alert,
            "comparison":     comparison,
            "cv_result":      cv_result,
            "nlp_result":     nlp_result
        }

    except HTTPException:
        raise

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
