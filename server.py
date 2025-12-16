# ==========================================================
# FASTAPI SERVER (Fireworks.ai + OCR)
# ==========================================================

import os
import json
import tempfile
import re
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import requests

# ----------------------------------------------------------
# ENV + API SETUP
# ----------------------------------------------------------
load_dotenv()

FIRE_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIRE_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3-8b-instruct")
FIRE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"

if not FIRE_API_KEY:
    raise Exception("❌ FIREWORKS_API_KEY missing in .env")

HEADERS = {
    "Authorization": f"Bearer {FIRE_API_KEY}",
    "Content-Type": "application/json"
}

# ----------------------------------------------------------
# FastAPI APP
# ----------------------------------------------------------
app = FastAPI(title="HealthGuard AI Backend (Fireworks Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------
# Pydantic Models
# ----------------------------------------------------------
class SymptomRequest(BaseModel):
    message: str
    history: list = []

class LifestyleRequest(BaseModel):
    inputData: dict

# ----------------------------------------------------------
# Fireworks Helper Function
# ----------------------------------------------------------
def ask_fireworks(prompt: str, model: str = FIRE_MODEL, max_tokens: int = 800):
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.6
        }

        res = requests.post(FIRE_URL, json=payload, headers=HEADERS)
        res.raise_for_status()

        data = res.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fireworks Error: {e}")

# ----------------------------------------------------------
# SYMPTOM CHECKER
# ----------------------------------------------------------
@app.post("/symptom-checker")
def symptom_checker(req: SymptomRequest):
    symptoms = req.message.strip()

    prompt = f"""
You are a medical assistant. Analyze symptoms and return STRICT JSON ONLY.

Example format:
{{
  "conditions": [{{ "name": "Condition A", "probability": "70%" }}],
  "tests": [{{ "name": "Test A", "description": "Reason" }}],
  "remedies": [{{ "name": "Remedy A", "description": "How it helps" }}],
  "doctor": {{ "when_to_visit": "Guidance" }},
  "disclaimer": {{ "text": "Disclaimer" }}
}}

Now analyze: {symptoms}
"""

    raw = ask_fireworks(prompt)

    try:
        parsed = json.loads(raw)
    except:
        parsed = {"raw_output": raw}

    return {"response": parsed}

# ----------------------------------------------------------
# LIFESTYLE ENHANCER
# ----------------------------------------------------------
@app.post("/lifestyle-enhancer")
def lifestyle_enhancer(req: LifestyleRequest):
    prompt = f"""
Analyze this health data and produce a structured lifestyle improvement plan.

Data:
{json.dumps(req.inputData, indent=2)}

Write detailed sections:
- Health Summary
- Diet Plan
- Fitness Plan
- Environmental Advice
- Risk Precautions
- Disclaimer
"""

    text = ask_fireworks(prompt, max_tokens=900)
    return {"report": text}

# ----------------------------------------------------------
# OCR PARSING
# ----------------------------------------------------------
def extract_text_from_image(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"[OCR error] {e}"

def extract_medical_values(text: str) -> dict:
    def find(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1) if m else ""

    return {
        "hemoglobin": find(r"hemoglobin[:\s]*([\d\.]+)"),
        "wbc": find(r"wbc[:\s]*([\d\.]+)"),
        "platelets": find(r"platelets[:\s]*([\d\.]+)"),
        "cholesterol": find(r"cholesterol[:\s]*([\d\.]+)"),
        "hdl": find(r"hdl[:\s]*([\d\.]+)"),
        "ldl": find(r"ldl[:\s]*([\d\.]+)"),
        "triglycerides": find(r"triglycerides[:\s]*([\d\.]+)"),
        "tsh": find(r"tsh[:\s]*([\d\.]+)")
    }

# ----------------------------------------------------------
# UPLOAD REPORT (PDF/IMAGE OCR)
# ----------------------------------------------------------
@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    ext = filename.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    extracted_text = ""
    try:
        if ext == "pdf":
            pages = convert_from_path(temp_path)
            for p in pages:
                extracted_text += extract_text_from_image(p) + "\n"
        else:
            img = Image.open(temp_path).convert("RGB")
            extracted_text = extract_text_from_image(img)
    except Exception as e:
        extracted_text = f"[OCR error] {e}"

    parsed_values = extract_medical_values(extracted_text)

    return {
        "filename": filename,
        "raw_text": extracted_text,
        "parsed_values": parsed_values
    }

# ----------------------------------------------------------
# Root
# ----------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Python Fireworks AI Backend Running 🚀"}

# ----------------------------------------------------------
# Run server
# ----------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=5500, reload=True)
