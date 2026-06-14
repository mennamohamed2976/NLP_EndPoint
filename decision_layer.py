import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import re
import os
import pickle          # ✅ لحفظ الموديل كـ pickle لو محتاج
import joblib          # ✅ الأفضل لحفظ الموديلات
import pandas as pd
from config import ORGANS
from PyPDF2 import PdfReader


MODEL_DIR = "models/bert_models"
RESULTS_DIR = "outputs"
NLP_OUTPUTS_DIR = "nlp_outputs"      # ✅ مجلد منفصل للـ pkl
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(NLP_OUTPUTS_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

# Label order تتطابق مع sklearn LabelEncoder الترتيب الأبجدي:
# missing=0, present=1, removed=2
LABELS = ["missing", "present", "removed"]

ORGAN_KEYWORDS = {
    "left_kidney": {
        "removed": ["left nephrectomy", "left kidney removed", "left kidney was removed",
                    "left kidney surgically removed", "left renal removal", "left kidney resected"],
        "missing": ["left kidney absent", "left kidney not seen", "left kidney nonvisualization",
                    "left kidney could not be identified", "left kidney invisible"],
        "present": ["left kidney normal", "left kidney appears normal", "left kidney is present",
                    "left kidney intact", "left kidney visualized", "left kidney seen"]
    },
    "right_kidney": {
        "removed": ["right nephrectomy", "right kidney removed", "right kidney was removed",
                    "right kidney surgically removed", "right renal removal", "right kidney resected"],
        "missing": ["right kidney absent", "right kidney not seen", "right kidney nonvisualization",
                    "right kidney could not be identified", "right kidney invisible"],
        "present": ["right kidney normal", "right kidney appears normal", "right kidney is present",
                    "right kidney intact", "right kidney visualized", "right kidney seen"]
    },
    "liver": {
        "removed": ["liver removed", "hepatectomy", "liver resected", "liver resection",
                    "liver surgically removed"],
        "missing": ["liver absent", "liver not seen", "liver nonvisualization",
                    "liver could not be identified"],
        "present": ["liver normal", "liver appears normal", "liver is present",
                    "liver intact", "liver visualized", "liver is normal"]
    },
    "spleen": {
        "removed": ["spleen removed", "splenectomy", "spleen resected",
                    "spleen surgically removed"],
        "missing": ["spleen absent", "spleen not seen", "spleen nonvisualization",
                    "spleen could not be identified"],
        "present": ["spleen normal", "spleen appears normal", "spleen is present",
                    "spleen intact", "spleen visualized", "spleen is present"]
    }
}

GENERAL_REMOVED_KEYWORDS = [
    "nephrectomy", "resection", "excision", "surgically removed",
    "surgery done", "resected", "was removed", "has been removed"
]
GENERAL_MISSING_KEYWORDS = [
    "nonvisualization", "absent", "not seen", "could not identify",
    "invisible", "lost", "not identified"
]
GENERAL_PRESENT_KEYWORDS = [
    "normal", "visualized", "seen", "intact", "appears normal", "present", "is present"
]

ORGAN_TEXT_ALIASES = {
    "left_kidney":  ["left kidney", "left renal", "left-sided kidney"],
    "right_kidney": ["right kidney", "right renal", "right-sided kidney",
                     "kidney for donation", "kidney was surgically removed"],
    "liver":        ["liver", "hepatic"],
    "spleen":       ["spleen", "splenic"]
}


# ─────────────────────────────────────────────
# تحميل الموديلات مرة واحدة عند start الملف
# ─────────────────────────────────────────────
MODELS = {}
TOKENIZERS = {}
for organ in ORGANS:
    path = f"{MODEL_DIR}/{organ}"
    TOKENIZERS[organ] = AutoTokenizer.from_pretrained(path)
    MODELS[organ] = AutoModelForSequenceClassification.from_pretrained(path)
    MODELS[organ].eval()


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def clean_text_for_bert(text):
    text = text.lower()
    text = re.sub(r"[^\w\s.,\-_/()]", "", text)
    text = " ".join(text.split())
    return text


def extract_findings_section(text):
    text_lower = text.lower()
    if "findings" in text_lower:
        start_idx = text_lower.index("findings")
        return text[start_idx + len("findings"):]
    return text


def extract_pid(text):
    lines = text.split("\n")
    for line in lines:
        if "patient id" in line.lower():
            pid = line.split(":")[-1].strip()
            return pid
    return "Unknown_PID"


def get_organ_context_sentences(report_text, organ):
    """استخرج الجمل المتعلقة بالعضو ده بس — منع تأثير الأعضاء على بعض."""
    report_lower = report_text.lower()
    aliases = ORGAN_TEXT_ALIASES.get(organ, [])
    sentences = re.split(r'[.\n]', report_lower)
    relevant = [s.strip() for s in sentences if any(alias in s for alias in aliases)]
    return " ".join(relevant) if relevant else report_lower


# ─────────────────────────────────────────────
# BERT Prediction
# ─────────────────────────────────────────────

def predict_organ(text, organ):
    """تشغيل موديل BERT لعضو معين — بيرجع missing / present / removed."""
    tokenizer = TOKENIZERS[organ]
    model = MODELS[organ]
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encoding)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return LABELS[pred]


# ─────────────────────────────────────────────
# Rule-Based Override
# ─────────────────────────────────────────────

def rule_based_override(bert_prediction, organ, report_text):
    """
    بيراجع الـ keywords الخاصة بكل عضو.
    لو لقى keyword واضح — يبدّل نتيجة BERT.
    لو ملقاش — يثق في BERT.
    """
    organ_context = get_organ_context_sentences(report_text, organ)
    organ_keywords = ORGAN_KEYWORDS.get(organ, {})

    # أولاً: Keywords خاصة بالعضو (أعلى دقة)
    for status in ["removed", "missing", "present"]:
        specific_kw = organ_keywords.get(status, [])
        if any(kw in organ_context for kw in specific_kw):
            return status

    # ثانياً: Keywords عامة على سياق العضو بس
    if any(kw in organ_context for kw in GENERAL_REMOVED_KEYWORDS):
        return "removed"
    if any(kw in organ_context for kw in GENERAL_MISSING_KEYWORDS):
        return "missing"
    if any(kw in organ_context for kw in GENERAL_PRESENT_KEYWORDS):
        return "present"

    return bert_prediction  # ثق في BERT لو ملقيتش حاجة


# ─────────────────────────────────────────────
# Decision Layer
# ─────────────────────────────────────────────

def decision_layer(bert_predictions, report_text):
    """
    بياخد نتايج BERT الخام + التقرير →
    يطبّق الـ rule override لكل عضو →
    يولّد alerts →
    يرجع (final_predictions, alerts)
    """
    final_predictions = {}
    alerts = {}

    for organ, bert_status in bert_predictions.items():
        # ── Override ──
        final_status = rule_based_override(bert_status, organ, report_text)
        final_predictions[organ] = final_status

        # ── Keywords للـ Alert ──
        organ_context = get_organ_context_sentences(report_text, organ)
        organ_kw = ORGAN_KEYWORDS.get(organ, {})
        has_removed_kw = any(
            kw in organ_context
            for kw in organ_kw.get("removed", []) + GENERAL_REMOVED_KEYWORDS
        )
        has_missing_kw = any(
            kw in organ_context
            for kw in organ_kw.get("missing", []) + GENERAL_MISSING_KEYWORDS
        )

        # ── Alert Logic ──
        if final_status == "removed":
            alerts[organ] = (
                "✅ Explained Removal — keyword confirmed"
                if has_removed_kw
                else "🚨 Unexplained Removal — no keyword found in report"
            )
        elif final_status == "missing":
            alerts[organ] = (
                "⚠️ Model Confusion (Missing vs Removed) — check laterality"
                if has_removed_kw
                else "🚨 Suspicious Missing — organ absent without explanation"
            )
        elif final_status == "present":
            alerts[organ] = (
                "⚠️ Possible Contradiction — organ marked present but removal/missing keyword found"
                if (has_removed_kw or has_missing_kw)
                else "✅ No Issue — organ present and confirmed"
            )

    return final_predictions, alerts


# ─────────────────────────────────────────────
# ✅ حفظ النتايج — CSV + PKL
# ─────────────────────────────────────────────

def save_results(pid, final_predictions, alerts):
    """
    يحفظ:
    1. CSV في outputs/{pid}_result.csv
    2. PKL في nlp_outputs/{pid}.pkl  ← ده اللي بيتبعت للـ API
    """
    # ── 1. CSV ──
    csv_data = {"PID": pid}
    for organ in ORGANS:
        csv_data[f"{organ}_prediction"] = final_predictions.get(organ, "unknown")
        csv_data[f"{organ}_alert"]      = alerts.get(organ, "")
 
    csv_path = os.path.join(RESULTS_DIR, f"{pid}_result2.csv")
    pd.DataFrame([csv_data]).to_csv(csv_path, index=False)
    print(f"💾 CSV saved: {csv_path}")
 
    # ── 2. PKL (نفس الشكل اللي بيتوقعه send_nlp_to_api) ──
    nlp_results = {
        pid: {
            organ: {
                "prediction": final_predictions.get(organ, "unknown"),
                "alert":      alerts.get(organ, "")
            }
            for organ in ORGANS
        }
    }
    pkl_path = os.path.join(NLP_OUTPUTS_DIR, f"{pid}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(nlp_results, f)
    print(f"💾 PKL saved:  {pkl_path}")
 
    return csv_path, pkl_path


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def read_report(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    else:
        raise ValueError("Only TXT or PDF files are supported!")
    return text


def predict_from_file(file_path):
    # ── 1. قراءة وتنظيف ──
    raw_text     = read_report(file_path)
    pid          = extract_pid(raw_text)
    findings_raw = extract_findings_section(raw_text)
    cleaned_text = clean_text_for_bert(findings_raw)

    # ── 2. BERT — مرة واحدة بس ──
    bert_predictions = {organ: predict_organ(cleaned_text, organ) for organ in ORGANS}

    # ── 3. Decision Layer (override + alerts) ──
    final_predictions, alerts = decision_layer(bert_predictions, findings_raw)

    # ── 4. حفظ النتايج ──
    csv_path, pkl_path = save_results(pid, final_predictions, alerts)

    # ── 5. طباعة ──
    print(f"\n🆔 Patient ID: {pid}")
    print("\n📄 Final Prediction + Decision Layer:")
    for organ in ORGANS:
        bert_orig = bert_predictions[organ]
        final     = final_predictions[organ]
        override_flag = " ← [Rule Override]" if bert_orig != final else ""
        print(f"  {organ}: {final}{override_flag}")
        print(f"    Alert: {alerts[organ]}")

    return pid, final_predictions, alerts


if __name__ == "__main__":
    FILE_PATH = input("Enter file path: ")
    predict_from_file(FILE_PATH)
