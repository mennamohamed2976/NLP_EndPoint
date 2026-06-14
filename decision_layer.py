import os
import re
import pickle

import torch
import pandas as pd
import spacy
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import ORGANS


MODEL_DIR = "/data/bert_models/bert_models"
RESULTS_DIR = "outputs"
NLP_OUTPUTS_DIR = "nlp_outputs"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(NLP_OUTPUTS_DIR, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

MODELS = {}
TOKENIZERS = {}

LABELS = ["missing", "present", "removed"]


ORGAN_KEYWORDS = {
    "left_kidney": {
        "removed": [
            "left nephrectomy",
            "left kidney removed",
            "left kidney was removed",
            "left kidney surgically removed",
            "left renal removal",
            "left kidney resected",
        ],
        "missing": [
            "left kidney absent",
            "left kidney not seen",
            "left kidney nonvisualization",
            "left kidney could not be identified",
            "left kidney invisible",
        ],
        "present": [
            "left kidney normal",
            "left kidney appears normal",
            "left kidney is present",
            "left kidney intact",
            "left kidney visualized",
            "left kidney seen",
        ],
    },
    "right_kidney": {
        "removed": [
            "right nephrectomy",
            "right kidney removed",
            "right kidney was removed",
            "right kidney surgically removed",
            "right renal removal",
            "right kidney resected",
        ],
        "missing": [
            "right kidney absent",
            "right kidney not seen",
            "right kidney nonvisualization",
            "right kidney could not be identified",
            "right kidney invisible",
        ],
        "present": [
            "right kidney normal",
            "right kidney appears normal",
            "right kidney is present",
            "right kidney intact",
            "right kidney visualized",
            "right kidney seen",
        ],
    },
    "liver": {
        "removed": [
            "liver removed",
            "hepatectomy",
            "liver resected",
            "liver resection",
            "liver surgically removed",
        ],
        "missing": [
            "liver absent",
            "liver not seen",
            "liver nonvisualization",
            "liver could not be identified",
        ],
        "present": [
            "liver normal",
            "liver appears normal",
            "liver is present",
            "liver intact",
            "liver visualized",
            "liver is normal",
        ],
    },
    "spleen": {
        "removed": [
            "spleen removed",
            "splenectomy",
            "spleen resected",
            "spleen surgically removed",
        ],
        "missing": [
            "spleen absent",
            "spleen not seen",
            "spleen nonvisualization",
            "spleen could not be identified",
        ],
        "present": [
            "spleen normal",
            "spleen appears normal",
            "spleen is present",
            "spleen intact",
            "spleen visualized",
        ],
    },
}

GENERAL_REMOVED_KEYWORDS = [
    "nephrectomy",
    "resection",
    "excision",
    "surgically removed",
    "surgery done",
    "resected",
    "was removed",
    "has been removed",
]

GENERAL_MISSING_KEYWORDS = [
    "nonvisualization",
    "absent",
    "not seen",
    "could not identify",
    "invisible",
    "lost",
    "not identified",
]

GENERAL_PRESENT_KEYWORDS = [
    "normal",
    "visualized",
    "seen",
    "intact",
    "appears normal",
    "present",
    "is present",
]

ORGAN_TEXT_ALIASES = {
    "left_kidney": ["left kidney", "left renal", "left-sided kidney"],
    "right_kidney": [
        "right kidney",
        "right renal",
        "right-sided kidney",
        "kidney for donation",
        "kidney was surgically removed",
    ],
    "liver": ["liver", "hepatic"],
    "spleen": ["spleen", "splenic"],
}


def load_models():
    global MODELS, TOKENIZERS

    if MODELS:
        return

    for organ in ORGANS:
        path = os.path.join(MODEL_DIR, organ)

        TOKENIZERS[organ] = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
        )

        MODELS[organ] = AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
        )

        MODELS[organ].eval()


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
            return line.split(":")[-1].strip()

    return "Unknown_PID"


def get_organ_context_sentences(report_text, organ):
    report_lower = report_text.lower()
    aliases = ORGAN_TEXT_ALIASES.get(organ, [])

    sentences = re.split(r"[.\n]", report_lower)
    relevant = [
        sentence.strip()
        for sentence in sentences
        if any(alias in sentence for alias in aliases)
    ]

    return " ".join(relevant) if relevant else report_lower


def predict_organ(text, organ):
    tokenizer = TOKENIZERS[organ]
    model = MODELS[organ]

    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoding)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return LABELS[pred]


def rule_based_override(bert_prediction, organ, report_text):
    organ_context = get_organ_context_sentences(report_text, organ)
    organ_keywords = ORGAN_KEYWORDS.get(organ, {})

    for status in ["removed", "missing", "present"]:
        specific_keywords = organ_keywords.get(status, [])

        if any(keyword in organ_context for keyword in specific_keywords):
            return status

    if any(keyword in organ_context for keyword in GENERAL_REMOVED_KEYWORDS):
        return "removed"

    if any(keyword in organ_context for keyword in GENERAL_MISSING_KEYWORDS):
        return "missing"

    if any(keyword in organ_context for keyword in GENERAL_PRESENT_KEYWORDS):
        return "present"

    return bert_prediction


def decision_layer(bert_predictions, report_text):
    final_predictions = {}
    alerts = {}

    for organ, bert_status in bert_predictions.items():
        final_status = rule_based_override(
            bert_status,
            organ,
            report_text,
        )

        final_predictions[organ] = final_status

        organ_context = get_organ_context_sentences(report_text, organ)
        organ_keywords = ORGAN_KEYWORDS.get(organ, {})

        has_removed_keyword = any(
            keyword in organ_context
            for keyword in organ_keywords.get("removed", []) + GENERAL_REMOVED_KEYWORDS
        )

        has_missing_keyword = any(
            keyword in organ_context
            for keyword in organ_keywords.get("missing", []) + GENERAL_MISSING_KEYWORDS
        )

        if final_status == "removed":
            alerts[organ] = (
                "✅ Explained Removal — keyword confirmed"
                if has_removed_keyword
                else "🚨 Unexplained Removal — no keyword found in report"
            )

        elif final_status == "missing":
            alerts[organ] = (
                "⚠️ Model Confusion (Missing vs Removed) — check laterality"
                if has_removed_keyword
                else "🚨 Suspicious Missing — organ absent without explanation"
            )

        elif final_status == "present":
            alerts[organ] = (
                "⚠️ Possible Contradiction — organ marked present but removal/missing keyword found"
                if has_removed_keyword or has_missing_keyword
                else "✅ No Issue — organ present and confirmed"
            )

    return final_predictions, alerts


def save_results(pid, final_predictions, alerts):
    csv_data = {
        "PID": pid,
    }

    for organ in ORGANS:
        csv_data[f"{organ}_prediction"] = final_predictions.get(
            organ,
            "unknown",
        )
        csv_data[f"{organ}_alert"] = alerts.get(
            organ,
            "",
        )

    csv_path = os.path.join(
        RESULTS_DIR,
        f"{pid}_result.csv",
    )

    pd.DataFrame([csv_data]).to_csv(
        csv_path,
        index=False,
    )

    nlp_results = {
        pid: {
            organ: {
                "prediction": final_predictions.get(organ, "unknown"),
                "alert": alerts.get(organ, ""),
            }
            for organ in ORGANS
        }
    }

    pkl_path = os.path.join(
        NLP_OUTPUTS_DIR,
        f"{pid}.pkl",
    )

    with open(pkl_path, "wb") as file:
        pickle.dump(nlp_results, file)

    return csv_path, pkl_path


def read_report(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

    elif ext == ".pdf":
        reader = PdfReader(file_path)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    else:
        raise ValueError("Only TXT or PDF files are supported!")

    return text


def predict_from_file(file_path):
    load_models()

    raw_text = read_report(file_path)
    pid = extract_pid(raw_text)
    findings_raw = extract_findings_section(raw_text)
    cleaned_text = clean_text_for_bert(findings_raw)

    bert_predictions = {
        organ: predict_organ(cleaned_text, organ)
        for organ in ORGANS
    }

    final_predictions, alerts = decision_layer(
        bert_predictions,
        findings_raw,
    )

    save_results(
        pid,
        final_predictions,
        alerts,
    )

    return pid, final_predictions, alerts


if __name__ == "__main__":
    FILE_PATH = input("Enter file path: ")
    predict_from_file(FILE_PATH)
