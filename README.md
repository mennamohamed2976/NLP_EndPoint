# Medical_Report_Analysis 🧠🩻  
NLP Model for Extracting Organ Presence Status from Medical Reports

This project is part of a broader **Medical Report Analysis System** designed to verify organ presence by comparing:

- **What is written in the clinical report (NLP)**  
  vs  
- **What is detected in MRI/CT scans by the Computer Vision model**

The goal is to detect any **inconsistency** between text and imaging that may indicate human error, reporting mistakes, or critical mismatch.

---

## 🧠 Project Description

Given an English **medical abdominal imaging report** (MRI/CT), the NLP model predicts the **presence status** of four abdominal organs:

- `left_kidney` → `present` / `missing` / `removed`  
- `right_kidney` → `present` / `missing` / `removed`  
- `liver` → `present` / `missing` / `removed`  
- `spleen` → `present` / `missing` / `removed`  

---

## 🔧 Project Structure

```bash
Medical_Report_Analysis2/
├── Data/
│   └── reports/
│       └── sample_report.txt
├── models/
│   └── bert_models/          # Download from Google Drive
│       ├── left_kidney/
│       ├── right_kidney/
│       ├── liver/
│       └── spleen/
├── src/
│   ├── config.py
│   ├── train_bert.py
│   ├── decision_layer.py
│   ├── evaluate_bert.py
│   ├── send_nlp_to_api.py
│   └── utils/
├── outputs/
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/mona-mazen/Medical_Report_Analysis2.git
cd Medical_Report_Analysis2
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Download models
Download `bert_models` from [Google Drive](https://drive.google.com/drive/folders/14rXu8Ny1sqq3ZRo9TgkDuCujL_AaFwsC?usp=drive_link)

Extract to: `models/bert_models/`

---

## 🚀 Run

```bash
# Decision layer only
python src/decision_layer.py

# Send results to API
python src/send_nlp_to_api.py
```

---

## 📊 Models

| Organ | Model | Labels |
|-------|-------|--------|
| left_kidney | Bio_ClinicalBERT | present / missing / removed |
| right_kidney | Bio_ClinicalBERT | present / missing / removed |
| liver | Bio_ClinicalBERT | present / missing / removed |
| spleen | Bio_ClinicalBERT | present / missing / removed |