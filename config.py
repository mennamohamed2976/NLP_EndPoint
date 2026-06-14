# src/config.py

DATA_PATH = "C:/Users/Mona/OneDrive/Desktop/Medical_Report_Analysis2/Data/raw/nlp_medical_reports_1500_balanced.csv"
PROCESSED_DATA_PATH = "C:/Users/Mona/OneDrive/Desktop/Medical_Report_Analysis2/Data/raw/processed_reports.csv"

TEXT_COLUMN = "text"
ORGANS = ["left_kidney", "right_kidney", "liver", "spleen"]

CLASSES = ["present", "removed", "missing"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
