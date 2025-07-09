# utils.py
import pandas as pd

def load_data():
    """Load and clean datasets."""
    def clean(df):
        df['Disease'] = df['Disease'].str.strip().str.lower()
        return df

    return {
        "desc": clean(pd.read_csv("datasets/description.csv")),
        "meds": clean(pd.read_csv("datasets/medications.csv")),
        "diet": clean(pd.read_csv("datasets/diets.csv")),
        "prec": clean(pd.read_csv("datasets/precautions_df.csv")),
        "workout": clean(pd.read_csv("datasets/workout_df.csv"))
    }

def get_recommendations(disease, data):
    disease = disease.strip().lower()

    def get_items(df):
        if disease in df['Disease'].values:
            return df[df['Disease'] == disease].iloc[0, 1:].dropna().tolist()
        return []

    try:
        return {
            "description": data["desc"][data["desc"]['Disease'] == disease]["Description"].values[0]
            if disease in data["desc"]['Disease'].values else "N/A",
            "medications": get_items(data["meds"]),
            "diet": get_items(data["diet"]),
            "precautions": get_items(data["prec"]),
            "workout": get_items(data["workout"]),
        }
    except Exception as e:
        return {
            "description": "N/A",
            "medications": [],
            "diet": [],
            "precautions": [],
            "workout": [],
            "error": str(e)
        }
