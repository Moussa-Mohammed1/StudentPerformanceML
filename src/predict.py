"""Prediction script for student final grade.

Loads the saved model and scaler, then predicts G3 for a new student.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.train import load_model
from src.preprocessing import load_scaler, get_feature_names


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def predict_student(student_data: dict) -> float:
    model = load_model()
    scaler = load_scaler()
    feature_names = get_feature_names()

    input_df = pd.DataFrame([student_data])
    categorical_cols = input_df.select_dtypes(include=["object", "str"]).columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    input_scaled = scaler.transform(input_encoded)
    prediction = model.predict(input_scaled)[0]
    return round(prediction, 2)


if __name__ == "__main__":
    sample_student = {
        "school": "GP",
        "sex": "M",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "teacher",
        "Fjob": "teacher",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 5,
        "freetime": 4,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 2,
    }

    grade = predict_student(sample_student)
    print(f"Predicted final grade (G3): {grade}")
