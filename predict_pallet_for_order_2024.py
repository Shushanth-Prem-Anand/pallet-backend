# ✅ Local-compatible version of predict_pallet_for_order_2024.py

import joblib
import os
import numpy as np

# Resolve paths relative to current script
base_dir = os.path.dirname(__file__)
load = lambda name: joblib.load(os.path.join(base_dir, name))

model_simple = load("model_simple_2024.pkl")
model_complex = load("model_complex_2024.pkl")
special_classifier = load("special_materials_auto.pkl")


def predict_pallet_for_task(features):
    """
    Predicts the number of pallets and the model used.
    Input: features = {
        "Unique_Materials", "Task_Count", "Total_Quantity",
        "Total_Weight", "Total_Volume", "Avg_Packing_Density",
        "Queue", "Special_Materials" (list of ints)
    }
    """
    # Step 1: Check for special materials
    special_materials = features.get("Special_Materials", [])
    if len(special_materials) > 0:
        special_input = np.array([len(special_materials)]).reshape(1, -1)
        if special_classifier.predict(special_input)[0] == 1:
            return {
                "predicted_pallets": 1,
                "model_used": "Manual override: Special Material"
            }

    # Step 2: Simple or complex routing
    is_simple = (
        features["Unique_Materials"] <= 2 and
        features["Task_Count"] <= 2 and
        features["Total_Quantity"] <= 20
    )

    model = model_simple if is_simple else model_complex
    model_type = "Simple" if is_simple else "Complex"

    # Step 3: Prepare input and predict
    input_array = np.array([
        features["Unique_Materials"],
        features["Task_Count"],
        features["Total_Quantity"],
        features["Total_Weight"],
        features["Total_Volume"],
        features["Avg_Packing_Density"]
    ]).reshape(1, -1)

    pallets = int(round(model.predict(input_array)[0]))

    return {
        "predicted_pallets": max(pallets, 1),
        "model_used": model_type
    }
