import os
import base64
import joblib
import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from flask import Flask, request, jsonify
from flask_cors import CORS

# ===================== Init =====================
app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["https://creative-pavlova-c07a2b.netlify.app"])

# ===================== Load Assets =====================
base_dir = os.path.dirname(__file__)
load = lambda name: joblib.load(os.path.join(base_dir, name))

model_simple = load("model_simple_2024.pkl")
model_complex = load("model_complex_2024.pkl")

master_df = pd.read_excel(os.path.join(base_dir, "Export material master data.xlsx"))
master_df.set_index("Material", inplace=True)

special_material_ids = load("special_materials_auto.pkl")

# ===================== Google Sheets =====================
sheet = None
try:
    creds_b64 = os.environ["GOOGLE_CREDS_B64"]
    creds_json = base64.b64decode(creds_b64).decode("utf-8")

    creds_path = "temp_google_creds.json"
    with open(creds_path, "w") as f:
        f.write(creds_json)

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Pallet_feedback_log").sheet1
except Exception as e:
    print("⚠️ Google Sheet setup failed:", e)

# ===================== Feature Computation =====================
def compute_features(material_list):
    total_quantity = 0
    total_weight = 0
    total_volume = 0
    special_materials = []
    material_ids = []

    for item in material_list:
        mat_id = int(item["id"])
        qty = float(item["quantity"])
        total_quantity += qty
        material_ids.append(mat_id)

        if mat_id in special_material_ids:
            special_materials.append(str(mat_id))

        if mat_id in master_df.index:
            row = master_df.loc[mat_id]
            try:
                weight = float(row.get("Weight", 0) or 0)
                length = float(row.get("Length", 0) or 0)
                width = float(row.get("Width", 0) or 0)
                height = float(row.get("Height", 0) or 0)

                volume_cm3 = length * width * height
                total_weight += weight * qty
                total_volume += volume_cm3 * qty

                print(f"✅ {mat_id}: W={weight}, L={length}, W={width}, H={height}, Q={qty}")
            except Exception as e:
                print(f"⚠️ Error processing material {mat_id}: {e}")
        else:
            print(f"❌ Material ID {mat_id} not found in Excel")

    unique_materials = len(set(material_ids))
    task_count = len(material_ids)
    avg_density = total_weight / total_volume if total_volume else 0

    return {
        "Unique_Materials": unique_materials,
        "Task_Count": task_count,
        "Total_Quantity": total_quantity,
        "Total_Weight": total_weight,
        "Total_Volume": total_volume,
        "Avg_Packing_Density": avg_density,
        "Special_Materials": special_materials
    }

# ===================== Prediction Logic =====================
def predict_pallets(features):
    is_simple = (
        features["Unique_Materials"] <= 2 and
        features["Task_Count"] <= 2 and
        features["Total_Quantity"] <= 20
    )
    model = model_simple if is_simple else model_complex
    model_type = "Simple" if is_simple else "Complex"

    X = np.array([
        features["Unique_Materials"],
        features["Task_Count"],
        features["Total_Quantity"],
        features["Total_Weight"],
        features["Total_Volume"],
        features["Avg_Packing_Density"]
    ]).reshape(1, -1)

    pallets = int(round(model.predict(X)[0]))

    response = {
        "predicted_pallets": max(pallets, 1),
        "model_used": model_type,
        "Unique_Materials": features["Unique_Materials"],
        "Task_Count": features["Task_Count"],
        "Total_Quantity": features["Total_Quantity"],
        "Total_Weight": features["Total_Weight"],
        "Total_Volume": features["Total_Volume"],
        "Avg_Packing_Density": features["Avg_Packing_Density"],
        "Special_Materials": features.get("Special_Materials", [])
    }

    print("✅ Final response →", response)
    return response

# ===================== Routes =====================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "materials" not in data:
        return jsonify({"error": "Missing material data"}), 400

    features = compute_features(data["materials"])
    result = predict_pallets(features)
    return jsonify(result)

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    try:
        data = request.get_json()
        actual = data.get("actual_pallets")
        predicted = data.get("predicted_pallets")
        model_used = data.get("model_used")

        if sheet:
            sheet.append_row([predicted, actual, model_used])
            return jsonify({"message": "Feedback submitted"}), 200
        else:
            return jsonify({"error": "Google Sheet not initialized"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "✅ Pallet Prediction API is running!"

# ===================== Run App =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
