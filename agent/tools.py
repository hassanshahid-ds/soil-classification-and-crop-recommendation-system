import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import os

# ─── Models Load Karo (ek baar load honge) ────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "models", "final_soil_mobilenetv2_tf_serving_model")
CROP_MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommendation_model.pkl")

print("Loading models...")
image_model = tf.saved_model.load(IMAGE_MODEL_PATH)
crop_model = joblib.load(CROP_MODEL_PATH)
print("Models loaded!")

# ─── Soil Classes ─────────────────────────────────────────────
SOIL_CLASSES = [
    'alluvial_soil', 'black_soil', 'clay_soil', 'loamy_soil',
    'peat_soil', 'red_soil', 'sandy_soil', 'yellow_soil'
]

# ─── Tool 1: Image se Soil Predict ────────────────────────────
def predict_soil(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    infer = image_model.signatures["serving_default"]
    output = infer(tf.constant(img_array))
    
    output_key = list(output.keys())[0]
    predictions = output[output_key].numpy()[0]
    
    soil_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100
    soil_type = SOIL_CLASSES[soil_index]

    return {
        "soil_type": soil_type,
        "confidence": round(confidence, 2)
    }

# ─── Tool 2: Parameters se Crop Recommend ─────────────────────
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall) -> dict:
    import pandas as pd
    
    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    
    crop = crop_model.predict(features)[0]
    probabilities = crop_model.predict_proba(features)[0]
    confidence = float(np.max(probabilities)) * 100
    
    classes = crop_model.classes_
    top_indices = np.argsort(probabilities)[::-1]
    alternatives = [classes[i] for i in top_indices[1:4]]

    return {
        "recommended_crop": crop,
        "confidence": round(confidence, 2),
        "alternative_crops": list(alternatives)
    }

#