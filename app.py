import os
import json
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
TFLITE_MODEL_PATH = "plant_disease_model.tflite"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = (380, 380)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

# ── DISEASE MANAGEMENT ADVICE ─────────────────────────────────────────────────
DISEASE_ADVICE = {
    "Banana_Healthy":              "Your banana plant appears healthy. Continue regular watering and fertilisation.",
    "Banana_Sigatoka":             "Black Sigatoka detected. Remove infected leaves, apply fungicide, and improve drainage.",
    "Banana_Xanthomonas":          "Xanthomonas wilt detected. Remove and destroy infected plants immediately. Disinfect tools.",
    "Cassava_Bacterial_Blight":    "Bacterial blight detected. Use disease-free cuttings and copper-based bactericides.",
    "Cassava_Brown_Spot":          "Brown spot detected. Improve soil fertility and apply appropriate fungicide.",
    "Cassava_Green_Mite":          "Green mite infestation detected. Apply miticide and remove heavily infested leaves.",
    "Cassava_Healthy":             "Your cassava plant appears healthy. Maintain good agricultural practices.",
    "Cassava_Mosaic":              "Cassava mosaic virus detected. Remove infected plants and control whitefly vectors.",
    "Maize_Blight":                "Northern leaf blight detected. Apply foliar fungicide and use resistant varieties.",
    "Maize_Common_Rust":           "Common rust detected. Apply fungicide early and use resistant maize varieties.",
    "Maize_Gray_Leaf_Spot":        "Gray leaf spot detected. Improve air circulation and apply appropriate fungicide.",
    "Maize_Healthy":               "Your maize plant appears healthy. Continue good crop management practices.",
    "Orange_Black_Spot":           "Black spot detected. Apply copper-based fungicide and remove fallen leaves.",
    "Orange_Canker":               "Citrus canker detected. Remove infected material and apply copper bactericide.",
    "Orange_Greening":             "Citrus greening (HLB) detected. This is serious — remove infected trees and control psyllid vectors.",
    "Orange_Healthy":              "Your orange plant appears healthy. Maintain regular pruning and fertilisation.",
    "Orange_Scab":                 "Citrus scab detected. Apply fungicide at bud break and after rain events.",
    "Pepper_Bacterial_Spot":       "Bacterial spot detected. Apply copper-based bactericide and avoid overhead irrigation.",
    "Pepper_Healthy":              "Your pepper plant appears healthy. Continue regular monitoring and good practices.",
    "Potato_Early_Blight":         "Early blight detected. Apply fungicide and remove infected plant debris.",
    "Potato_Healthy":              "Your potato plant appears healthy. Monitor regularly for signs of disease.",
    "Potato_Late_Blight":          "Late blight detected. This spreads rapidly — apply fungicide immediately and remove infected tissue.",
    "Tomato_Bacterial_Spot":       "Bacterial spot detected. Use copper-based bactericide and disease-free seed.",
    "Tomato_Early_Blight":         "Early blight detected. Remove lower infected leaves and apply fungicide.",
    "Tomato_Healthy":              "Your tomato plant appears healthy. Continue good watering and fertilisation practices.",
    "Tomato_Late_Blight":          "Late blight detected. Apply fungicide immediately — this disease spreads very rapidly.",
    "Tomato_Leaf_Mold":            "Leaf mold detected. Improve ventilation and apply appropriate fungicide.",
    "Tomato_Mosaic_Virus":         "Mosaic virus detected. Remove infected plants and control aphid vectors.",
    "Tomato_Septoria_Leaf_Spot":   "Septoria leaf spot detected. Remove infected leaves and apply fungicide.",
    "Tomato_Spider_Mites":         "Spider mite infestation detected. Apply miticide or neem oil spray.",
    "Tomato_Target_Spot":          "Target spot detected. Apply fungicide and improve air circulation.",
    "Tomato_Yellow_Leaf_Curl_Virus":"Yellow leaf curl virus detected. Remove infected plants and control whitefly vectors.",
}

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

print(f"Model loaded — {len(class_indices)} classes")
print(f"Input shape: {input_details[0]['shape']}")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)  # [0, 255] — EfficientNet handles normalisation
    arr = np.expand_dims(arr, axis=0)
    return arr


def run_inference(image_array):
    input_scale, input_zero_point = input_details[0].get("quantization", (1.0, 0))
    if input_details[0]["dtype"] == np.uint8:
        image_array = (image_array / input_scale + input_zero_point).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    output_scale, output_zero_point = output_details[0].get("quantization", (1.0, 0))
    if output_details[0]["dtype"] == np.uint8:
        output = (output.astype(np.float32) - output_zero_point) * output_scale

    return output[0]


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a JPG, PNG, or BMP image"}), 400

    image_bytes = file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Maximum size is 10MB"}), 400

    try:
        image_array  = preprocess_image(image_bytes)
        probabilities = run_inference(image_array)

        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [
            {
                "class":      idx_to_class[int(i)].replace("_", " "),
                "confidence": float(round(probabilities[i] * 100, 2))
            }
            for i in top3_indices
        ]

        predicted_class = idx_to_class[int(top3_indices[0])]
        advice = DISEASE_ADVICE.get(predicted_class, "Consult your local agricultural extension officer.")

        return jsonify({
            "prediction":  predicted_class.replace("_", " "),
            "confidence":  top3[0]["confidence"],
            "top3":        top3,
            "advice":      advice,
            "is_healthy":  "Healthy" in predicted_class
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
