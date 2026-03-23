import os
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from PIL import Image
from ultralytics import YOLO
from waitress import serve

app = Flask(__name__)

# 앱 시작 시 모델 1회만 로드
model = YOLO("best.pt")


@app.route("/", methods=["GET"])
def home():
    return send_file("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image_file" not in request.files:
        return jsonify({"error": "image_file not found"}), 400

    file = request.files["image_file"]

    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        image = Image.open(BytesIO(file.read())).convert("RGB")
        results = model.predict(image)
        result = results[0]

        output = []

        for box in result.boxes:
            x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item())
            prob = round(float(box.conf[0].item()), 2)
            class_name = result.names[class_id]

            output.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_name": class_name,
                "probability": prob
            })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    serve(app, host="0.0.0.0", port=port)
