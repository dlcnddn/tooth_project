import os
import gc
from io import BytesIO

from flask import Flask, request, jsonify, send_file
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from waitress import serve

app = Flask(__name__)

# 업로드 파일 크기 제한: 5MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# 너무 큰 이미지로 인한 폭주 방지
Image.MAX_IMAGE_PIXELS = 20_000_000

# 서버 시작 시 모델 1회만 로드
model = YOLO("best.pt")


@app.route("/", methods=["GET"])
def home():
    return send_file("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/detect", methods=["POST"])
def detect():
    if "image_file" not in request.files:
        return jsonify({"error": "image_file not found"}), 400

    file = request.files["image_file"]

    if not file or file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    image = None
    results = None

    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "empty file"}), 400

        image = Image.open(BytesIO(file_bytes)).convert("RGB")

        # 큰 이미지는 축소해서 RAM 사용량 줄임
        # 원본이 4000x3000이어도 최대 1024x1024 안으로 줄어듦
        image.thumbnail((1024, 1024))

        # 추론
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        result = results[0]
        output = []

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
                class_id = int(box.cls[0].item())
                prob = round(float(box.conf[0].item()), 4)
                class_name = result.names[class_id]

                output.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "class_name": class_name,
                    "probability": prob
                })

        return jsonify({
            "count": len(output),
            "detections": output
        }), 200

    except UnidentifiedImageError:
        return jsonify({"error": "invalid image file"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if results is not None:
                del results
            if image is not None:
                del image
            gc.collect()
        except Exception:
            pass


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "file too large. max 5MB allowed"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    serve(
        app,
        host="0.0.0.0",
        port=port,
        threads=1
    )
