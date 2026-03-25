import os
import gc
import time
import traceback
from io import BytesIO

import psutil
from flask import Flask, request, jsonify, send_file
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from waitress import serve

app = Flask(__name__)

# 업로드 최대 크기 제한 (5MB)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# 너무 큰 이미지 방지
Image.MAX_IMAGE_PIXELS = 20_000_000

process = psutil.Process(os.getpid())


def mem_mb():
    return round(process.memory_info().rss / 1024 / 1024, 2)


def log(msg):
    print(f"[MEM {mem_mb()} MB] {msg}", flush=True)


log("=== object_detector.py start ===")
log(f"PORT env = {os.environ.get('PORT')}")
log("before model load")

model = None
try:
    model = YOLO("best.pt")
    log("after model load success")
except Exception as e:
    log(f"model load failed: {e}")
    traceback.print_exc()
    raise


@app.route("/", methods=["GET"])
def home():
    log("GET /")
    return send_file("index.html")


@app.route("/health", methods=["GET"])
def health():
    log("GET /health")
    return jsonify({
        "status": "ok",
        "memory_mb": mem_mb()
    }), 200


@app.route("/detect", methods=["POST"])
def detect():
    start_time = time.time()
    log("POST /detect start")

    if "image_file" not in request.files:
        log("image_file not found in request.files")
        return jsonify({"error": "image_file not found"}), 400

    file = request.files["image_file"]

    if not file or file.filename == "":
        log("empty filename")
        return jsonify({"error": "empty filename"}), 400

    image = None
    results = None
    file_bytes = None

    try:
        log("before file.read()")
        file_bytes = file.read()
        log(f"after file.read() - bytes={len(file_bytes) if file_bytes else 0}")

        if not file_bytes:
            return jsonify({"error": "empty file"}), 400

        log("before Image.open()")
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        log(f"after Image.open() - image size={image.size}")

        # 큰 이미지 강제 축소
        log("before thumbnail()")
        image.thumbnail((1024, 1024))
        log(f"after thumbnail() - resized image size={image.size}")

        log("before model.predict()")
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )
        log("after model.predict()")

        result = results[0]
        output = []

        if result.boxes is not None:
            log(f"boxes detected = {len(result.boxes)}")
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
        else:
            log("result.boxes is None")

        elapsed = round(time.time() - start_time, 3)
        log(f"before return jsonify() - elapsed={elapsed}s")

        return jsonify({
            "count": len(output),
            "detections": output,
            "elapsed_sec": elapsed,
            "memory_mb": mem_mb()
        }), 200

    except UnidentifiedImageError:
        log("invalid image file")
        return jsonify({"error": "invalid image file"}), 400

    except Exception as e:
        log(f"exception in /detect: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        log("enter finally")
        try:
            if results is not None:
                del results
            if image is not None:
                del image
            if file_bytes is not None:
                del file_bytes
            gc.collect()
            log("after cleanup in finally")
        except Exception as e:
            log(f"cleanup error: {e}")


@app.errorhandler(413)
def too_large(e):
    log("413 file too large")
    return jsonify({"error": "file too large. max 5MB allowed"}), 413


@app.errorhandler(Exception)
def handle_all_exceptions(e):
    log(f"global exception handler: {e}")
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    log(f"starting waitress on 0.0.0.0:{port}")
    serve(
        app,
        host="0.0.0.0",
        port=port,
        threads=1
    )
