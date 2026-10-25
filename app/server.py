#!/usr/bin/env python

import flask
from flask_compress import Compress
from flask import abort, make_response, request, jsonify
from os import path, environ
import cv2
import numpy as np
import requests
import backoff
import base64
from requests.exceptions import HTTPError

from auth import token_required
from lib.detection_model import load_net, detect

THRESH = 0.2  # The threshold for a box to be considered a positive detection
SESSION_TTL_SECONDS = 60*2

app = flask.Flask(__name__)
Compress(app)

status = dict()

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = environ.get('DEBUG') == 'True'

model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main = load_net(path.join(model_dir, 'model.cfg'), path.join(model_dir, 'model.meta'))


def draw_bounding_boxes(image, detections):
    for detection in detections:
        label, confidence, bbox = detection
        x, y, w, h = [int(v) for v in bbox]
        color = (0, 0, 255)  # Red color for bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 5)
        text = f"{label}: {confidence:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    return image


def _should_retry(exc):
    """Retry on transient network errors and HTTP 5xx server errors."""
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500
    return False


@backoff.on_exception(
    backoff.expo,
    (HTTPError, requests.exceptions.Timeout, requests.exceptions.ConnectionError),
    max_tries=3,
    giveup=lambda exc: not _should_retry(exc)
)
def _get_with_retry(url, timeout, stream=True):
    """Make HTTP GET request with retry logic for transient errors."""
    resp = requests.get(url, stream=stream, timeout=timeout)
    resp.raise_for_status()
    return resp


@app.route('/p/', methods=['GET'])
@token_required
def get_p():
    if 'img' in request.args:
        try:
            # Use longer timeout for Google Cloud Storage as it's slower
            if 'storage.googleapis.com' in request.args['img']:
                timeout = (10, 30)  # 10s connection, 30s read
            else:
                timeout = (0.1, 5)  # 0.1s connection, 5s read

            resp = _get_with_retry(request.args['img'], timeout, stream=True)
            img_array = np.array(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            detections = detect(net_main, img, thresh=THRESH)
            return jsonify({'detections': detections})
        except Exception as err:
            app.logger.error(f"Failed to get image {request.args} - {err}")
            abort(
                make_response(
                    jsonify(
                        detections=[],
                        message=f"Failed to get image {request.args} - {err}",
                    ),
                    400,
                )
            )
    else:
        app.logger.warn(f"Invalid request params: {request.args}")
        abort(
            make_response(
                jsonify(
                    detections=[], message=f"Invalid request params: {request.args}"
                ),
                422,
            )
        )


@app.route('/hc/', methods=['GET'])
def health_check():
    return 'ok' if net_main is not None else 'error'

@app.route('/', methods=['GET'])
def index():
    return 'Hello from the Obico ML Addon!'

@app.route('/detect/', methods=['POST'])
@token_required
def failure_detect():
    data = request.get_json()

    img_base64 = data.get("img", None)
    if img_base64 is None:
        return jsonify({"error": "No image provided"}), 400

    try:
        img_bytes = base64.b64decode(img_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        threshold = float(data.get("threshold", THRESH))

        detections = detect(net_main, img, thresh=threshold)

        img_with_boxes = draw_bounding_boxes(img, detections)

        _, buffer = cv2.imencode('.jpg', img_with_boxes)
        img_with_boxes_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "detections": detections,
            "image_with_detections": img_with_boxes_base64
        }), 200

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Failed to process image - {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3333, threaded=False)
