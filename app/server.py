#!/usr/bin/env python

import flask
from flask_compress import Compress
from flask import abort, make_response, request, send_file, jsonify
from os import path, environ
import cv2
import json
import numpy as np
import requests
import io
import logging

from lib.detection_model import load_net, detect

THRESH = 0.2  # The threshold for a box to be considered a positive detection
NMS_THRESHOLD = 0.8  # Non-max suppression IoU threshold
SESSION_TTL_SECONDS = 60*2

app = flask.Flask(__name__)
Compress(app)

status = dict()

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = True

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main = load_net(path.join(model_dir, 'model.cfg'), path.join(model_dir, 'model.meta'))

def non_max_suppression(boxes, scores, threshold):
    if len(boxes) == 0:
        return []
    
    # Convert boxes to the format [x1, y1, x2, y2]
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep

def draw_bounding_boxes(image, detections):
    if not detections:
        logger.warning("No detections to draw")
        return image

    logger.debug(f"Detections: {detections}")

    try:
        boxes = np.array([detection[2] for detection in detections])
        scores = np.array([detection[1] for detection in detections])
        
        logger.debug(f"Boxes shape: {boxes.shape}, Scores shape: {scores.shape}")
        
        # Apply non-max suppression
        keep = non_max_suppression(boxes, scores, NMS_THRESHOLD)
        
        for idx in keep:
            label, confidence, bbox = detections[idx]
            x, y, w, h = [int(v) for v in bbox]
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
        logger.error(f"Error in draw_bounding_boxes: {e}")
        
    return image

@app.route('/detect/', methods=['GET'])
def get_detections():
    if 'img' in request.args:
        try:
            resp = requests.get(request.args['img'], stream=True, timeout=(5.0, 5))
            resp.raise_for_status()
            img_array = np.array(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            
            threshold = float(request.args.get('threshold', THRESH))
            detections = detect(net_main, img, thresh=threshold)
            
            # Draw bounding boxes on the image
            img_with_boxes = draw_bounding_boxes(img, detections)
            
            # Convert the image to bytes
            is_success, buffer = cv2.imencode(".jpg", img_with_boxes)
            if not is_success:
                raise Exception("Failed to encode image")
            
            byte_io = io.BytesIO(buffer)
            byte_io.seek(0)
            
            return send_file(byte_io, mimetype='image/jpeg')
        
        except Exception as err:
            app.logger.error(f"Failed to process image {request.args} - {err}")
            abort(
                make_response(
                    f"Failed to process image - {err}",
                    400,
                )
            )
    else:
        app.logger.warn(f"Invalid request params: {request.args}")
        abort(
            make_response(
                f"Invalid request params: {request.args}",
                422,
            )
        )

@app.route('/hc/', methods=['GET'])
def health_check():
    return 'ok' if net_main is not None else 'error'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3333, threaded=False)
