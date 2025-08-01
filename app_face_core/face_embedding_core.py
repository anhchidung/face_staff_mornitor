import os
from pathlib import Path

import cv2
import numpy as np
from app_face_core.arcface_ncnn import ArcFaceFeature
from app_face_core.scrfd import SCRFD


current_path = Path(__file__).resolve().parent
model_file_path = os.path.join(current_path, "weights", "scrfd", "scrfd_10g_gnkps_bk.onnx")

face_detector = SCRFD(model_file=model_file_path)

if face_detector is not None:
    face_detector.prepare(-1)

param_path = os.path.join(current_path, "weights", "arcface", "r50_opt.param")
bin_path = os.path.join(current_path, "weights", "arcface", "r50_opt.bin")
face_extractor = ArcFaceFeature(param_path=param_path, bin_path=bin_path)


def face_detect_embedding(image):
    embedding = []
    bboxes, kpss = face_detector.detect(image, 0.5, input_size=(640, 640))
    face_aligned = None
    if len(bboxes):
        face_aligned = face_extractor.alignment(image, bboxes, kpss)
        embedding = face_extractor.extract(face_aligned)
    return bboxes[0],kpss[0],  embedding, face_aligned

