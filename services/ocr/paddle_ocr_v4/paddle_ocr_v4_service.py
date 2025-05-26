import cv2  # OpenCV for image manipulation
import numpy as np
import ray
import ray.serve
from fastapi import FastAPI, File, UploadFile

from modules.ocr.paddle_ocr_v4 import PaddleOCRv4TextDetector, PaddleOCRv4TextRecognizer

app = FastAPI()

from modules.ocr.paddle_ocr_v4.utils import *


@ray.serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@ray.serve.ingress(app)
class PaddleOCRv4Service:
    def __init__(self, detector, recognizer):
        self.detector = detector
        self.recognizer = recognizer

    @app.post("/ocr")
    async def ocr_endpoint(self, image: UploadFile = File(...)):

        contents = image.file.read()
        img_byte_arr = np.fromstring(contents, np.uint8)
        original_cv2_image = cv2.imdecode(img_byte_arr, cv2.IMREAD_COLOR)
        image_ref = ray.put(original_cv2_image)
        
        detector_input = {
            "image_ref": image_ref,
            "limit_side_len": DET_LIMIT_SIDE_LEN,
            "limit_max_len": DET_LIMIT_MAX_LEN,
            "mean": DET_NORM_MEAN,
            "std": DET_NORM_STD
        }
        
        dt_boxes = self.detector.predict.remote(**detector_input)
        
        recognizer_input = {
            "image_ref": image_ref,
            "dt_boxes": dt_boxes,
            "rec_img_h": REC_IMAGE_SHAPE_H
        }
        
        results = self.recognizer.predict.remote(**recognizer_input)
        
        return await results

ocr_detector_cls = ray.serve.deployment(
    PaddleOCRv4TextDetector,
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.5}
)
ocr_recognizer_cls = ray.serve.deployment(
    PaddleOCRv4TextRecognizer,
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.5}
)

detector = ocr_detector_cls.bind(DET_MODEL_FILE, DET_PARAMS_FILE, use_gpu=True)
recoginzer = ocr_recognizer_cls.bind(REC_MODEL_FILE, CHAR_DICT_PATH, REC_PARAMS_FILE, use_gpu=True)
ocr_app = PaddleOCRv4Service.bind(detector, recoginzer)