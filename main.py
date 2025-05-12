from services.ocr.paddle_ocr_v4 import PaddleOCRv4Service

from modules.ocr.paddle_ocr_v4.detector import PaddleOCRv4TextDetector
from modules.ocr.paddle_ocr_v4.recognizer import PaddleOCRv4TextRecognizer

from modules.ocr.paddle_ocr_v4.utils import DET_MODEL_FILE, DET_PARAMS_FILE, REC_MODEL_FILE, CHAR_DICT_PATH, REC_PARAMS_FILE

from fastapi import FastAPI

