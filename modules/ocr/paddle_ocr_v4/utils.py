import os
import random

import numpy as np
import paddle.inference as paddle_infer

random_name_choices = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
def random_name(length=8):
    """生成随机文件名"""
    return ''.join(random.choice(random_name_choices) for _ in range(length))

BASE_DIR = "/app"

# --- 1. 配置区域 ---
# 检测模型配置
DET_MODEL_DIR = os.path.join(BASE_DIR, "inference", "ch_PP-OCRv4_det_server_infer")
DET_MODEL_FILE = os.path.join(DET_MODEL_DIR, "inference.pdmodel")
DET_PARAMS_FILE = os.path.join(DET_MODEL_DIR, "inference.pdiparams")

# 识别模型配置

REC_MODEL_DIR = os.path.join(BASE_DIR, "inference", "ch_PP-OCRv4_rec_server_infer") 
REC_MODEL_FILE = os.path.join(REC_MODEL_DIR, "inference.pdmodel")
REC_PARAMS_FILE = os.path.join(REC_MODEL_DIR, "inference.pdiparams")
CHAR_DICT_PATH = os.path.join(REC_MODEL_DIR, os.path.join(BASE_DIR, "inference", "ppocr_keys_v1.txt"))

REC_IMAGE_SHAPE_H = 48

# 检测模型参数 (PP-OCRv4 det)
DET_LIMIT_SIDE_LEN = 960  # 图片短边resize的目标大小
DET_LIMIT_MAX_LEN = 2048 # 在某些情况下，限制长边的最大值 (PP-OCRv4 det可能不严格需要这个，但可以保留)
DET_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DET_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# DBNet后处理参数
DB_THRESH = 0.3
DB_BOX_THRESH = 0.6 # 对于DB，这个值通常较高，如0.6-0.7
DB_UNCLIP_RATIO = 1.5 # 决定文本框放大多少
DB_MAX_CANDIDATES = 1000

COMPLEX_IMAGE_PATH = "./demo_text_ocr.jpg"

# --- 2. 辅助函数 ---
# load_predictor, load_character_list, CTCDecode, preprocess_rec_image, get_rotate_crop_image 保持不变 (从您之前的代码)

def load_predictor(model_file, params_file, use_gpu=False, device_id=0, use_trt=False, use_mkldnn=True):
    """加载 PaddlePaddle 推理模型 (与之前相同)"""
    config = paddle_infer.Config(model_file, params_file)
    if use_gpu:
        config.enable_use_gpu(100, device_id)
        if use_trt:
            config.enable_tensorrt_engine(
                workspace_size=1 << 30, max_batch_size=1, min_subgraph_size=5,
                precision_mode=paddle_infer.PrecisionType.Float32,
                use_static=False, use_calib_mode=False)
    else:
        config.disable_gpu()
        if use_mkldnn:
            config.enable_mkldnn()
            try: # PaddlePaddle 2.5+
                config.set_cpu_math_library_num_threads(max(1, os.cpu_count() // 2))
            except Exception: # Older versions
                pass
    config.switch_ir_optim(True)
    config.enable_memory_optim()
    config.switch_use_feed_fetch_ops(False)
    predictor = paddle_infer.create_predictor(config)
    return predictor

def load_character_list(dict_path):
    """加载字符字典 (与之前相同)"""
    char_list = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            char_list.append(line.strip())
    return char_list

class CTCDecode:
    """简易CTC解码器 (与之前相同)"""
    def __init__(self, character_list):
        self.character = ['blank'] + character_list
    def __call__(self, preds_idx):
        text = []
        last_char_idx = 0
        for char_idx in preds_idx:
            if char_idx != 0 and char_idx != last_char_idx:
                if char_idx > 0 and char_idx < len(self.character):
                    text.append(self.character[char_idx])
            last_char_idx = char_idx
        return "".join(text)