import ray.serve
from modules.ocr.paddle_ocr_v4.utils import *

import math
import numpy as np
import cv2
import paddle.inference as paddle_infer
import ray
import os




@ray.serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1, "num_gpus": 1})
class PaddleOCRv4TextDetector:
    def __init__(self, model_file, params_file, use_gpu=False, device_id=0):
        self.predictor = load_predictor(model_file, params_file, use_gpu=use_gpu, device_id=device_id)
    
    def resize_norm_img_det(self, img, limit_side_len, limit_max_len, mean, std):
        """
        对图像进行resize和归一化，用于DBNet检测模型。
        返回:
            norm_img_padded_chw: 归一化和padding后的图像 (CHW, for model input)
            original_h, original_w: 原始图像高宽
            resized_h, resized_w: 图像等比例缩放后的高宽 (在padding前的有效内容高宽)
            pad_h, pad_w: padding后的总高宽 (模型实际输入的高宽)
        """
        original_h, original_w = img.shape[:2]

        # 1. 等比例缩放图像，使短边等于 limit_side_len
        if original_h < original_w:
            resized_h = limit_side_len
            resized_w = int(round(original_w * resized_h / original_h))
        else:
            resized_w = limit_side_len
            resized_h = int(round(original_h * resized_w / original_w))

        # 限制长边不超过 limit_max_len (如果启用了)
        if limit_max_len > 0: # limit_max_len=0 或负数表示不限制
            if resized_h > limit_max_len:
                ratio = limit_max_len / resized_h
                resized_h = limit_max_len
                resized_w = int(round(resized_w * ratio))
            elif resized_w > limit_max_len:
                ratio = limit_max_len / resized_w
                resized_w = limit_max_len
                resized_h = int(round(resized_h * ratio))
        
        # 如果resize后任何一边为0，则设为1 (防止除零错误)
        if resized_h == 0: resized_h = 1
        if resized_w == 0: resized_w = 1

        resized_img = cv2.resize(img, (resized_w, resized_h))

        # 2. Padding 到32的倍数
        pad_h = math.ceil(resized_h / 32.0) * 32
        pad_w = math.ceil(resized_w / 32.0) * 32

        padded_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        padded_img[0:resized_h, 0:resized_w, :] = resized_img # 将内容放在左上角

        # 3. 归一化
        padded_img = padded_img / 255.0
        norm_img_padded = (padded_img - mean) / std # mean 和 std 应该是 (3,) 或 (1,1,3)

        # 4. HWC to CHW
        norm_img_padded_chw = norm_img_padded.transpose((2, 0, 1))
        norm_img_padded_chw = np.expand_dims(norm_img_padded_chw, axis=0) # Add batch dimension

        return norm_img_padded_chw, original_h, original_w, resized_h, resized_w, pad_h, pad_w

    def postprocess_det_db(self, det_output_map, original_h, original_w,
                       resized_h, resized_w, pad_h, pad_w, # 从预处理中获取的尺寸信息
                       db_thresh=0.3, db_box_thresh=0.6, db_unclip_ratio=1.5):
        """
        DBNet检测结果后处理。
        det_output_map: 模型输出的概率图谱 (N, 1, H_pad, W_pad)
        original_h, original_w: 原始图像高宽
        resized_h, resized_w: 图像在padding板上的有效内容高宽
        pad_h, pad_w: padding后的总高宽 (即det_output_map的高宽)
        """
        # 确保 Pyclipper 和 Shapely 可用
        try:
            import pyclipper
            from shapely.geometry import Polygon
        except ImportError:
            print("警告: Pyclipper 或 Shapely 未安装，DB unclip 功能将受限，检测框可能偏小。")
            print("请运行: pip install pyclipper shapely")
            # 定义一个虚拟的 unclip 函数，如果不安装这两个库
            def unclip(box, unclip_ratio): return box 

        # 1. 获取概率图并裁剪到有效内容区域
        # det_output_map 的 H, W 对应 pad_h, pad_w
        # 我们只关心 :resized_h, :resized_w 这部分有效内容
        segmentation_map_padded = det_output_map[0, 0, :, :] # (H_pad, W_pad)
        segmentation_map_content = segmentation_map_padded[:resized_h, :resized_w]

        # 2. 二值化
        binary_map = (segmentation_map_content > db_thresh).astype(np.uint8)

        # 3. 寻找轮廓 (在二值化的、仅包含有效内容的分割图上)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        dt_boxes_original_coords = []
        for contour in contours:
            if len(contour) < 3: # 至少需要3个点形成轮廓
                continue

            # a. 获取最小外接四边形 (点集)
            #   使用 minAreaRect 然后 boxPoints 来获得四边形的四个角点
            #   这比 approxPolyDP 更适合不规则的文本区域
            rect = cv2.minAreaRect(contour)
            box_points_on_content_map = cv2.boxPoints(rect) # (4, 2)

            # b. 计算box得分 (可选，但推荐)
            #   为了简化，我们这里跳过精确的box_score_fast，而使用box的面积和周长
            #   实际应用中，应该计算box内概率图的均值
            area = cv2.contourArea(box_points_on_content_map)
            if area < 10: # 过滤掉面积过小的噪声
                continue
            
            # c. Unclip (放大) box，在 content_map 坐标系下进行
            #    Pyclipper 需要整数坐标
            try:
                poly = Polygon(box_points_on_content_map)
                distance = poly.area * db_unclip_ratio / poly.length
                offset = pyclipper.PyclipperOffset()
                offset.AddPath(box_points_on_content_map.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                expanded_box_on_content_map_list = offset.Execute(distance)
                
                if not expanded_box_on_content_map_list or len(expanded_box_on_content_map_list[0]) < 3 :
                    # print(f"Warning: Unclip failed or produced too few points for contour. Original box: {box_points_on_content_map.tolist()}")
                    unclipped_box_on_content_map = box_points_on_content_map # Fallback to original box
                else:
                    # Pyclipper可能返回多个多边形，通常取最大的一个，或者第一个
                    # 并且我们需要确保它仍然是近似四边形
                    # 为了简化，我们取第一个，并用minAreaRect再次规范化它为四边形
                    expanded_poly_points = np.array(expanded_box_on_content_map_list[0])
                    if len(expanded_poly_points) >=3:
                        expanded_rect = cv2.minAreaRect(expanded_poly_points)
                        unclipped_box_on_content_map = cv2.boxPoints(expanded_rect)
                    else:
                        unclipped_box_on_content_map = box_points_on_content_map # Fallback

            except Exception as e:
                # print(f"Pyclipper unclip error: {e}. Using original box.")
                unclipped_box_on_content_map = box_points_on_content_map


            if len(unclipped_box_on_content_map) != 4:
                # print(f"Warning: Box after unclip does not have 4 points: {unclipped_box_on_content_map}")
                continue # 或者使用原始box


            # d. 将 unclipped_box_on_content_map 的坐标还原到原始图像坐标系
            #    当前坐标是相对于 resized_h, resized_w 的
            #    缩放比例: original / resized
            scale_h = original_h / resized_h
            scale_w = original_w / resized_w

            box_original_coords = np.copy(unclipped_box_on_content_map).astype(np.float32)
            box_original_coords[:, 0] *= scale_w
            box_original_coords[:, 1] *= scale_h
            
            # 确保点是 tl, tr, br, bl 顺序 (PaddleOCR DBPostProcess的顺序)
            # cv2.boxPoints 返回的顺序是变化的，我们需要排序
            sorted_box = np.zeros((4, 2), dtype=np.float32)
            s = box_original_coords.sum(axis=1)
            sorted_box[0] = box_original_coords[np.argmin(s)] # Top-left
            sorted_box[2] = box_original_coords[np.argmax(s)] # Bottom-right
            
            diff = np.diff(box_original_coords, axis=1)
            sorted_box[1] = box_original_coords[np.argmin(diff)] # Top-right
            sorted_box[3] = box_original_coords[np.argmax(diff)] # Bottom-left
            
            # 限制box坐标在图像范围内
            sorted_box[:,0] = np.clip(sorted_box[:,0], 0, original_w -1)
            sorted_box[:,1] = np.clip(sorted_box[:,1], 0, original_h -1)

            dt_boxes_original_coords.append(sorted_box.tolist())

        return dt_boxes_original_coords

    def predict(self, predict_method = "default", **inputs):
        original_cv2_image = inputs.get("image_ref") # 获取原始图像
        # 预处理
        preprocessed_det_img_chw, orig_h, orig_w, res_h, res_w, pad_h, pad_w = self.resize_norm_img_det(original_cv2_image, inputs.get("limit_side_len"), inputs.get("limit_max_len"), inputs.get("mean"), inputs.get("std"))
        
        if preprocessed_det_img_chw is None:
            print("文本检测预处理失败。")
            return []
        
        # 推理
        det_input_names = self.predictor.get_input_names()
        det_input_tensor = self.predictor.get_input_handle(det_input_names[0])
        det_input_tensor.reshape(preprocessed_det_img_chw.shape)
        det_input_tensor.copy_from_cpu(preprocessed_det_img_chw)
        self.predictor.run()
        det_output_names = self.predictor.get_output_names()
        det_output_tensor = self.predictor.get_output_handle(det_output_names[0])
        det_output_map = det_output_tensor.copy_to_cpu() # (N, 1, H_pad, W_pad)
        
        # 后处理获取文本框 (在原始图像坐标系下)
        dt_boxes = self.postprocess_det_db(det_output_map, orig_h, orig_w, res_h, res_w, pad_h, pad_w,
                                    db_thresh=DB_THRESH, db_box_thresh=DB_BOX_THRESH,
                                    db_unclip_ratio=DB_UNCLIP_RATIO)

        return dt_boxes