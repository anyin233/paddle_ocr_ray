import cv2
import numpy as np
import ray

from modules.ocr.paddle_ocr_v4.utils import *


# @ray.serve.deployment(num_replicas = 2, ray_actor_options={"num_cpus": 1, "num_gpus": 1})
class PaddleOCRv4TextRecognizer:
    def __init__(self, model_file, char_dict_file, params_file, use_gpu=False, device_id=0):
        self.predictor = load_predictor(model_file, params_file, use_gpu=use_gpu, device_id=device_id)
        self.char_list = load_character_list(char_dict_file)
        self.ctc_decoder = CTCDecode(self.char_list)

    def get_rotate_crop_image(self, img, points_list_of_lists): # 为了清晰，重命名了输入参数
        """根据检测到的四个点points裁剪并透视变换图像"""
        assert len(points_list_of_lists) == 4, "points number must be 4"

        # 在函数开始时就将列表的列表转换为NumPy数组
        points = np.array(points_list_of_lists, dtype=np.float32)

        # 现在 points[0], points[1] 等是NumPy数组的行，可以直接进行向量减法
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                                    np.linalg.norm(points[1] - points[2])))
        
        if img_crop_width == 0 or img_crop_height == 0:
            # print(f"警告: get_rotate_crop_image 中出现零尺寸。宽度: {img_crop_width}, 高度: {img_crop_height}. 点: {points_list_of_lists}")
            if img_crop_width == 0: img_crop_width = 10 # 设置一个最小尺寸以避免后续错误
            if img_crop_height == 0: img_crop_height = 10

        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                            [img_crop_width, img_crop_height], [0, img_crop_height]])
        
        # 'points' 已经是我们需要的 float32 NumPy 数组
        M = cv2.getPerspectiveTransform(points, pts_std) 

        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                    borderMode=cv2.BORDER_REPLICATE, # 使用 BORDER_REPLICATE 减少黑色区域
                                    flags=cv2.INTER_CUBIC)
        
        # 如果裁剪后图像高度远大于宽度，可能需要旋转90度（文本是垂直的）
        if dst_img.shape[0] > dst_img.shape[1] * 1.5: # 简单判断是否为垂直文本
            dst_img = np.rot90(dst_img)
        return dst_img 
    
    def preprocess_rec_image(self, img_cv2, rec_image_shape_h): # Renamed img to img_cv2
        """识别模型的图像预处理 (与之前类似，但输入是cv2 image array)"""
        h, w = img_cv2.shape[:2]
        if h == 0 or w == 0: return None # 空图像
        ratio = w / float(h)
        resized_w = int(rec_image_shape_h * ratio)
        if resized_w <= 0: resized_w = 1 # 最小宽度

        resized_image = cv2.resize(img_cv2, (resized_w, rec_image_shape_h)) # use img_cv2
        resized_image = resized_image.astype(np.float32)
        resized_image = resized_image / 255.0
        resized_image = (resized_image - 0.5) / 0.5
        if len(resized_image.shape) == 2: #灰度图转三通道
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        resized_image = resized_image.transpose((2, 0, 1))
        input_image = np.expand_dims(resized_image, axis=0)
        return input_image
    
    def predict(self, predict_method="default", **inputs):
        
        original_cv2_image = inputs.get('image_ref') # 获取原始图像
        # --- 阶段2: 文本识别 ---
        results = []

        dt_boxes = inputs.get('dt_boxes') # 获取检测到的文本框  
        if not dt_boxes:
            print("未检测到任何文本框。")
            return []

        # 按从上到下，从左到右的顺序对box排序 (可选，但有助于结果的有序性)
        # dt_boxes = sorted(dt_boxes, key=lambda box: (box[0][1], box[0][0])) # 按左上角y, 再按x排序

        for i, box_coords_list in enumerate(dt_boxes): # box_coords_list 是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if not box_coords_list or len(box_coords_list) != 4:
                # print(f"跳过无效的box: {box_coords_list}")
                continue
            
            # print(f"处理第 {i+1}/{len(dt_boxes)} 个检测框: {box_coords_list}")
            
            try:
                # 裁剪并校正图像区域 (使用原始图像和原始坐标系的box)
                cropped_img = self.get_rotate_crop_image(original_cv2_image, box_coords_list)

                # 预处理识别图像
                preprocessed_rec_img = self.preprocess_rec_image(cropped_img, inputs.get('rec_img_h'))
                
                if cropped_img is None or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                    # print("裁剪图像为空，跳过。")
                    continue
                if preprocessed_rec_img is None:
                    # print("识别图像预处理失败，跳过。")
                    continue
                
                # 识别
                rec_input_names = self.predictor.get_input_names()
                rec_input_tensor = self.predictor.get_input_handle(rec_input_names[0])
                rec_input_tensor.reshape(preprocessed_rec_img.shape)
                rec_input_tensor.copy_from_cpu(preprocessed_rec_img)
                self.predictor.run()
                rec_output_names = self.predictor.get_output_names()
                rec_output_tensor = self.predictor.get_output_handle(rec_output_names[0])
                rec_preds = rec_output_tensor.copy_to_cpu()

                preds_idx = np.argmax(rec_preds[0], axis=1)
                text = self.ctc_decoder(preds_idx)
                
                # 将box坐标转换为整数以便记录
                cleaned_box_int = [[int(p[0]), int(p[1])] for p in box_coords_list]
                results.append({'box': cleaned_box_int, 'text': text, 'confidence': 0.0}) # confidence未计算

            except Exception as e:
                print(f"处理box {box_coords_list} 时发生错误: {e}")
                import traceback
                traceback.print_exc()
            
        return results