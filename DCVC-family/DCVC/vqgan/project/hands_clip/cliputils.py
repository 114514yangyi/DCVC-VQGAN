import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import warnings

# 抑制 MediaPipe 的内部警告（可选）
# 这些警告通常不影响功能，但可能会产生大量日志
# 关于 "NORM_RECT without IMAGE_DIMENSIONS" 警告：
# 这是 MediaPipe 内部的警告，MediaPipe Image 对象已经包含了图像尺寸信息
# 这个警告通常不影响功能，但可以通过设置日志级别来抑制
# 如果需要看到这些警告，可以注释掉下面的代码或设置为 '1'（只显示 WARNING 及以上）
if 'GLOG_minloglevel' not in os.environ:
    os.environ['GLOG_minloglevel'] = '2'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Tuple, Optional, Dict

class HandCropper:
    def __init__(self, model_path: Optional[str] = None, device_type: str = 'cpu'):
        """
        初始化 MediaPipe 0.10+ 手部检测器
        :param model_path: .task 模型文件的路径，如果为 None 则使用默认模型（需要下载）
        :param device_type: 'cpu' 或 'gpu' (取决于 mediapipe 编译版本)
        :raises: 如果初始化失败，直接抛出异常中断程序
        """
        # 根据系统环境选择硬件加速
        delegate = python.BaseOptions.Delegate.CPU if device_type == 'cpu' else python.BaseOptions.Delegate.GPU
        
        # 创建 BaseOptions
        base_options = python.BaseOptions(
            model_asset_path="hands_clip/hand_landmarker.task",
            delegate=delegate
        )
        
        # 创建 HandLandmarkerOptions
        self.options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 如果初始化失败，直接抛出异常
        self.detector = vision.HandLandmarker.create_from_options(self.options)
        self.available = True

    def detect_hands_in_image(self, image_np: np.ndarray) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
        """
        检测单张图片中的手部边界
        :param image_np: numpy array, 形状为 (H, W, 3)，RGB 格式，值域 [0, 255]
        """
        if not self.available or self.detector is None:
            raise RuntimeError("HandCropper 检测器未正确初始化，无法进行手部检测")
        
        h, w = image_np.shape[:2]
        
        # 确保图像是 RGB 格式
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        elif image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)
        
        try:
            # MediaPipe 0.10+ 必须转换成 mp.Image
            # 确保图像是连续的 numpy array，格式正确
            if not image_np.flags['C_CONTIGUOUS']:
                image_np = np.ascontiguousarray(image_np)
            
            # 确保图像数据类型是 uint8
            if image_np.dtype != np.uint8:
                image_np = image_np.astype(np.uint8)
            
            # 创建 MediaPipe Image 对象
            # 注意：MediaPipe 会自动从 data 获取图像尺寸 (H, W)
            # 关于 NORM_RECT 警告：这是 MediaPipe 内部的警告，通常不影响功能
            # 它提示在处理归一化矩形时需要图像尺寸，但 MediaPipe Image 已经包含了这些信息
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_np
            )
            
            result = self.detector.detect(mp_image)
            
            left_bbox = None
            right_bbox = None
            
            if result.hand_landmarks:
                for idx, hand_landmarks in enumerate(result.hand_landmarks):
                    # 获取左右手分类标签
                    if result.handedness and len(result.handedness) > idx:
                        label = result.handedness[idx][0].display_name  # 'Left' 或 'Right'
                    else:
                        label = 'Unknown'
                    
                    x_coords = [lm.x for lm in hand_landmarks]
                    y_coords = [lm.y for lm in hand_landmarks]
                    
                    # 计算边界并增加 15% 的边距 (Margin)
                    margin = 0.15
                    x_min = max(0.0, min(x_coords) - margin)
                    y_min = max(0.0, min(y_coords) - margin)
                    x_max = min(1.0, max(x_coords) + margin)
                    y_max = min(1.0, max(y_coords) + margin)
                    
                    bbox = (int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h))
                    
                    if label == 'Left':
                        left_bbox = bbox
                    elif label == 'Right':
                        right_bbox = bbox
            
            return {'left_hand': left_bbox, 'right_hand': right_bbox}
        except Exception as e:
            # 如果检测失败，返回 None
            return {'left_hand': None, 'right_hand': None}

    def get_hand_crops(
        self,
        images: torch.Tensor,
        images_recon: torch.Tensor,
        target_size: Tuple[int, int] = (128, 128)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 Batch 张量中裁剪手部并缩放
        :param images: (B, C, H, W) 归一化后的张量
        :param images_recon: (B, C, H, W) 重建图像张量
        :param target_size: 目标尺寸，默认为 (128, 128)
        :return: (hand_crops_orig, hand_crops_recon) 裁剪并缩放后的手部区域
        """
        B, C, H, W = images.shape
        device = images.device
        
        crops_orig = []
        crops_recon = []

        for i in range(B):
            # 1. 张量转 Numpy (处理归一化并转为 RGB uint8)
            img_tensor = images[i].detach()
            img_min, img_max = img_tensor.min(), img_tensor.max()
            img_norm = (img_tensor - img_min) / (img_max - img_min + 1e-5)
            img_np = (img_norm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # 确保图像是 3 通道 RGB
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            elif img_np.shape[2] > 3:
                img_np = img_np[:, :, :3]
            
            # 2. 检测
            bboxes = self.detect_hands_in_image(img_np)
            l_box, r_box = bboxes['left_hand'], bboxes['right_hand']
            
            # 3. 确定最终裁剪区域 (取并集或中心区域)
            if l_box and r_box:
                x1 = min(l_box[0], r_box[0])
                y1 = min(l_box[1], r_box[1])
                x2 = max(l_box[2], r_box[2])
                y2 = max(l_box[3], r_box[3])
            elif l_box:
                x1, y1, x2, y2 = l_box
            elif r_box:
                x1, y1, x2, y2 = r_box
            else:
                # 未检出：取中心 25% 区域作为 fallback（与原始设计一致）
                size = min(H, W) // 4
                x1, y1 = (W - size) // 2, (H - size) // 2
                x2, y2 = x1 + size, y1 + size

            # 确保 bbox 有效
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W))
            y2 = max(y1 + 1, min(y2, H))

            # 4. 裁剪并插值缩放
            # 使用 Tensor 切片保持梯度/计算在同一设备
            crop_o = images[i:i+1, :, y1:y2, x1:x2]
            crop_r = images_recon[i:i+1, :, y1:y2, x1:x2]
            
            # 缩放到目标尺寸
            crops_orig.append(F.interpolate(crop_o, size=target_size, mode='bilinear', align_corners=False))
            crops_recon.append(F.interpolate(crop_r, size=target_size, mode='bilinear', align_corners=False))

        return torch.cat(crops_orig, dim=0), torch.cat(crops_recon, dim=0)
    
    def draw_hand_bboxes_on_images(
        self,
        images: torch.Tensor,
        b: int,
        d: int
    ) -> torch.Tensor:
        """
        在图像上绘制手部检测边界框（红框）
        :param images: (b*d, c, h, w) 或 (d, c, h, w) 张量，值域 [0, 1]
        :param b: batch size（如果输入是 (d, c, h, w)，则 b=1）
        :param d: sequence length (帧数)
        :return: 绘制了边界框的图像张量，形状与输入相同，值域 [0, 1]
        """
        if not self.available or self.detector is None:
            return images
        
        # 处理输入形状：可能是 (b*d, c, h, w) 或 (d, c, h, w)
        if len(images.shape) == 4:
            if images.shape[0] == d:  # (d, c, h, w)
                images_reshaped = images.unsqueeze(0)  # (1, d, c, h, w)
                b = 1
            else:  # (b*d, c, h, w)
                images_reshaped = images.view(b, d, images.shape[1], images.shape[2], images.shape[3])
        else:
            images_reshaped = images.view(b, d, images.shape[1], images.shape[2], images.shape[3])
        
        images_with_bboxes = []
        
        for batch_idx in range(b):
            batch_images = []
            for frame_idx in range(d):
                # 获取单帧图像
                img_tensor = images_reshaped[batch_idx, frame_idx]  # (c, h, w)
                
                # 转换为 numpy，值域 [0, 1] -> [0, 255]
                img_np_rgb = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # 确保图像是 3 通道 RGB
                if len(img_np_rgb.shape) == 2:  # 灰度图
                    img_np_rgb = cv2.cvtColor(img_np_rgb, cv2.COLOR_GRAY2RGB)
                elif img_np_rgb.shape[2] == 1:
                    img_np_rgb = np.repeat(img_np_rgb, 3, axis=2)
                elif img_np_rgb.shape[2] > 3:
                    img_np_rgb = img_np_rgb[:, :, :3]
                
                # 检测手部（使用 RGB 格式）
                bboxes = self.detect_hands_in_image(img_np_rgb.copy())
                left_bbox = bboxes['left_hand']
                right_bbox = bboxes['right_hand']
                
                # 转换为 BGR 用于 cv2.rectangle（cv2 使用 BGR 格式）
                img_np_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
                
                # 绘制边界框（红色，线宽 3）
                # cv2.rectangle 使用 BGR 格式，红色为 (0, 0, 255)
                if left_bbox is not None:
                    x1, y1, x2, y2 = left_bbox
                    cv2.rectangle(img_np_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)  # BGR格式，红色，线宽3
                
                if right_bbox is not None:
                    x1, y1, x2, y2 = right_bbox
                    cv2.rectangle(img_np_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)  # BGR格式，红色，线宽3
                
                # 转换回 RGB 格式
                img_np_rgb_with_bbox = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2RGB)
                
                # 转换回 torch tensor，值域 [0, 255] -> [0, 1]
                img_tensor_with_bbox = torch.from_numpy(img_np_rgb_with_bbox).permute(2, 0, 1).float() / 255.0
                batch_images.append(img_tensor_with_bbox)
            
            images_with_bboxes.append(torch.stack(batch_images, dim=0))  # (d, c, h, w)
        
        # 重新排列回原始形状
        if len(images.shape) == 4 and images.shape[0] == d:  # 原始是 (d, c, h, w)
            result = images_with_bboxes[0]  # (d, c, h, w)
        else:  # 原始是 (b*d, c, h, w)
            result = torch.cat(images_with_bboxes, dim=0)  # (b*d, c, h, w)
        
        return result.to(images.device)

# --- 使用示例 ---
# cropper = HandCropper(model_path='hand_landmarker.task')
# orig_hand, recon_hand = cropper.get_hand_crops(batch_orig, batch_recon)