# src/image_utils.py
import cv2
import numpy as np
from PIL import Image

HEAD_PARTS_INDICES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 17] 

def create_matted_head(original_image_path, mask_path, output_path):
    """根据语义分割掩码, 从原图中抠出人头 (脸+头发)."""
    original_image = Image.open(original_image_path).convert("RGBA")
    mask_image = Image.open(mask_path).convert("L")
    
    mask_np = np.array(mask_image)
    head_mask = np.isin(mask_np, HEAD_PARTS_INDICES).astype(np.uint8) * 255
    
    kernel = np.ones((5, 5), np.uint8)
    head_mask = cv2.dilate(head_mask, kernel, iterations=4)
    head_mask = cv2.GaussianBlur(head_mask, (15, 15), 0)
    
    head_mask_pil = Image.fromarray(head_mask)
    
    matted_head = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    matted_head.paste(original_image, mask=head_mask_pil)
    
    matted_head.save(output_path)
    print(f"[+] Matted head saved to: {output_path}")

def create_inpainting_assets(aligned_head_path, template_no_head_path, long_neck_mask_path, 
                             output_to_inpaint_path, output_inpaint_mask_path):
    """创建用于inpainting的 "待修复图" 和 "修复区域掩码"."""
    aligned_head = Image.open(aligned_head_path).convert("RGBA")
    template_no_head = Image.open(template_no_head_path).convert("RGBA")
    
    # --- 鲁棒地加载长脖掩码 ---
    # 1. 以灰度模式加载
    # 2. 二值化处理，确保只有纯黑(0)和纯白(255)
    long_neck_mask_raw = Image.open(long_neck_mask_path).convert("L")
    long_neck_mask = long_neck_mask_raw.point(lambda p: 255 if p > 128 else 0, 'L')

    # 1. 创建待修复图 (合成悬浮头和无头模板)
    to_inpaint_image = template_no_head.copy()
    # 使用头部自身的alpha通道进行粘贴，以获得平滑边缘
    to_inpaint_image.paste(aligned_head, (0, 0), aligned_head)
    to_inpaint_image.convert("RGB").save(output_to_inpaint_path)
    print(f"[+] Image to inpaint saved to: {output_to_inpaint_path}")

    # 2. "长脖法" 创建修复区域掩码
    long_neck_np = np.array(long_neck_mask)
    head_alpha = np.array(aligned_head.split()[-1])
    
    # 布尔减法: 从长脖子区域减去头部已存在的部分
    inpaint_mask_np = np.clip(long_neck_np.astype(np.int16) - head_alpha.astype(np.int16), 0, 255).astype(np.uint8)
    
    # 再次二值化确保结果干净
    _, inpaint_mask_np = cv2.threshold(inpaint_mask_np, 127, 255, cv2.THRESH_BINARY)
    
    Image.fromarray(inpaint_mask_np).save(output_inpaint_mask_path)
    print(f"[+] Inpainting mask saved to: {output_inpaint_mask_path}")

def post_process(inpainted_image_path, template_path, bg_color=(255, 255, 255)):
    """后处理，换指定背景色，并返回Pillow Image对象"""
    inpainted_image = Image.open(inpainted_image_path).convert("RGB")
    background = Image.new('RGB', inpainted_image.size, bg_color)
    template = Image.open(template_path).convert("RGBA") # 加载为RGBA以获取透明度
    
    # --- 鲁棒地从模板的Alpha通道生成前景蒙版 ---
    # 如果模板图本身就是背景透明的，直接使用其alpha通道作为mask
    if 'A' in template.getbands():
        mask = template.getchannel('A')
        # 二值化处理，应对半透明像素
        mask = mask.point(lambda p: 255 if p > 128 else 0, 'L')
    else:
        # 如果模板图不透明，则回退到基于颜色的方法
        template_rgb = template.convert("RGB")
        template_np = np.array(template_rgb)
        bg_template_color = template_np[0, 0] 
        mask_np = (np.abs(template_np.astype(int) - bg_template_color.astype(int)).sum(axis=2) > 50).astype(np.uint8) * 255
        mask = Image.fromarray(mask_np)

    final_image = Image.composite(inpainted_image, background, mask)
    return final_image