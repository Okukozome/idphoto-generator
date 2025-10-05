import cv2
import numpy as np
from PIL import Image

# 1:skin, 2:l_brow, 3:r_brow, 4:l_eye, 5:r_eye, 7:l_ear, 8:r_ear, 9:ear_r, 10:nose, 11:mouth, 12:u_lip, 13:l_lip, 17:hair
HEAD_PARTS_INDICES = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 17] 

def create_matted_head(original_image_path, mask_path, output_path):
    """根据语义分割掩码, 从原图中抠出人头 (脸+头发+耳朵)."""
    original_image = Image.open(original_image_path).convert("RGBA")
    mask_image = Image.open(mask_path).convert("L")
    
    mask_np = np.array(mask_image)
    head_mask = np.isin(mask_np, HEAD_PARTS_INDICES).astype(np.uint8) * 255
    
    # kernel = np.ones((5, 5), np.uint8)
    # head_mask = cv2.dilate(head_mask, kernel, iterations=4)
    # head_mask = cv2.GaussianBlur(head_mask, (15, 15), 0)
    
    head_mask_pil = Image.fromarray(head_mask)
    
    matted_head = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    matted_head.paste(original_image, mask=head_mask_pil)
    
    matted_head.save(output_path)
    print(f"[+] Matted head (hard edge) saved to: {output_path}")

def create_inpainting_assets(aligned_head_path, template_no_head_path, long_neck_mask_path, 
                             output_to_inpaint_path, output_inpaint_mask_path):
    """创建用于inpainting的 "待修复图" 和 "修复区域掩码"."""
    aligned_head = Image.open(aligned_head_path).convert("RGBA")
    template_no_head = Image.open(template_no_head_path).convert("RGBA")
    
    long_neck_mask_raw = Image.open(long_neck_mask_path).convert("L")
    long_neck_mask = long_neck_mask_raw.point(lambda p: 255 if p > 128 else 0, 'L')

    # 创建一个白色底板
    width, height = template_no_head.size
    to_inpaint_image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    
    to_inpaint_image.paste(template_no_head, (0, 0), template_no_head)
    to_inpaint_image.paste(aligned_head, (0, 0), aligned_head)

    
    to_inpaint_image.convert("RGB").save(output_to_inpaint_path)
    print(f"[+] Image to inpaint with WHITE background saved to: {output_to_inpaint_path}")

    # "长脖法" 创建修复区域掩码
    long_neck_np = np.array(long_neck_mask)
    head_alpha = np.array(aligned_head.split()[-1])
    
    inpaint_mask_np = np.clip(long_neck_np.astype(np.int16) - head_alpha.astype(np.int16), 0, 255).astype(np.uint8)
    _, inpaint_mask_np = cv2.threshold(inpaint_mask_np, 127, 255, cv2.THRESH_BINARY)
    
    # 膨胀掩码以覆盖更多区域
    kernel = np.ones((5, 5), np.uint8) 
    dilated_mask_np = cv2.dilate(inpaint_mask_np, kernel, iterations=1)
    Image.fromarray(dilated_mask_np).save(output_inpaint_mask_path)
    print(f"[+] Dilated inpainting mask saved to: {output_inpaint_mask_path}")

def post_process(inpainted_image_path, bg_color=(255, 255, 255)):
    """
    后处理，时间有限暂时不处理，直接返回
    """
    final_image = Image.open(inpainted_image_path).convert("RGB")
    
    return final_image