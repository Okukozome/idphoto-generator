# src/pipeline.py
import os
import requests
import base64
from io import BytesIO
from PIL import Image
import time  # 导入 time 模块

from src.image_utils import create_matted_head, create_inpainting_assets, post_process
from src.alignment import align_head

def main_pipeline(user_image_path: str, template_id: str):
    """
    完整的证件照生成流水线 (已添加详细计时)。
    """
    # --- 总计时开始 ---
    total_start_time = time.time()
    last_step_time = total_start_time

    def print_lap_time(step_name):
        nonlocal last_step_time
        current_time = time.time()
        print(f"    [TIMER] Step '{step_name}' took: {current_time - last_step_time:.4f} seconds.")
        last_step_time = current_time

    # --- 0. 定义路径 ---
    print("\n--- Step 0: Initializing paths ---")
    template_dir = f'assets/templates/{template_id}'
    face_parsing_output_mask_path = 'outputs/0_face_parsing_mask.png'
    matted_head_path = 'outputs/1_matted_head.png'
    aligned_head_path = 'outputs/2_aligned_head.png'
    to_inpaint_path = 'outputs/3_to_inpaint.png'
    inpaint_mask_path = 'outputs/4_inpaint_mask.png'
    inpainted_result_path = 'outputs/5_inpainted_result.png'
    template_image_path = f'{template_dir}/template.png'
    landmark_template_path = f'{template_dir}/landmark_template.npy'
    template_no_head_path = f'{template_dir}/template_no_head.png'
    long_neck_mask_path = f'{template_dir}/long_neck_mask.png'
    print_lap_time("Initialization")

    # --- 1. 人像语义分割 (调用服务) ---
    print("\n--- Step 1: Face Parsing (via Service) ---")
    face_parsing_service_url = "http://127.0.0.1:8001/parse"
    with open(user_image_path, "rb") as f:
        files = {'image': (os.path.basename(user_image_path), f)}
        response = requests.post(face_parsing_service_url, files=files)
    response.raise_for_status()
    response_data = response.json()
    if response_data['status'] != 'success':
        raise RuntimeError(f"Face parsing service returned an error: {response_data.get('message', 'Unknown error')}")
    mask_b64 = response_data['mask_base64']
    mask_bytes = base64.b64decode(mask_b64)
    with open(face_parsing_output_mask_path, 'wb') as f:
        f.write(mask_bytes)
    print(f"[+] Face parsing mask saved to: {face_parsing_output_mask_path}")
    print_lap_time("Face Parsing Service Call")

    # --- 2. 头部Matting ---
    print("\n--- Step 2: Head Matting ---")
    create_matted_head(user_image_path, face_parsing_output_mask_path, matted_head_path)
    print_lap_time("Head Matting")

    # --- 3. 面部对齐 ---
    print("\n--- Step 3: Head Alignment ---")
    align_head(matted_head_path, user_image_path, landmark_template_path, template_image_path, aligned_head_path)
    print_lap_time("Head Alignment")

    # --- 4. 创建Inpainting素材 ---
    print("\n--- Step 4: Creating Inpainting Assets ---")
    create_inpainting_assets(aligned_head_path, template_no_head_path, long_neck_mask_path, to_inpaint_path, inpaint_mask_path)
    print_lap_time("Create Inpainting Assets")

    # --- 5. 调用Inpainting服务 ---
    print("\n--- Step 5: Neck Inpainting ---")
    with open(to_inpaint_path, "rb") as f_init, open(inpaint_mask_path, "rb") as f_mask:
        files = {'init_image': f_init, 'mask_image': f_mask}
        response = requests.post("http://127.0.0.1:8000/inpaint", files=files)
    response.raise_for_status()
    img_b64 = response.json()['image_base64']
    img_bytes = base64.b64decode(img_b64)
    with open(inpainted_result_path, 'wb') as f:
        f.write(img_bytes)
    print(f"[+] Inpainted result saved to: {inpainted_result_path}")
    print_lap_time("Inpainting Service Call")

    # --- 6. 后处理 ---
    print("\n--- Step 6: Post-processing with WHITE background ---")
    final_image = post_process(inpainted_result_path, bg_color=(255, 255, 255))
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    results = {"id_photo_white_background": "data:image/jpeg;base64," + img_str}
    print(f"[+] Generated white background version.")
    print_lap_time("Post-processing")

    # --- 总计时结束 ---
    total_end_time = time.time()
    print(f"\n[TOTAL TIME] Full pipeline took: {total_end_time - total_start_time:.4f} seconds.")

    return results