# src/pipeline.py
import os
import subprocess
import requests
import base64
from io import BytesIO
from PIL import Image

from src.image_utils import create_matted_head, create_inpainting_assets, post_process
from src.alignment import align_head

def run_command(command):
    """执行并打印shell命令"""
    print(f"[*] Running command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout)
    if result.stderr:
        print("[!] Stderr:", result.stderr)

def main_pipeline(user_image_path: str, template_id: str):
    """
    完整的证件照生成流水线
    """
    # --- 0. 定义路径 ---
    print("\n--- Step 0: Initializing paths ---")
    template_dir = f'assets/templates/{template_id}'
    user_image_name = os.path.basename(user_image_path)
    
    # 中间文件路径
    face_parsing_output_mask = f'face-parsing/assets/results/resnet18/{os.path.splitext(user_image_name)[0]}_raw.png'
    matted_head_path = 'outputs/1_matted_head.png'
    aligned_head_path = 'outputs/2_aligned_head.png'
    to_inpaint_path = 'outputs/3_to_inpaint.png'
    inpaint_mask_path = 'outputs/4_inpaint_mask.png'
    inpainted_result_path = 'outputs/5_inpainted_result.png'
    
    # 模板资源路径 (已更新为 .png)
    template_image_path = f'{template_dir}/template.png'
    landmark_template_path = f'{template_dir}/landmark_template.npy'
    template_no_head_path = f'{template_dir}/template_no_head.png'
    long_neck_mask_path = f'{template_dir}/long_neck_mask.png'

    # --- 1. 人像语义分割 ---
    print("\n--- Step 1: Face Parsing ---")
    face_parsing_input_dir = 'face-parsing/assets/images'
    temp_face_parsing_input = os.path.join(face_parsing_input_dir, user_image_name)
    import shutil
    shutil.copy(user_image_path, temp_face_parsing_input)
    
    face_parsing_cmd = [
        'python', 'face-parsing/inference.py',
        '--model', 'resnet18',
        '--weight', 'face-parsing/weights/resnet18.pt',
        '--input', temp_face_parsing_input,
        '--output', 'face-parsing/assets/results'
    ]
    run_command(face_parsing_cmd)
    os.remove(temp_face_parsing_input)

    # --- 2. 头部Matting ---
    print("\n--- Step 2: Head Matting ---")
    create_matted_head(user_image_path, face_parsing_output_mask, matted_head_path)

    # --- 3. 面部对齐 ---
    print("\n--- Step 3: Head Alignment ---")
    align_head(matted_head_path, user_image_path, landmark_template_path, template_image_path, aligned_head_path)

    # --- 4. 创建Inpainting素材 ---
    print("\n--- Step 4: Creating Inpainting Assets ---")
    create_inpainting_assets(aligned_head_path, template_no_head_path, long_neck_mask_path, to_inpaint_path, inpaint_mask_path)

    # --- 5. 调用Inpainting服务 ---
    print("\n--- Step 5: Neck Inpainting ---")
    with open(to_inpaint_path, "rb") as f_init, open(inpaint_mask_path, "rb") as f_mask:
        files = {'init_image': f_init, 'mask_image': f_mask}
        response = requests.post("http://127.0.0.1:8000/inpaint", files=files)
    
    response.raise_for_status() # 失败时抛出异常
    img_b64 = response.json()['image_base64']
    img_bytes = base64.b64decode(img_b64)
    with open(inpainted_result_path, 'wb') as f:
        f.write(img_bytes)
    print(f"[+] Inpainted result saved to: {inpainted_result_path}")

    # --- 6. 后处理，仅生成白色背景 ---
    print("\n--- Step 6: Post-processing with WHITE background ---")
    
    # 调用 post_process 生成最终的白底图
    final_image = post_process(inpainted_result_path, template_image_path, bg_color=(255, 255, 255))
    
    # 将图片转换为 base64 字符串
    buffered = BytesIO()
    final_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # 构建返回结果
    results = {
        "id_photo_white_background": "data:image/jpeg;base64," + img_str
    }
    print(f"[+] Generated white background version.")
        
    return results