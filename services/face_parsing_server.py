# services/face_parsing_server.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import base64
import os
import sys
import time

# --- 路径设置 (保持不变) ---
current_file_path = os.path.abspath(__file__)
services_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(services_dir)
face_parsing_path = os.path.join(project_root, 'face-parsing')
if face_parsing_path not in sys.path:
    sys.path.insert(0, face_parsing_path)

from models.bisenet import BiSeNet

# --- FastAPI 应用和模型加载 ---
app = FastAPI(title="Face Parsing Service")

print("[*] Loading Face Parsing model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(project_root, 'face-parsing/weights/resnet18.pt')
num_classes = 19
model = BiSeNet(num_classes, backbone_name='resnet18')
model.to(device)

# --- 【修复 1】添加 weights_only=True 消除警告 ---
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print(f"[+] Face Parsing model loaded successfully on device '{device}'.")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def prepare_image(image: Image.Image, input_size=(512, 512)):
    resized_image = image.resize(input_size, resample=Image.BILINEAR)
    image_tensor = transform(resized_image)
    return image_tensor.unsqueeze(0)

# --- 【修复 2】移除 @torch.no_grad() 装饰器 ---
@app.post("/parse")
async def parse_face(image: UploadFile = File(...)):
    service_start_time = time.time()
    try:
        t0 = time.time()
        img_bytes = await image.read()
        pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        original_size = pil_image.size
        image_batch = prepare_image(pil_image).to(device)
        t1 = time.time()
        print(f"    [FaceParse-TIMER] Image read & preprocess took: {t1 - t0:.4f}s")

        # --- 【修复 2】将 torch.no_grad 用作 with 语句 ---
        with torch.no_grad():
            output = model(image_batch)[0]
            
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
        t2 = time.time()
        print(f"    [FaceParse-TIMER] Model inference took: {t2 - t1:.4f}s")

        mask_pil = Image.fromarray(predicted_mask.astype(np.uint8))
        restored_mask = mask_pil.resize(original_size, resample=Image.NEAREST)
        buffered = BytesIO()
        restored_mask.save(buffered, format="PNG")
        mask_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        t3 = time.time()
        print(f"    [FaceParse-TIMER] Post-process & encode took: {t3 - t2:.4f}s")
        
        service_end_time = time.time()
        print(f"    [FaceParse-TIMER] Full request took: {service_end_time - service_start_time:.4f}s")

        return {"status": "success", "mask_base64": mask_str}
    except Exception as e:
        print(f"[!!!] Face Parsing Error: {e}")
        # 在调试时可以返回更详细的错误
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)