# services/inpainting_server.py
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import base64
import time # 导入 time 模块

app = FastAPI()

# --- 模型加载 ---
print("[*] Loading Stable Diffusion Inpainting model...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")
print("[+] Model loaded successfully.")

@app.post("/inpaint")
async def inpaint(init_image: UploadFile = File(...), mask_image: UploadFile = File(...)):
    service_start_time = time.time()
    
    # --- 图像读取计时 ---
    t0 = time.time()
    init_img_bytes = await init_image.read()
    mask_img_bytes = await mask_image.read()
    init_img = Image.open(BytesIO(init_img_bytes)).convert("RGB")
    mask_img = Image.open(BytesIO(mask_img_bytes)).convert("L")
    t1 = time.time()
    print(f"    [Inpaint-TIMER] Image read took: {t1 - t0:.4f}s")

    prompt = "neck, high quality, detailed skin texture, plain white background, suit, detailed skin texture, photorealistic, professional lighting"
    negative_prompt = "snake, nsfw, jewelry, necklace, scarf, pattern, tattoo, cartoon, painting, 3d render, illustration"

    # --- 模型推理计时 ---
    with torch.no_grad():
        generated_image = pipe(
            prompt=prompt,
            image=init_img,
            mask_image=mask_img,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=8,
            strength=0.9
        ).images[0]
    t2 = time.time()
    print(f"    [Inpaint-TIMER] SD pipe inference took: {t2 - t1:.4f}s")
        
    # --- 编码计时 ---
    buffered = BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    t3 = time.time()
    print(f"    [Inpaint-TIMER] Encode took: {t3 - t2:.4f}s")
    
    service_end_time = time.time()
    print(f"    [Inpaint-TIMER] Full request took: {service_end_time - service_start_time:.4f}s")
    
    return {"status": "success", "image_base64": img_str}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)