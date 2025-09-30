# services/inpainting_server.py
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
import base64

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
    init_img_bytes = await init_image.read()
    mask_img_bytes = await mask_image.read()
    
    init_img = Image.open(BytesIO(init_img_bytes)).convert("RGB")
    mask_img = Image.open(BytesIO(mask_img_bytes)).convert("L")

    prompt = "male, necklace, 8k, high quality"
    negative_prompt = "shirt, clothes, jewelry, backend, blurry, lowres, bad anatomy, error body, error arm, error hand, error finger, error leg, error foot, error face, multiple face, multiple body, multiple arm, multiple hand, multiple finger, multiple leg, multiple foot, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limb, ugly, disgusting, blurry, dehydrated, bad proportions"

    
    with torch.no_grad():
        generated_image = pipe(
            prompt=prompt,
            image=init_img,
            mask_image=mask_img,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            guidance_scale=8,
            strength=0.9
        ).images[0]
        
    buffered = BytesIO()
    generated_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"status": "success", "image_base64": img_str}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)