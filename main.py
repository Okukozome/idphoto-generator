# main.py
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Dict
import os
import time
import shutil

from src.pipeline import main_pipeline

app = FastAPI(title="Intelligent ID Photo Generator API")

@app.post("/api/v1/idphoto/generate", summary="Generate ID Photo")
async def generate_id_photo(
    user_image: UploadFile = File(..., description="User's portrait photo."),
    template_id: str = Form(..., description="ID of the template to use (e.g., '001').")
) -> Dict:
    
    start_time = time.time()
    
    # --- 1. 验证模板存在性 ---
    template_dir = f'assets/templates/{template_id}'
    if not os.path.exists(os.path.join(template_dir, 'template.png')):
        raise HTTPException(status_code=404, detail=f"Template ID '{template_id}' is missing template.png.")

    # --- 2. 保存上传的图片 ---
    # 使用原始文件名以避免并发问题中的命名冲突（在实际生产中应使用UUID）
    user_image_path = os.path.join('inputs', user_image.filename)
    with open(user_image_path, "wb") as buffer:
        shutil.copyfileobj(user_image.file, buffer)

    try:
        # --- 3. 执行核心流水线 ---
        results_base64 = main_pipeline(user_image_path, template_id)
        
        # --- 4. 构造成功响应 ---
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        response_data = {
            "status": "success",
            "processing_time_seconds": processing_time,
            "results": results_base64
        }
        return response_data

    except Exception as e:
        # --- 5. 处理异常 ---
        print(f"[!!!] Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # --- 6. 清理临时文件 ---
        if os.path.exists(user_image_path):
            os.remove(user_image_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)