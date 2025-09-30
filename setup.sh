#!/bin/bash

# --- 1. 初始化 face-parsing 子模块 ---
echo "[*] Initializing face-parsing submodule..."
git submodule add https://github.com/yakhyo/face-parsing.git
git submodule update --init --recursive

# --- 2. 下载 face-parsing 模型 ---
echo "[*] Downloading face-parsing models..."
cd face-parsing
bash download.sh
cd ..

# --- 3. 创建必要的资产目录 ---
echo "[*] Creating asset directories..."
mkdir -p assets/dlib_models
mkdir -p assets/landmark_templates
mkdir -p inputs
mkdir -p outputs

# --- 4. 下载 Dlib 面部关键点模型 ---
echo "[*] Downloading Dlib landmark model..."
wget -O assets/dlib_models/shape_predictor_68_face_landmarks.dat https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat

# --- 安装完成提示 ---
echo "[+] Setup complete! Please prepare your template images and run prepare_template.py, then run services and main api."