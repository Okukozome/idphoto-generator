# src/alignment.py
import dlib
import cv2
import numpy as np

DLIB_MODEL_PATH = 'assets/dlib_models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

def landmarks_to_np(landmarks, dtype="int"):
    coords = np.zeros((landmarks.num_parts, 2), dtype=dtype)
    for i in range(0, landmarks.num_parts):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

def align_head(matted_head_path, user_image_path, landmark_template_path, template_image_path, output_path):
    """将抠出的人头对齐到模板位置"""
    matted_head_image = cv2.imread(matted_head_path, cv2.IMREAD_UNCHANGED)
    user_image = cv2.imread(user_image_path)
    template_image = cv2.imread(template_image_path)
    target_landmarks = np.load(landmark_template_path)

    if matted_head_image is None or user_image is None or template_image is None:
        raise IOError("Could not load one of the required images for alignment.")

    h, w, _ = template_image.shape
    gray_user = cv2.cvtColor(user_image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray_user, 1)
    if not rects:
        raise ValueError("No face found in the user image.")
    rect = max(rects, key=lambda r: r.width() * r.height())
    user_landmarks = landmarks_to_np(predictor(gray_user, rect))

    stable_indices = [36, 45, 30, 48, 54, 8] # 左眼角, 右眼角, 鼻尖, 左嘴角, 右嘴角, 下巴
    M, _ = cv2.estimateAffinePartial2D(user_landmarks[stable_indices], target_landmarks[stable_indices])
    
    if M is None:
        raise ValueError("Could not estimate transformation matrix.")

    aligned_head = cv2.warpAffine(matted_head_image, M, (w, h))
    cv2.imwrite(output_path, aligned_head)
    print(f"[+] Aligned head saved to: {output_path}")