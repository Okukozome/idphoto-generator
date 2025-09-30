# prepare_template.py
import dlib
import cv2
import numpy as np
import os
import argparse

DLIB_MODEL_PATH = 'assets/dlib_models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_MODEL_PATH)

def landmarks_to_np(landmarks, dtype="int"):
    coords = np.zeros((landmarks.num_parts, 2), dtype=dtype)
    for i in range(0, landmarks.num_parts):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

def main(args):
    template_dir = os.path.join('assets/templates', args.template_id)
    template_image_path = os.path.join(template_dir, 'template.png')
    output_path = os.path.join(template_dir, 'landmark_template.npy')

    if not os.path.exists(template_image_path):
        print(f"[!] Error: Template image not found at {template_image_path}")
        return

    print(f"[*] Processing template image: {template_image_path}")
    image = cv2.imread(template_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    if not rects:
        print("[!] Error: No faces found in the template image.")
        return
        
    rect = max(rects, key=lambda r: r.width() * r.height())
    landmarks = predictor(gray, rect)
    landmark_points = landmarks_to_np(landmarks)

    np.save(output_path, landmark_points)
    print(f"[+] Landmark template saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a facial landmark template from a template directory.")
    parser.add_argument('--template_id', type=str, required=True, help="The ID of the template folder in assets/templates.")
    args = parser.parse_args()
    main(args)