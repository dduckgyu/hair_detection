import cv2
from glob import glob
from pathlib import Path

def convert_label(image_name, image_width, image_height, orig_label_dir, save_dir):
    """Converts a single image's DOTA annotation to YOLO OBB format and saves it to a specified directory."""
    orig_label_path = Path(f"{orig_label_dir}/{image_name}.txt")
    save_path = Path(f"{save_dir}/{image_name}.txt")

    with orig_label_path.open('r') as f, save_path.open('w') as g:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            class_name = parts[8]
            difficult = parts[9]
            class_idx = class_mapping[class_name]
            coords = [float(p) for p in parts[:8]]
            normalized_coords = [
                coords[i] / image_width if i % 2 == 0 else coords[i] / image_height for i in range(8)]
            formatted_coords = ['{:.6g}'.format(coord) for coord in normalized_coords]
            g.write(f"{' '.join(formatted_coords)} {class_idx} {difficult}\n")

class_mapping = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3
    }

image_paths = "/workspace/yolov5_obb/dataset/val/images/*"
image_paths = glob(image_paths)

for image_path in image_paths:
    image_path2 = Path(image_path)
    image_name_without_ext = image_path2.stem
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    orig_label_dir = "/workspace/yolov5_obb/dataset/val/labelTxt"
    save_dir = "/workspace/yolov5_obb/dataset/val/label"
    convert_label(image_name_without_ext, w, h, orig_label_dir, save_dir)