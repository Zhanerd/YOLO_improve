import os
import cv2
import numpy as np
import random

class CopyPasteAugmentor:
    def __init__(self, save_dir, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                 scale_range=(0.8, 1.2), rotate=True):
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.scale_range = scale_range
        self.rotate = rotate

    def load_yolo_labels(self, label_path):
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.strip().split())
                boxes.append((cls, cx, cy, w, h))
        return boxes

    def convert_yolo_to_xyxy(self, cx, cy, w, h, img_w, img_h):
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        return x1, y1, x2, y2

    def convert_xyxy_to_yolo(self, x1, y1, x2, y2, img_w, img_h):
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return cx, cy, w, h

    def adjust_brightness_contrast(self, img):
        alpha = random.uniform(*self.contrast_range)
        beta = random.uniform(*self.brightness_range) * 255 - 127
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def random_rotate_scale(self, img):
        if not self.rotate:
            return img
        angle = random.choice([0, 90, 180, 270])
        scale = random.uniform(*self.scale_range)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return rotated

    def paste_object(self, dst_img, obj_img, dst_boxes, cls):
        h_dst, w_dst = dst_img.shape[:2]
        h_obj, w_obj = obj_img.shape[:2]

        max_x = w_dst - w_obj
        max_y = h_dst - h_obj
        if max_x <= 0 or max_y <= 0:
            return

        px = random.randint(0, max_x)
        py = random.randint(0, max_y)

        roi = dst_img[py:py+h_obj, px:px+w_obj]
        mask = (obj_img > 0).any(axis=2)
        roi[mask] = obj_img[mask]
        dst_img[py:py+h_obj, px:px+w_obj] = roi

        new_cx, new_cy, new_w, new_h = self.convert_xyxy_to_yolo(px, py, px+w_obj, py+h_obj, w_dst, h_dst)
        dst_boxes.append((cls, new_cx, new_cy, new_w, new_h))

    def augment_pair(self, src_img_path, src_label_path, dst_img_path, dst_label_path):
        src_img = cv2.imread(src_img_path)
        dst_img = cv2.imread(dst_img_path)
        dst_img_copy = dst_img.copy()

        src_boxes = self.load_yolo_labels(src_label_path)
        dst_boxes = self.load_yolo_labels(dst_label_path)

        h_src, w_src = src_img.shape[:2]

        for cls, cx, cy, w, h in src_boxes:
            x1, y1, x2, y2 = self.convert_yolo_to_xyxy(cx, cy, w, h, w_src, h_src)
            obj = src_img[y1:y2, x1:x2]

            if obj.size == 0:
                continue

            obj = self.adjust_brightness_contrast(obj)
            obj = self.random_rotate_scale(obj)
            self.paste_object(dst_img_copy, obj, dst_boxes, cls)

        # Save image and label
        img_name = os.path.basename(dst_img_path)
        label_name = os.path.basename(dst_label_path)
        cv2.imwrite(os.path.join(self.save_dir, 'images', img_name), dst_img_copy)

        with open(os.path.join(self.save_dir, 'labels', label_name), 'w') as f:
            for cls, cx, cy, w, h in dst_boxes:
                f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def batch_augment(self, src_pairs, dst_pairs):
        for src, dst in zip(src_pairs, dst_pairs):
            self.augment_pair(*src, *dst)

if __name__ == "__main__":
    aug = CopyPasteAugmentor(
        save_dir="aug_out",
        brightness_range=(0.9, 1.1),
        contrast_range=(0.9, 1.1),
        scale_range=(0.9, 1.2),
        rotate=True
    )

    # 每个元组：图像路径，标签路径
    src_pairs = [
        ("src/images/ball1.jpg", "src/labels/ball1.txt"),
        ("src/images/ball2.jpg", "src/labels/ball2.txt")
    ]

    dst_pairs = [
        ("dst/images/bg1.jpg", "dst/labels/bg1.txt"),
        ("dst/images/bg2.jpg", "dst/labels/bg2.txt")
    ]

    aug.batch_augment(src_pairs, dst_pairs)
