import os
import cv2
import json

class Yolo2LabelmeConverter:
    def __init__(self, image_dir, label_dir, class_file, output_dir, version='5.6.0'):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.class_file = class_file
        self.output_dir = output_dir
        self.class_names = self.load_class_names()

        self.label_me_version = version

        os.makedirs(self.output_dir, exist_ok=True)

    def load_class_names(self):
        with open(self.class_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def yolo_to_bbox(self, yolo_bbox, img_w, img_h):
        x_c, y_c, w, h = yolo_bbox
        x1 = (x_c - w / 2) * img_w
        y1 = (y_c - h / 2) * img_h
        x2 = (x_c + w / 2) * img_w
        y2 = (y_c + h / 2) * img_h
        return [[x1, y1], [x2, y2]]

    def convert_single_image(self, img_name):
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(label_path):
            return  # skip images with no labels

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            return

        h, w = img.shape[:2]

        shapes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # invalid line
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                points = self.yolo_to_bbox(bbox, w, h)

                shape = {
                    "label": self.class_names[class_id],
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None
                }
                shapes.append(shape)

        labelme_json = {
            "version": self.label_me_version,
            "flags": {},
            "shapes": shapes,
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }

        out_path = os.path.join(self.output_dir, os.path.splitext(img_name)[0] + '.json')
        with open(out_path, 'w') as f:
            json.dump(labelme_json, f, indent=4)

    def convert_all(self):
        for img_name in os.listdir(self.image_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.convert_single_image(img_name)

if __name__ == '__main__':
    converter = Yolo2LabelmeConverter(
        image_dir=r'D:\coco_humart\train1\images',
        label_dir=r'D:\coco_humart\train1\labels',
        class_file=r'E:\datasets\val\classes.txt',
        output_dir=r'E:\datasets\train1\images'
    )

    converter.convert_all()