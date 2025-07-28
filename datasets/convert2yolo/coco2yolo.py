import os
import json
import shutil
from tqdm import tqdm

class Coco2YoloConverter:
    def __init__(self, coco_json_path, coco_img_path, save_label_path, save_img_path):
        self.json_path = coco_json_path
        self.img_path = coco_img_path
        self.save_label_path = save_label_path
        self.save_img_path = save_img_path

        os.makedirs(self.save_label_path, exist_ok=True)
        os.makedirs(self.save_img_path, exist_ok=True)

        self.category_id_map = {}  # coco id → yolo id
        self.img_id_to_info = {}
        self.img_id_to_anns = {}

    def convert_bbox(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = box[0] + box[2] / 2.0
        y = box[1] + box[3] / 2.0
        w = box[2]
        h = box[3]
        return round(x * dw, 6), round(y * dh, 6), round(w * dw, 6), round(h * dh, 6)

    def load_coco_json(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        # 建立类别映射
        with open(os.path.join(self.save_label_path, 'classes.txt'), 'w') as f_class:
            for i, cat in enumerate(data['categories']):
                self.category_id_map[cat['id']] = i
                f_class.write(cat['name'] + '\n')

        # 图像信息索引
        for img in data['images']:
            self.img_id_to_info[img['id']] = img

        # 注释按 image_id 分组
        for ann in data['annotations']:
            img_id = ann['image_id']
            self.img_id_to_anns.setdefault(img_id, []).append(ann)

    def convert(self, need_crowd=False):
        self.load_coco_json()

        for img_id, img_info in tqdm(self.img_id_to_info.items(), desc=f"converting"):
            filename = os.path.basename(img_info["file_name"])
            img_width = img_info["width"]
            img_height = img_info["height"]

            yolo_label_file = os.path.splitext(filename)[0] + ".txt"
            yolo_label_path = os.path.join(self.save_label_path, yolo_label_file)
            with open(yolo_label_path, 'w') as f:
                for ann in self.img_id_to_anns.get(img_id, []):
                    if not need_crowd and ann.get('iscrowd', 0) == 1:
                        continue  # 跳过 crowd
                    bbox = self.convert_bbox((img_width, img_height), ann["bbox"])
                    cat_id = self.category_id_map[ann["category_id"]]
                    f.write(f"{cat_id} {' '.join(map(str, bbox))}\n")

            # 复制图像
            src_img_path = os.path.join(self.img_path, filename)
            dst_img_path = os.path.join(self.save_img_path, filename)
            if os.path.exists(src_img_path):
                shutil.copy(src_img_path, dst_img_path)
            else:
                if os.path.exists(yolo_label_path):
                    os.remove(yolo_label_path)
                    print(f"Warning: image not found {src_img_path}, remove label file {yolo_label_file}")


        print("✅ Conversion completed.")

    def filter_labels(self, valid_classes={0, 13, 32, 56, 57}):
        need = False
        for filename in tqdm(os.listdir(self.save_label_path), desc=f"filter labels"):
            if filename.endswith('.txt'):
                print('Processing:', filename)
                filepath = os.path.join(self.save_label_path, filename)
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                with open(filepath, 'w') as file:
                    for line in lines:
                        parts = line.strip().split()
                        try:
                            class_id = int(parts[0])
                            if class_id in valid_classes:
                                file.write(' '.join(parts) + '\n')
                                need = True
                        except ValueError:
                            print(f"Invalid class ID: {parts[0]}")
                            continue
                if need:
                    need = False
                else:
                    print("No valid classes found in the file:", filename)
                    if os.path.exists(os.path.join(self.save_img_path, filename.split('.')[0] + '.jpg')):
                        os.remove(os.path.join(self.save_img_path, filename.split('.')[0] + '.jpg'))
                    if os.path.exists(filepath):
                        os.remove(filepath)

    def update_labels(self, mapping={'0': '0','13': '1','32': '2', '56': '3', '57': '4'}):
        for filename in tqdm(os.listdir(self.save_label_path), desc=f"update labels"):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.save_label_path, filename)
                sz = os.path.getsize(filepath)
                if not sz:
                    os.remove(filepath)
                else:
                    with open(filepath, 'r') as file:
                        lines = file.readlines()

                    with open(filepath, 'w') as file:
                        for line in lines:
                            parts = line.strip().split()
                            if parts[0] in mapping:
                                parts[0] = mapping[parts[0]]
                            file.write(' '.join(parts) + '\n')

    def merge_other_yolo_datasets(self, other_datasets):
        """
        将多个 YOLO 数据集合并到第一个数据集（dataset_base）中。

        参数：
        - dataset_base: str，第一个数据集的根目录，其他数据将合并入这个目录
        - other_datasets: list[str]，其余数据集路径，将合并入 dataset_base
        """
        for subset in ['train', 'eval']:
            base_img_dir = os.path.join(self.save_img_path, subset, 'images')
            base_lbl_dir = os.path.join(self.save_label_path, subset, 'labels')

            os.makedirs(base_img_dir, exist_ok=True)
            os.makedirs(base_lbl_dir, exist_ok=True)

            existing_names = set(os.listdir(base_img_dir))

            for other_ds in other_datasets:
                other_img_dir = os.path.join(other_ds, subset, 'images')
                other_lbl_dir = os.path.join(other_ds, subset, 'labels')

                if not os.path.isdir(other_img_dir) or not os.path.isdir(other_lbl_dir):
                    continue

                for img_name in tqdm(os.listdir(other_img_dir), desc=f"Converting {other_img_dir}"):
                    if not img_name.lower().endswith(('.jpg', '.png')):
                        continue

                    src_img_path = os.path.join(other_img_dir, img_name)
                    src_lbl_path = os.path.join(other_lbl_dir, os.path.splitext(img_name)[0] + '.txt')

                    # 避免重名
                    new_img_name = img_name
                    base_name, ext = os.path.splitext(img_name)
                    count = 1
                    while new_img_name in existing_names:
                        new_img_name = f"{base_name}_{count}{ext}"
                        count += 1

                    new_lbl_name = os.path.splitext(new_img_name)[0] + '.txt'

                    # 拷贝文件
                    shutil.copy(src_img_path, os.path.join(base_img_dir, new_img_name))
                    if os.path.exists(src_lbl_path):
                        shutil.copy(src_lbl_path, os.path.join(base_lbl_dir, new_lbl_name))

                    existing_names.add(new_img_name)

        print("✅ 数据已成功合并入:", self.save_img_path)

    def convert_to_obb(self, input_dir, output_dir):
        # 获取目录下所有txt文件
        for filename in tqdm(os.listdir(input_dir), desc=f"Converting to obb"):
            if os.path.exists(output_dir) == False:
                os.makedirs(output_dir)
            print(filename)
            if filename.endswith(".txt"):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, filename)

                with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
                    for line in infile:
                        # 跳过空行
                        if not line.strip():
                            continue

                        try:
                            # 尝试解析每一行
                            parts = line.strip().split()

                            # 检查是否有足够的数据
                            if len(parts) != 5:
                                print(f"Skipping invalid line (not enough values): {line.strip()}")
                                continue

                            # 提取数据
                            cls = parts[0]
                            x1, y1, x2, y2 = map(float, parts[1:])

                            # 按顺时针顺序生成四个角点
                            new_format = f"{cls} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"

                            # 写入新的格式
                            outfile.write(new_format + '\n')

                        except ValueError as e:
                            # 处理解析错误的情况
                            print(f"Skipping line due to error: {line.strip()} | Error: {e}")
                            continue


# 示例用法
if __name__ == '__main__':
    converter = Coco2YoloConverter(
        coco_json_path=r'C:\Users\84728\Documents\coco\annotations\instances_train2017.json',
        coco_img_path=r'D:\coco_humart\train\images',
        save_label_path=r'D:\coco_humart\train1\labels',
        save_img_path=r'D:\coco_humart\train1\images'
    )
    # converter = Coco2YoloConverter(
    #     coco_json_path=r'C:\Users\84728\Documents\HumanArt\annotations\training_humanart_dance.json',
    #     coco_img_path=r'C:\Users\84728\Documents\HumanArt\dance',
    #     save_label_path=r'D:\coco_humart\dance\labels',
    #     save_img_path=r'D:\coco_humart\dance\images'
    # )
    converter.convert()
    converter.filter_labels()
    converter.update_labels()
