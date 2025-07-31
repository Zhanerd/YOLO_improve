import os
import random
import shutil
from tqdm import tqdm

def split_and_merge_yolo_datasets(dataset_paths, output_root, split_ratio=(8, 1, 1), seed=42):
    """
    合并多个 YOLO 数据集，并按比例划分 train/eval/test。
    - 跳过没有标签（.txt）对应的图像；
    - 标签和图像一一对应；
    """
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_image_label_pairs = []

    print("📂 收集所有数据集中的图像和标签对...")
    for dataset_path in dataset_paths:
        img_dir = os.path.join(dataset_path, "images")
        lbl_dir = os.path.join(dataset_path, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            print(f"⚠️  跳过无效数据集目录：{dataset_path}")
            continue

        for fname in os.listdir(img_dir):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                continue
            image_path = os.path.join(img_dir, fname)
            label_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            if not os.path.exists(label_path):
                continue  # 跳过没有标签的图像
            all_image_label_pairs.append((image_path, label_path))

    print(f"✅ 总共找到 {len(all_image_label_pairs)} 个有效样本（有标签）")

    # 随机划分
    random.seed(seed)
    random.shuffle(all_image_label_pairs)

    total = len(all_image_label_pairs)
    n_train = int(total * split_ratio[0] / sum(split_ratio))
    n_eval = int(total * split_ratio[1] / sum(split_ratio))
    n_test = total - n_train - n_eval

    split_map = {
        "train": all_image_label_pairs[:n_train],
        "eval": all_image_label_pairs[n_train:n_train + n_eval],
        "test": all_image_label_pairs[n_train + n_eval:]
    }

    # 拷贝文件
    for subset, pairs in split_map.items():
        out_img_dir = os.path.join(output_root, subset, "images")
        out_lbl_dir = os.path.join(output_root, subset, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        for img_path, lbl_path in tqdm(pairs, desc=f"处理 {subset}"):
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(lbl_path)
            shutil.copy(img_path, os.path.join(out_img_dir, img_name))
            shutil.copy(lbl_path, os.path.join(out_lbl_dir, lbl_name))

    print("🎉 数据合并与划分完成！")
    print(f"📊 训练集：{n_train}，验证集：{n_eval}，测试集：{n_test}")

if __name__ == '__main__':
    dataset_list = [
        "/home/hz/Desktop/human_ball_chair/img1",
        "/home/hz/Desktop/human_ball_chair/img2",
        "/home/hz/Desktop/human_ball_chair/img4",
        "/home/hz/Desktop/human_ball_chair/img5"
    ]

    split_and_merge_yolo_datasets(dataset_list, "/home/hz/Desktop/merged_output")