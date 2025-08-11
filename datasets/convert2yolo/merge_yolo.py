import os
import shutil
import tqdm

def merge_yolo_datasets(dataset_paths, output_root):
    """
    !!!注意，前提是同一个类别
    合并多个 YOLO 数据集，将所有子文件夹下的 images 和 labels 文件夹中的数据合并到输出路径，
    并保持 train、eval 等结构。

    参数：
    - dataset_paths: list，每个数据集的根目录路径，例如 ["path/to/dataset1", "path/to/dataset2"]
    - output_root: str，合并后数据集的根输出路径
    """
    # 记录已存在的文件名，防止重名
    existing_files = {
        "train": set(),
        "eval": set()
    }
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    global_class_list = []
    dataset_class_maps = {}  # 每个数据集的本地ID → 全局ID 映射

    def load_classes(path):
        class_file = os.path.join(path, "classes.txt")
        if not os.path.exists(class_file):
            raise FileNotFoundError(f"未找到 {class_file}")
        with open(class_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def ensure_unique_filename(name, existing_set):
        base, ext = os.path.splitext(name)
        i = 1
        new_name = name
        while new_name in existing_set:
            new_name = f"{base}_{i}{ext}"
            i += 1
        return new_name

    def remap_label_file(src_label, dst_label, id_map):
        with open(src_label, "r") as f_in, open(dst_label, "w") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                parts = line.strip().split()
                old_id = int(parts[0])
                new_id = id_map.get(old_id, -1)
                if new_id == -1:
                    continue  # 忽略未知类别
                parts[0] = str(new_id)
                f_out.write(" ".join(parts) + "\n")

    # 1. 生成 global_class_list 和每个数据集的映射表
    for dataset_path in dataset_paths:
        classes = load_classes(dataset_path)
        id_map = {}
        for local_id, name in enumerate(classes):
            if name not in global_class_list:
                global_class_list.append(name)
            id_map[local_id] = global_class_list.index(name)
        dataset_class_maps[dataset_path] = id_map

    print(f"🔢 合并后共 {len(global_class_list)} 个类别：{global_class_list}")

    # 2. 拷贝图片+重映射标签
    for dataset_path in dataset_paths:
        id_map = dataset_class_maps[dataset_path]
        for subset in ["train", "eval"]:
            img_dir = os.path.join(dataset_path, subset, "images")
            lbl_dir = os.path.join(dataset_path, subset, "labels")
            if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
                continue

            out_img_dir = os.path.join(output_root, subset, "images")
            out_lbl_dir = os.path.join(output_root, subset, "labels")
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_lbl_dir, exist_ok=True)

            for fname in tqdm.tqdm(os.listdir(img_dir), desc=f"Merging {subset} from {dataset_path}"):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMG_EXTS:
                    continue

                img_src = os.path.join(img_dir, fname)
                lbl_src = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

                new_img_name = ensure_unique_filename(fname, existing_files[subset])
                new_lbl_name = os.path.splitext(new_img_name)[0] + ".txt"
                img_dst = os.path.join(out_img_dir, new_img_name)
                lbl_dst = os.path.join(out_lbl_dir, new_lbl_name)

                shutil.copy(img_src, img_dst)
                if os.path.exists(lbl_src):
                    remap_label_file(lbl_src, lbl_dst, id_map)
                existing_files[subset].add(new_img_name)

    # 3. 写出全局 classes.txt
    class_out_path = os.path.join(output_root, "classes.txt")
    with open(class_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(global_class_list))
    print(f"✅ 数据集合并完成，类别写入：{class_out_path}")

if __name__ == "__main__":
    # 示例调用
    merge_yolo_datasets(
        dataset_paths=["/home/hz/Desktop/merged_output", "/home/hz/Desktop/yolo_cps_ball"],
        output_root = "/home/hz/Desktop/yolo_ball",
    )

