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

    def copy_and_rename(subset, src_images, src_labels, dest_images, dest_labels):
        os.makedirs(dest_images, exist_ok=True)
        os.makedirs(dest_labels, exist_ok=True)
        for image_name in tqdm(os.listdir(src_images), desc=f"Converting {src_images}"):
            if image_name.endswith(('.jpg', '.png')):
                # 如果文件重名，在名称后添加后缀
                new_image_name = image_name
                if new_image_name in existing_files[subset]:
                    base_name, ext = os.path.splitext(new_image_name)
                    new_image_name = f"{base_name}_1{ext}"
                    print(f"重命名文件 {image_name} 为 {new_image_name} 以避免冲突")

                # 复制图像文件
                shutil.copy(os.path.join(src_images, image_name), os.path.join(dest_images, new_image_name))

                # 处理标签文件
                label_name = image_name.rsplit('.', 1)[0] + '.txt'
                new_label_name = new_image_name.rsplit('.', 1)[0] + '.txt'
                if os.path.exists(os.path.join(src_labels, label_name)):
                    shutil.copy(os.path.join(src_labels, label_name), os.path.join(dest_labels, new_label_name))

                # 记录新文件名，避免重复
                existing_files[subset].add(new_image_name)

    # 遍历每个数据集路径
    for dataset_path in dataset_paths:
        # 查找 train、eval 等子文件夹中的 images 和 labels
        for subset in ["train", "eval"]:
            subset_images = os.path.join(dataset_path, subset, "images")
            subset_labels = os.path.join(dataset_path, subset, "labels")
            output_images = os.path.join(output_root, subset, "images")
            output_labels = os.path.join(output_root, subset, "labels")

            # 如果子文件夹存在则进行合并
            if os.path.isdir(subset_images) and os.path.isdir(subset_labels):
                copy_and_rename(subset, subset_images, subset_labels, output_images, output_labels)

    print("数据集合并完成！")

if __name__ == "__main__":
    # 示例调用
    merge_yolo_datasets(
        dataset_paths=[r"D:\b", r"D:\a"],
        output_root = r"D:\output",
    )

