import os

# 读取文件并转换为新的格式
def convert_to_new_format(input_dir, output_dir):
    # 获取目录下所有txt文件
    for filename in os.listdir(input_dir):
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

if __name__ == "__main__":
    # 使用例子
    input_directory = r"C:\Users\84728\Desktop\yolo\test\labels"  # 输入目录
    output_directory = r"C:\Users\84728\Desktop\yolo\test\labels2"  # 输出目录
    convert_to_new_format(input_directory, output_directory)

    print("转换完成")
