import cv2
import numpy as np
import math
import os
import json
import tkinter as tk
from tkinter import messagebox

def draw_boxes(image, boxes, selected_ids=None):
    """在图像上绘制所有旋转框，选中的用红色表示"""
    for bid, pts in boxes.items():
        pts_np = np.array(pts, np.int32)
        pts_np = pts_np.reshape((-1, 1, 2))
        color = (0, 0, 255) if bid in selected_ids else (0, 255, 0)
        for pt in pts:
            cv2.putText(image, str(pts.index(pt) + 1), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color,
                        2)
        cv2.polylines(image, [pts_np], isClosed=True, color=color, thickness=2)
        # 标注框的ID
        cx = int(sum([p[0] for p in pts]) / 4)
        cy = int(sum([p[1] for p in pts]) / 4)
        cv2.putText(image, str(bid) + 'c', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def get_center(pts):
    """计算四个点的中心"""
    cx = sum([p[0] for p in pts]) / len(pts)
    cy = sum([p[1] for p in pts]) / len(pts)
    return (cx, cy)


def mouse_callback(event, x, y, flags, param):
    global selected_box_ids
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pt = np.array([x, y])
        min_dist = float('inf')
        selected = None
        # 遍历所有框，计算中心点与点击点的距离
        for bid, pts in scaled_boxes.items():
            center = np.array(get_center(pts))
            dist = np.linalg.norm(center - click_pt)
            if dist < min_dist and dist < max_click_distance:
                min_dist = dist
                selected = bid
            # 选择框的处理：按住Shift键时，允许多选
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                if selected is not None:
                    if selected not in selected_box_ids:
                        selected_box_ids.append(selected)
            else:
                selected_box_ids = [selected] if selected is not None else []


def save_json(save_path, data, json_file):
    assert save_path.split('.')[-1] == 'json'
    for label in json_file['shapes']:
        idx = json_file['shapes'].index(label)
        json_file['shapes'][idx]['points'] = data[idx]
    with open(save_path, 'w') as file:
        json.dump(json_file, file)

def ask_save_confirmation():
    """弹出保存确认对话框"""
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口
    result = messagebox.askyesno("保存确认", "是否保存修改？")
    return result

if __name__ == "__main__":
    label_path = r'D:\14z'

    all_files = os.listdir(label_path)
    files = [f for f in all_files if f.endswith('.json')]

    # 使用索引遍历，而不是 for 循环
    index = 0
    while 0 <= index < len(files):
        file = files[index]
        # 读取json文件
        json_path = os.path.join(label_path, file)
        json_file = json.load(open(json_path, 'r'))

        # 准备 boxes 数据

        boxes = {}
        for label in json_file['shapes']:
            boxes[json_file['shapes'].index(label)] = label['points']

        # 读取图片
        img_path = os.path.join(label_path, file.replace('.json', '.jpg'))
        img = cv2.imread(img_path)

        # 缩放比例，设置一个固定的缩放因子
        scale_factor = 0.8  # 缩放因子，比如0.5表示缩小一半
        scaled_img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))

        selected_box_ids = []
        max_click_distance = 50  # 限制左键点击搜索范围
        print(f"\n正在查看第 {index+1}/{len(files)} 张文件：{json_path}")

        while True:
            # 创建窗口并设置鼠标回调
            cv2.namedWindow("Adjust Rotated Boxes")
            cv2.setMouseCallback("Adjust Rotated Boxes", mouse_callback)
            disp = scaled_img.copy()

            # 将框坐标按照缩放比例进行缩放
            scaled_boxes = {}
            for bid, pts in boxes.items():
                scaled_pts = [(int(pt[0] * scale_factor), int(pt[1] * scale_factor)) for pt in pts]
                scaled_boxes[bid] = scaled_pts

            draw_boxes(disp, scaled_boxes, selected_box_ids)

            cv2.imshow("Adjust Rotated Boxes",disp)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('r'):
                # 如果有选中的框，则顺时针旋转其点序（右移一个位置）
                for selected_box_id in selected_box_ids:
                    if selected_box_id is not None:
                        pts = boxes[selected_box_id]
                        boxes[selected_box_id] = pts[-1:] + pts[:-1]
                        print(f"Box {selected_box_id} points rotated clockwise.")
            elif key == ord('d'):  # 删除选中的框
                if len(selected_box_ids) == 1:
                    selected = selected_box_ids[0]
                    del boxes[selected]
                    selected_box_ids = []  # 清空选中的框
                    print(f"Deleted box with ID: {selected}")
                else:
                    print("No box selected or multiple boxes selected. Deletion not possible.")
            elif key == ord('c'):
                # 如果有选中的框，则上下翻转标签
                for selected_box_id in selected_box_ids:
                    if selected_box_id is not None:
                        pts = boxes[selected_box_id]
                        boxes[selected_box_id] = pts[-1:] + pts[-2:-1] + pts[1:2] + pts[:1]
                        print(f"Box {selected_box_id} points convert.")
            elif key == ord('s'):
                # 弹出保存确认框
                save_confirmation = ask_save_confirmation()
                if save_confirmation:
                    print(f"Saving all boxes: {boxes}")
                    save_json(json_path, boxes, json_file)
                    print("跳转到下一张图片...")
                    index += 1
                    break
                else:
                    print("保存已取消.")
            elif key == ord('b'):
                # 返回上一张图片（如果 index > 0）
                if index > 0:
                    print("返回上一张图片...")
                    index -= 1
                    break
                else:
                    print("已经是第一张，无法再往前了.")
            elif key == ord('q'):
                # 按特定按键退出
                print("跳转到下一张图片...")
                index += 1
                break

        cv2.destroyAllWindows()
