import os
import cv2
import numpy as np
import random
from glob import glob
from tqdm import tqdm

class BallDataAugmentor:
    def __init__(self,
                 ball_dir,
                 background_dir,
                 output_dir,
                 max_balls_per_image=10,
                 use_prob=0.7,
                 save_size=(640, 640),
                 class_id=0, corner_clip_prob=0.6, corner_clip_ratio=(0.1, 0.6), corner_clip_max_corners=2, corner_erode_prob=0.5):
        self.ball_dir = ball_dir
        self.background_dir = background_dir
        self.output_dir = output_dir
        self.max_balls_per_image = max_balls_per_image
        self.use_prob = use_prob
        self.save_size = save_size
        self.class_id = class_id

        ### 单张背景最多球数
        self.num_samples_per_bg = 40

        ### 单张球最大最小重复次数
        self.min_balls_per_type = 0
        self.max_balls_per_type = 2

        # 最大贴图尺寸，防止太大
        self.max_paste_ratio = 1.5 / max_balls_per_image

        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/labels", exist_ok=True)

        ### 裁剪相关
        self.corner_clip_prob = corner_clip_prob                # 执行边角剪辑的概率
        self.corner_clip_ratio = corner_clip_ratio              # 每个角裁掉的强度占短边比例区间
        self.corner_clip_max_corners = corner_clip_max_corners  # 最多剪几个角
        self.corner_erode_prob = corner_erode_prob              # 剪完后是否做轻度腐蚀，让边缘更自然

        # 匹配常见图像类型
        img_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp']
        self.ball_paths = []
        self.background_paths = []
        for ext in img_exts:
            self.ball_paths.extend(glob(os.path.join(ball_dir, ext)))
            self.background_paths.extend(glob(os.path.join(background_dir, ext)))

    @staticmethod
    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xi1, yi1 = max(x1, x1_), max(y1, y1_)
        xi2, yi2 = min(x2, x2_), min(y2, y2_)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area else 0

    def apply_augmentations(self, ball_img, bg_shape):
        bg_h, bg_w = bg_shape[:2]

        # 原最大尺寸
        max_dim = int(min(bg_w, bg_h) * self.max_paste_ratio)

        h, w = ball_img.shape[:2]
        scale0 = min(max_dim / w, max_dim / h, 1.0)

        # 加入随机波动，比如 0.5x 到 1.5x 之间
        random_scale = random.uniform(0.1, 3)
        scale0 *= random_scale

        w0, h0 = int(w * scale0), int(h * scale0)
        ball_img = cv2.resize(ball_img, (w0, h0))

        # ⚡ 额外增强因子
        scale = random.uniform(0.8, 1.2)
        stretch_x = random.uniform(0.9, 1.1)  # 水平方向拉伸
        stretch_y = random.uniform(0.9, 1.1)  # 垂直方向拉伸
        angle = random.uniform(0, 360)
        brightness = random.uniform(0.2, 1.8)

        h, w = ball_img.shape[:2]
        bgr = ball_img[:, :, :3]
        alpha = ball_img[:, :, 3]

        # 🌟 修复 16 位图像问题（转换为 8 位）
        if bgr.dtype == np.uint16:
            bgr = (bgr / 256).astype(np.uint8)
        elif bgr.dtype != np.uint8:
            bgr = bgr.astype(np.uint8)

        # 🌟 非等比缩放（模拟形变）
        new_w = max(1, int(w * scale * stretch_x))
        new_h = max(1, int(h * scale * stretch_y))
        bgr = cv2.resize(bgr, (new_w, new_h))
        alpha = cv2.resize(alpha, (new_w, new_h))

        # 🌟 旋转
        M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
        bgr = cv2.warpAffine(bgr, M, (new_w, new_h), borderValue=(0, 0, 0))
        alpha = cv2.warpAffine(alpha, M, (new_w, new_h), borderValue=0)

        # 🌟 模拟运动模糊（有概率加）
        if random.random() < 0.5:
            ksize = random.choice([3, 5, 7])
            direction = random.choice(['horizontal', 'vertical'])
            if direction == 'horizontal':
                kernel = np.zeros((ksize, ksize))
                kernel[ksize // 2, :] = np.ones(ksize)
            else:
                kernel = np.zeros((ksize, ksize))
                kernel[:, ksize // 2] = np.ones(ksize)
            kernel /= ksize
            bgr = cv2.filter2D(bgr, -1, kernel)

        # 🌟 亮度增强
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 合并 RGBA
        alpha = np.clip(alpha, 0, 255).astype(np.uint8)

        # 随机执行边角剪辑
        if random.random() < self.corner_clip_prob:
            return self._clip_corners_rgba(cv2.merge((bgr, alpha)))
        else:
            return cv2.merge((bgr, alpha))

    def paste_ball(self, background, ball_img, boxes):
        print("Pasting ball image...")
        bh, bw = background.shape[:2]
        for _ in range(100):
            h, w = ball_img.shape[:2]
            if h >= bh or w >= bw:
                continue
            x = random.randint(0, bw - w)
            y = random.randint(0, bh - h)
            new_box = (x, y, x + w, y + h)
            if all(self.iou(new_box, b) < 0.01 for b in boxes):
                alpha = ball_img[:, :, 3] / 255.0
                for c in range(3):
                    background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * ball_img[:, :, c]
                return new_box
        return None

    def read_existing_boxes(self, label_path):
        print("Reading existing boxes...")
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = map(float, parts)
                        x1 = (cx - w / 2) * self.save_size[0]
                        y1 = (cy - h / 2) * self.save_size[1]
                        x2 = (cx + w / 2) * self.save_size[0]
                        y2 = (cy + h / 2) * self.save_size[1]
                        boxes.append((x1, y1, x2, y2))
                        labels.append(line.strip())
        return boxes, labels

    def generate(self):
        for bg_path in tqdm(self.background_paths):
            bg_raw = cv2.imread(bg_path)
            if bg_raw is None:
                continue
            for i in range(self.num_samples_per_bg):
                bg_img = cv2.resize(bg_raw.copy(), self.save_size)
                filename_base = os.path.splitext(os.path.basename(bg_path))[0]
                filename = f"{filename_base}_{i}"  # 加上序号，避免文件名冲突
                label_path = os.path.join(self.background_dir, f"{filename_base}.txt")

                # 重置 boxes & labels
                boxes, labels = self.read_existing_boxes(label_path)
                total_balls = len(labels)

                for ball_path in self.ball_paths:
                    if random.random() > self.use_prob:
                        continue
                    count = random.randint(self.min_balls_per_type, self.max_balls_per_type)
                    for _ in range(count):
                        if total_balls >= self.max_balls_per_image:
                            break
                        ball_img = cv2.imread(ball_path, cv2.IMREAD_UNCHANGED)
                        if ball_img is None or ball_img.shape[2] != 4:
                            continue
                        ball_aug = self.apply_augmentations(ball_img, bg_img.shape)
                        new_box = self.paste_ball(bg_img, ball_aug, boxes)
                        if new_box:
                            x1, y1, x2, y2 = new_box
                            cx = np.clip((x1 + x2) / 2 / self.save_size[0], 0, 1)
                            cy = np.clip((y1 + y2) / 2 / self.save_size[1], 0, 1)
                            w = np.clip((x2 - x1) / self.save_size[0], 0, 1)
                            h = np.clip((y2 - y1) / self.save_size[1], 0, 1)
                            labels.append(f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                            boxes.append(new_box)
                            total_balls += 1

                cv2.imwrite(f"{self.output_dir}/images/{filename}.jpg", bg_img)
                with open(f"{self.output_dir}/labels/{filename}.txt", "w") as f:
                    f.write("\n".join(labels))

    def _clip_corners_rgba(self, rgba):
        """对 RGBA 图像随机裁掉若干个角（以三角形方式），并可选做轻度腐蚀让边缘自然。"""
        h, w = rgba.shape[:2]
        if h < 4 or w < 4:
            return rgba

        # 随机决定裁几个角
        n = random.randint(1, max(1, self.corner_clip_max_corners))
        corners = ["tl", "tr", "bl", "br"]
        random.shuffle(corners)
        corners = corners[:n]

        # 基于短边计算三角形尺寸
        rmin, rmax = self.corner_clip_ratio
        cut = int(min(h, w) * random.uniform(rmin, rmax))
        cut = max(2, cut)

        # 生成 alpha 掩码（白=保留，黑=裁切）
        mask = np.full((h, w), 255, np.uint8)

        def cut_triangle(mask, corner, s):
            if corner == "tl":
                pts = np.array([[0, 0], [s, 0], [0, s]], np.int32)
            elif corner == "tr":
                pts = np.array([[w-1, 0], [w-1-s, 0], [w-1, s]], np.int32)
            elif corner == "bl":
                pts = np.array([[0, h-1], [s, h-1], [0, h-1-s]], np.int32)
            else:  # "br"
                pts = np.array([[w-1, h-1], [w-1-s, h-1], [w-1, h-1-s]], np.int32)
            cv2.fillPoly(mask, [pts], 0)

        for c in corners:
            cut_triangle(mask, c, cut)

        # 可选轻度腐蚀，让剪切边缘不那么生硬
        if random.random() < self.corner_erode_prob:
            k = random.choice([3, 3, 5])  # 以小核为主
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.erode(mask, kernel, iterations=1)

        # 应用到 alpha 通道
        out = rgba.copy()
        alpha = out[:, :, 3]
        # 若 alpha 非 8bit，先转 8bit
        if alpha.dtype != np.uint8:
            alpha = (alpha / alpha.max() * 255.0).astype(np.uint8)
        alpha = cv2.min(alpha, mask)  # 被剪区域 alpha 变 0
        out[:, :, 3] = alpha
        return out

if __name__=="__main__":
    augmentor = BallDataAugmentor(
        ball_dir="/home/hz/Desktop/ball_png/",            # 扣好背景的球类 PNG 图路径
        background_dir="/home/hz/Desktop/background/",    # 背景图（可能包含标签）路径
        output_dir="output/",             # 输出路径
        max_balls_per_image=10,           # 每张背景最多贴几个球
        use_prob=0.2,                     # 每张球图被选中的概率
        save_size=(640, 640),             # 输出图像大小
        class_id=2                        # 球的类别ID
    )

    augmentor.generate()

