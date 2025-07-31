from yolo_onnx_infer import YOLOv8

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import cv2
import json
import time
import base64
import threading
from queue import Queue
from tqdm import tqdm

class Video2YoloConverter:
    def __init__(self, yolo_model_path, class_txt, output_dir="output_video_yolo", score_thre=0.3, labelme_version='5.6.0'):
        self.yolo = YOLOv8(yolo_model_path)
        self.class_file = class_txt
        self.output_dir = output_dir
        self.score_thre = score_thre
        self.class_names = self.load_class_names()
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        self.detected_img_dir = os.path.join(output_dir, "detected/images")
        self.detected_txt_dir = os.path.join(output_dir, "detected/labels")
        ### 为方便labelme加载，我把标注和图片保存在一个路径
        # self.detected_json_dir = os.path.join(output_dir, "detected/labelme_json")
        self.detected_json_dir = os.path.join(output_dir, "detected/images")
        self.undetected_img_dir = os.path.join(output_dir, "undetected/images")

        self.labelme_version = labelme_version

        for path in [self.detected_img_dir, self.detected_txt_dir, self.detected_json_dir, self.undetected_img_dir]:
            os.makedirs(path, exist_ok=True)

    def load_class_names(self):
        with open(self.class_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def bbox_to_yolo(self, bbox, img_w, img_h):
        x1, y1, x2, y2 = bbox
        x_c = (x1 + x2) / 2 / img_w
        y_c = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return round(x_c, 6), round(y_c, 6), round(w, 6), round(h, 6)

    def bbox_to_labelme(self, bbox):
        x1, y1, x2, y2 = bbox
        return [[x1, y1], [x2, y2]]

    def encode_image_base64(self, img_path):
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def save_labelme_json(self, img_name, h, w, bboxes, cls_inds, json_path):
        shapes = []
        for i in range(len(bboxes)):
            label = self.class_names[int(cls_inds[i])]
            bbox = bboxes[i].tolist()
            shapes.append({
                "label": label,
                "points": self.bbox_to_labelme(bbox),
                "shape_type": "rectangle",
                "group_id": None,
                "flags": {}
            })
        labelme_data = {
            "version": self.labelme_version,
            "flags": {},
            "shapes": shapes,
            "imagePath": img_name,
            "imageHeight": h,
            "imageWidth": w,
            "imageData": None
        }
        with open(json_path, 'w') as f:
            json.dump(labelme_data, f, indent=4)

    def process_frame(self, frame, frame_id, need_yolo=False, need_labelme=True):
        h, w = frame.shape[:2]
        bboxes, _, cls_inds = self.yolo(frame, self.score_thre, cls=[0, 13, 32, 56, 57])

        base_name = f"frame_{frame_id:06d}"
        img_name = base_name + ".jpg"
        img_dir = self.detected_img_dir if len(bboxes) > 0 else self.undetected_img_dir
        img_path = os.path.join(img_dir, img_name)

        count = 1
        while os.path.exists(img_path):
            base_name = f"frame_{frame_id:06d}_{count}"
            img_name = base_name + ".jpg"
            img_path = os.path.join(img_dir, img_name)
            count += 1

        cv2.imwrite(img_path, frame)

        if len(bboxes) > 0:
            # save YOLO txt
            if need_yolo:
                txt_path = os.path.join(self.detected_txt_dir, base_name + ".txt")
                with open(txt_path, 'w') as f:
                    for i in range(len(bboxes)):

                        x, y, w_box, h_box = self.bbox_to_yolo(bboxes[i], w, h)
                        class_id = self.class_to_id[cls_inds[i]]
                        f.write(f"{class_id} {x} {y} {w_box} {h_box}\n")
            # save labelme json
            if need_labelme:
                json_path = os.path.join(self.detected_json_dir, base_name + ".json")
                self.save_labelme_json(img_name, h, w, bboxes, cls_inds, json_path)

    def change_save_path(self, save_path):
        output_dir = save_path
        self.detected_img_dir = os.path.join(output_dir, "detected/images")
        self.detected_txt_dir = os.path.join(output_dir, "detected/labels")
        ### 为方便labelme加载，我把标注和图片保存在一个路径
        # self.detected_json_dir = os.path.join(output_dir, "detected/labelme_json")
        self.detected_json_dir = os.path.join(output_dir, "detected/images")
        self.undetected_img_dir = os.path.join(output_dir, "undetected/images")

    # def process_video(self, video_path, frame_interval_sec=1.0, resize=None, threads=4):
    #     cap = cv2.VideoCapture(video_path)
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     interval_frames = int(fps * frame_interval_sec)
    #     frame_id = 0
    #     read_id = 0
    #     queue = Queue(maxsize=threads*2)
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     pbar = tqdm(total=total_frames, desc="Processing video frames")
    #
    #     def worker():
    #         while True:
    #             item = queue.get()
    #             if item is None:
    #                 break
    #             self.process_frame(*item)
    #             queue.task_done()
    #
    #     threads_pool = [threading.Thread(target=worker) for _ in range(threads)]
    #     for t in threads_pool:
    #         t.start()
    #
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if read_id % interval_frames == 0:
    #             if resize:
    #                 frame = cv2.resize(frame, resize)
    #             queue.put((frame.copy(), frame_id))
    #             frame_id += 1
    #         read_id += 1
    #         pbar.update(1)
    #     pbar.close()
    #     cap.release()
    #     for _ in threads_pool:
    #         queue.put(None)
    #     for t in threads_pool:
    #         t.join()
    #
    #     print("✅ 所有视频帧已处理完毕。")

    def process_video(self, video_path, frame_interval_sec=1.0, resize=None, threads=4):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval_frames = int(fps * frame_interval_sec)
        frame_id = 0
        read_id = 0
        queue = Queue(maxsize=threads*2)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=total_frames, desc="Processing video frames")
        if total_frames < 5:
            print('video is null')
            return
        # ✅ 替代线程池
        executor = ThreadPoolExecutor(max_workers=threads)

        def worker():
            while True:
                item = queue.get()
                if item is None:
                    break
                try:
                    future = executor.submit(self.process_frame, *item)
                    # 设置处理帧的最大时间，例如 10 秒
                    future.result(timeout=10)
                except TimeoutError:
                    print("⚠️ 处理帧超时，跳过该帧")
                except Exception as e:
                    print(f"❌ 处理帧异常：{e}")
                finally:
                    queue.task_done()

        threads_pool = [threading.Thread(target=worker) for _ in range(threads)]
        for t in threads_pool:
            t.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id > total_frames:
                break
            if read_id % interval_frames == 0:
                if resize:
                    frame = cv2.resize(frame, resize)
                queue.put((frame.copy(), frame_id))
                frame_id += 1
            read_id += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        for _ in threads_pool:
            queue.put(None)
        for t in threads_pool:
            t.join()

        executor.shutdown(wait=True)
        print("✅ 所有视频帧已处理完毕。")

if __name__ == "__main__":
    video_path = '/home/hz/Desktop/Volleyball'
    converter = Video2YoloConverter(
        yolo_model_path="/home/hz/ai_sport_server/ai/yolo11l.onnx",
        class_txt='/home/hz/Desktop/classes.txt',
        output_dir='/home/hz/Desktop/volleyball_converted_output',
        score_thre=0.5
    )
    count = 0
    # skip = 920
    for root, _, files in os.walk(video_path):
        for file in files:
            # if count < skip:
            #     count += 1
            #     continue
            if file.endswith('.mp4'):
                videopath = os.path.join(root, file)
                print(f"Processing video: {videopath}")
                converter.process_video(videopath, frame_interval_sec=0.5, resize=(640, 480), threads=4)

    # videopath = r'C:\Users\84728\Desktop\ch0003_20250627T105218Z_20250627T141017Z_X00000006325000000 截取视频.mp4'
    # for i in range(1, 3):
    #     converter.process_video(videopath, frame_interval_sec=1.0, resize=(640, 480), threads=4)