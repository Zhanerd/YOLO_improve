import torch
import os

from ultralytics import YOLO

model_path = "/home/hz/YOLO_improve/runs/detect/train11/weights/best.pt"

model = YOLO(model_path)
metric = model.val(data="/home/hz/YOLO_improve/human_ball_chair.yaml",split='test',imgsz=640,batch=16)
model.export(format="onnx", dynamic=False, simplify=True, nms=False)

# model_path = "/home/hz/YOLO_improve/runs/detect/train6/weights/best.pt"
# # model_path = "/home/hz/YOLO_improve/best.pt"
# print(os.path.exists(model_path))
# model_l = torch.load(model_path, map_location="cpu", weights_only=False)
# model = model_l["model"].eval()
# model = model.float()
# dummy_input = torch.randn(1, 3, 640, 640).to("cpu")
# # dummy_input = dummy_input.type(torch.float16)
# torch.onnx.export(model, dummy_input, "yolov8_m3.onnx", opset_version=12,
#                   input_names=["input"], output_names=["output"])
#                   # ,
#                   # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})