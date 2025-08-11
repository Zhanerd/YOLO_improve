from typing import List, Tuple
import torch
from abc import ABCMeta, abstractmethod
import os
from typing import Any
import cv2
import numpy as np

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nmsv2(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy.

    Class-aware version.
    """
    final_dets = []
    #classes = np.unique(scores[:, 1])
    num_classes = np.unique(scores[:, 1])
    for cls_ind in num_classes:
        cls_scores = scores[scores[:,1]==cls_ind][:,0]
        cls_boxes = boxes[scores[:,1]==cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = cls_boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

class BaseTool(metaclass=ABCMeta):
    def __init__(self,  #### just support onnx and rt now
                 model_path: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 gpu_id: int = 0):

        if not os.path.exists(model_path):
            print('model path dont exist')

        self.model_path = model_path
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.gpu_id = gpu_id
        self.backend = None
        self.onnx_input = None
        self.onnx_output = None
        if "onnx" in os.path.basename(model_path):
            import onnxruntime as ort
            self.backend = 'onnxruntime'
            if self.gpu_id < 0:
                self.session = ort.InferenceSession(path_or_bytes=model_path,
                                                    providers=['CPUExecutionProvider'])
            else:
                self.session = ort.InferenceSession(path_or_bytes=model_path,
                                                    providers=['CUDAExecutionProvider','CPUExecutionProvider'])

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def release(self):
        if self.backend == 'onnxruntime':
            del self.session
        else:
            pass
        self.session = None

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        if img.ndim == 3:
            # build input to (B, C, H, W)
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            input = img[None, :, :, :]
        elif img.ndim == 4:
            img = img.transpose(0,3, 1, 2)
            input = np.ascontiguousarray(img, dtype=np.float32)
        else:
            print('img type not support')
            return None

        # run model
        if self.backend == "onnxruntime":
            onnx_input = {self.session.get_inputs()[0].name: input}
            onnx_output = []
            for out in self.session.get_outputs():
                onnx_output.append(out.name)
            outputs = self.session.run(onnx_output, onnx_input)
        else:
            print('backend not support')
            outputs = None
        return outputs

class YOLOv8(BaseTool):
    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (640, 640),
                 nms_thr=0.45,
                 score_thr=0.7,
                 gpu_id: int = 0):
        super().__init__(model_path=model_path,
                         model_input_size=model_input_size,
                         gpu_id=gpu_id)
        self.nms_thr = nms_thr
        self.score_thr = score_thr
        self.final_cls = list()

    ##### yolov8的调用看需求增加cls参数，默认只检测人，物体id要查coco_cat
    def __call__(self, image: np.ndarray,score_thr:float,cls:list = [0]):
        self.score_thr = score_thr
        if len(cls) == 0:
            self.final_cls = [0]
        else:
            self.final_cls = cls
        image, ratio = self.preprocess(image)
        image = np.expand_dims(image,axis=0)
        outputs = self.inference(image)[0]
        outputs = self.postprocess(outputs, ratio)
        return outputs

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(self.model_input_size[0] / img.shape[0],
                    self.model_input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img
        padded_img = padded_img / 255
        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMPose model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMPose model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        if outputs.ndim == 2 :
            outputs = np.expand_dims(outputs,axis=0)
        max_wh = 7680
        max_det = 300
        max_nms = 30000

        bs = outputs.shape[0]  # batch size
        nc = outputs.shape[1] - 4  # number of classes
        xc = np.amax(outputs[:, 4:4 + nc],1) > self.score_thr  # candidates

        output = [np.empty((0, 6))] * bs
        final_boxes = np.array([])
        final_scores = np.array([])
        final_cls_inds = np.array([])
        for index, x in enumerate(outputs):  # image index, image inference
            x = x.transpose(1, 0)[xc[index]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue
            box = x[:,:4]
            cls = x[:, 4:]

            box = wh2xy(box)  # (cx, cy, w, h) to (x1, y1, x2, y2)
            if nc > 1:
                i, j = np.nonzero(cls > self.score_thr)
                x = np.concatenate((box[i.ravel(), :], x[i.ravel(), 4 + j.ravel(), None], j[:, None].astype(np.float32)), 1)
            else:  # best class only
                i, j = np.nonzero(cls > self.score_thr)
                x = np.concatenate((box[i.ravel(), :], x[i.ravel(), 4 + j.ravel(), None], j[:, None].astype(np.float32)), 1)
            if not x.shape[0]:  # no boxes
                continue
            sorted_idx = np.argsort(x[:, 4])[::-1][:max_nms]
            x = x[sorted_idx]
            # Batched NMS
            # c = x[:, 5:6] * max_wh  # classes
            # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            boxes, scores = x[:, :4], x[:, 4]
            s = x[:, 4:]
            boxes /= ratio
            dets = multiclass_nmsv2(boxes, s, self.nms_thr, self.score_thr)
            if dets is not None:
                pack_dets = (dets[:, :4], dets[:, 4], dets[:, 5])
                final_boxes, final_scores, final_cls_inds = pack_dets
                isscore = final_scores > self.score_thr
                # iscat = final_cls_inds.astype(int) >= 0 ### 在这里的0即人的类别，参考coco或者其他数据集训练的标签序号增加
                iscat = list()
                for i in final_cls_inds:
                    iscat.append(i in self.final_cls)
                iscat = np.array(iscat)
                isbbox = [i and j for (i, j) in zip(isscore, iscat)]

                final_boxes = final_boxes[isbbox]
                final_scores = final_scores[isbbox]
                final_cls_inds = final_cls_inds[isbbox]
                final_cls_inds = final_cls_inds.astype(np.int8)
                ### 过滤超出边界的框
                # filt = list(zip(final_boxes, final_scores))
                # fil_boxes = list()
                # fil_scores = list()
                # for box, score in filt:
                #     if sum(box) < 15000:
                #         fil_boxes.append(box)
                #         fil_scores.append(score)
                # final_boxes = np.array(fil_boxes)
                # final_scores = np.array(fil_scores)


                # results = np.concatenate((final_boxes, final_scores.reshape(-1, 1), final_cls_inds.reshape(-1, 1)), axis=1)
        # if self.final_cls==[0]:
        #     return final_boxes,final_scores
        # else:
        return final_boxes,final_scores,final_cls_inds
        # results = dict()
        # results['det'] = final_boxes
        # results['scores'] = final_scores
        # results['cls'] = final_cls_inds
        # return results

def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


if __name__ == '__main__':
    model_path = "/home/hz/YOLO_improve/export2.onnx"
    model = YOLOv8(model_path=model_path,gpu_id=-1)
    video_path = "/home/hz/Desktop/Volleyball/111136_12121_9_095211_183.mp4"
    video_path = "/home/hz/Desktop/basketball/a1.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        frame_idx+=1
        if not success:
            continue
        boxes,scores,cls_inds = model(frame, 0.2, [0,2])
        for box,score,cls in zip(boxes,scores,cls_inds):
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, str(cls)+str(score), (int(box[2]), int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),2)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break