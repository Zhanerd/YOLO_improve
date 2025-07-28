from ultralytics import YOLO


# model = YOLO('/home/hz/hz_lyb/yolo/runs/detect/train6/weights/last.pt')
# model.train(resume=True)
# model.train(data="coco_humanart.yaml",resume=True,epochs=500)

# model = YOLO("yolo11m.yaml")  # build a new model from YAML
# model = YOLO('yolo11l.pt')

# model.train(data="coco_humanart.yaml",epochs=500,imgsz=640,save_period=10,device=0,batch=0.9,optimizer='AdamW',lr0=0.0005)

model = YOLO("yolo11n-obb.yaml")

results = model.train(data="/home/hz/hz_lyb/yolo/chair_yl/chair_yl_obb.yaml", epochs=100,imgsz=640,pretrained=False)

############# 超参数说明
### batch 设置0-1意味着显存占用率，0.9是90%的占用，会自己分配对应的batchsize
### obb的yaml文件和数据集格式参考/home/hz/hz_lyb/yolo/chair_yl/chair_yl_obb.yaml
### patience 早停参数
### save_period 参数存储周期
### worker dataloader数
### optimizer 优化器不选择的话一般库里面默认sgd （SGD, Adam, AdamW, NAdam, RAdam, RMSProp，etc)
### lr
###
### ！！！注意，设置好的epoch数跑完之后没法增加训练轮次，想继续要修改ultralytics库。 参考https://blog.csdn.net/weixin_51368349/article/details/133985331