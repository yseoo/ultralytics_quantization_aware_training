import nncf
from ultralytics import YOLO
import torch

def transform_fn(data_item):
    print("Data item:", data_item)
    return data_item['img']

model = YOLO('yolov8n.pt',task='detect')
model.set_trainer(data='/home/hinard/ultralytics_quantization_aware_training/cfg_retripa.yaml', 
                name='quant_aw_test', 
                epochs=300, 
                patience=150, 
                imgsz=640, 
                batch=1, 
                device='0')

dataloader = model.get_dataloader('/home/hinard/Conveyor-Waste-Detection/dataset/images/train_retripa',
                                    batch_size=1,
                                    rank=-1,
                                    mode='train',
                                    nncf=True)

quantization_dataset = nncf.Dataset(dataloader, transform_fn)
quantized_model = nncf.quantize(model, quantization_dataset)