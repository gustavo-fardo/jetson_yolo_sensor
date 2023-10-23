import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv
import serial


class ObjectDetection:

    def __init__(self, capture_index, model_path, serial_port):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model(model_path)
        
        self.CLASS_NAMES_DICT = self.model.model.names
        
        self.serial_port = serial_port
    

    def load_model(self):
       
        model = YOLO(model_path)  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def plot_bboxes(self, results, frame):
        
        byte_string = f"{time()}".encode('utf-8')
        serial_port.write(byte_string)
        print(byte_string)

        # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            class_id = boxes.cls[0]
            
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
            
        byte_string = f"{xyxy} {conf} {cls}".encode('utf-8')
        print(byte_string)
        serial_port.write(byte_string)   
        
        # Setup detections for visualization
        detections = sv.Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections]
        
        return frame
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break      
        
        cap.release()
        cv2.destroyAllWindows()
        serial_port.close()
        
        
serial_port = serial.Serial()
detector = ObjectDetection(capture_index="inspecao_5.MP4", model="runs/detect/train3/weights/best.pt", serial_port=serial_port)
detector()
