import torch
import numpy
import cv2
from time import time
from ultralytics import YOLO
import serial
import argparse

class ObjectDetection:

    def __init__(self, capture_index, model_path, serial_port):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model(model_path)
        
        self.CLASS_NAMES_DICT = self.model.model.names
        
        self.serial_port = serial_port
    

    def load_model(self, model_path):
       
        model = YOLO(model_path)  # load a pretrained YOLOv8n model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results

    def plot_bboxes(self, results):
        
        byte_string = f"{time()}".encode('utf-8')
        serial_port.write(byte_string)

        # Extract detections for person class

        for result in results[0].boxes.cpu().numpy():
            
            xyxy = result.xyxy[0]
            conf = result.conf[0]
            cls = self.CLASS_NAMES_DICT[result.cls[0].astype(int)]
                
            byte_string = f"{xyxy}, {conf}, {cls}".encode('utf-8')
            serial_port.write(byte_string)   
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            frame = self.plot_bboxes(results)

            if cv2.waitKey(5) & 0xFF == 27:
                break      
        
        cap.release()
        cv2.destroyAllWindows()
        serial_port.close()
        
parser = argparse.ArgumentParser(description='Implementação do YOLOv8 que comunica pela porta serial')
parser.add_argument('--model-path', type=str, default='RedeCopel/runs/detect/train/weights/best.pt', help='caminho do modelo pre-treinado')
parser.add_argument('--capture-index', type=str, default="videos/inspecao_4.MP4", help='caminho do video para teste (ou 0 para captura da camera)')
parser.add_argument('--serial-port', type=str, default=None, help='porta serial escolhida para comunicação')
parser.add_argument('--baudrate', type=int, default=9600, help='baudrate da comunicação serial')
        
args = parser.parse_args()

serial_port = serial.Serial(port=args.serial_port, baudrate=args.baudrate)
detector = ObjectDetection(capture_index=args.capture_index, model_path=args.model_path, serial_port=serial_port)
detector()

