import torch
import cv2
from time import time
from ultralytics import YOLO
import serial
import argparse
import supervision as sv

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

    def plot_bboxes(self, frame, results):
        
        byte_string = f"{time()}".encode('utf-8')
        serial_port.write(byte_string)

        # Extract detections for person class

        for result in results[0].boxes.cpu().numpy():    
            xyxy = result.xyxy[0]
            conf = result.conf[0]
            cls = self.CLASS_NAMES_DICT[result.cls[0].astype(int)]
                
            byte_string = f"{xyxy}, {conf}, {cls}".encode('utf-8')
            serial_port.write(byte_string)   

        if args.show_detection == True:
            # Create annotator object
            box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)

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
            
            # Annotate and display frame
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

            cv2.imshow('YOLOv8 Detection', frame)
    
    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
        while True:
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            self.plot_bboxes(frame, results)

            if cv2.waitKey(5) & 0xFF == 27:
                break      
        
        cap.release()
        cv2.destroyAllWindows()
        serial_port.close()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser(description='Implementação do YOLOv8 que comunica pela porta serial')
parser.add_argument('--model-path', type=str, default='./best.pt', help='caminho do modelo pre-treinado')
parser.add_argument('--capture-index', type=str, default=0, help='caminho do video para teste (ou 0 para captura da camera)')
parser.add_argument('--serial-port', type=str, default="/dev/ttyS0", help='porta serial escolhida para comunicação')
parser.add_argument('--baudrate', type=int, default=9600, help='baudrate da comunicação serial')
parser.add_argument('--show-detection', type=str2bool, default=True, help='apresenta a deteccao na tela ou nao')
        
args = parser.parse_args()

serial_port = serial.Serial(port=args.serial_port, baudrate=args.baudrate)
detector = ObjectDetection(capture_index=args.capture_index, model_path=args.model_path, serial_port=serial_port)
detector()
