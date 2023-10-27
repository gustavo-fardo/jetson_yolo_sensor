import torch
import cv2
from time import time
from ultralytics import YOLO
import serial
import argparse
import supervision as sv

class ObjectDetection:
    """
    Manages YOLO object detection from pretrained weights with serial output of detected bounding boxes

    Attributes
    ----------
    capture_index : str
        video/camera path/index
    device : str
        device defintion for YOLO model
    model
        loaded model weights from a .pt file
    CLASS_NAMES_DICT : dict (int : str)
        dictionary containing the labels for a given class index
    serial_port : SerialPort
        serial port object

    Methods
    -------
    load_model(model_path)
        loads the model to YOLO architecture
    predict(frame)
        sends image to model and returns the prediction
    plot_bboxes(frame, results)
        extracts prediction information, formats to serial communication and sends
    __call__()
        main function, opens video capture, sends image to prediction, formats and sends data for a frame through serial port
    """
    def __init__(self, capture_index, model_path, serial_port):
        """
        Parameters
        ----------
        capture_index : str
            index/path of the camera/video
        model_path : str
            path of the .pt file containing model weights
        serial_port : SerialPort
            serial port object instance
        """
        
        self.capture_index = capture_index
       
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model(model_path)
        
        self.CLASS_NAMES_DICT = self.model.model.names
        
        self.serial_port = serial_port
        
    def load_model(self, model_path):   
        """
        Loads the model to YOLO architecture

        Parameters
        ----------
        model_path : str
            path of the .pt file containing model weights
        """
        
        model = YOLO(model_path)  # load a pretrained YOLOv8n model
        model.fuse()
        return model
        
    def predict(self, frame):   
        """
        Sends image to model and returns the prediction

        Parameters
        ----------
        frame : numpy array
            a RGB image frame
        """
        
        results = self.model(frame)     
        return results

    def plot_bboxes(self, frame, results):   
        """
        Extracts prediction information, formats to serial communication and sends

        Parameters
        ----------
        frame : numpy array
            a RGB image frame
        results 
            results of a model prediction
        """
        # Send timestamp for frame
        byte_string = f"{time()}".encode('utf-8')
        serial_port.write(byte_string)

        # Extract detections for person class
        for result in results[0].boxes.cpu().numpy():    
            xyxy = result.xyxy[0]
            conf = result.conf[0]
            cls = self.CLASS_NAMES_DICT[result.cls[0].astype(int)]

            # Send formatted bounding boxes, confidence and class
            byte_string = f"{xyxy}; {conf}; {cls}".encode('utf-8')
            serial_port.write(byte_string)   

        # Shows detection image in a window if True
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
        """
        Main function, opens video capture, sends image to prediction, formats and sends data for a frame through serial port
        """
        
        # Define Video Capture
        if self.capture_index == 'csi':
            cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        elif int(self.capture_index) == 0:
            cap = cv2.VideoCapture(0)
            # Define camera frame resolution dimensions
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            cap = cv2.VideoCapture(self.capture_index)
            # Define camera frame resolution dimensions
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
        # Check if the video is available
        assert cap.isOpened()
      
        while True:
            ret, frame = cap.read()     # Retrieved flag, image frame
            assert ret
            
            results = self.predict(frame)
            self.plot_bboxes(frame, results)

            # Break loop with ctrl c
            if cv2.waitKey(5) & 0xFF == 27:
                break      
        
        cap.release()
        cv2.destroyAllWindows()
        serial_port.close()


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
    ):
    """ 
    gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
    Flip the image by setting the flip_method (most common values: 0 and 2)
    display_width and display_height determine the size of each camera pane in the window on the screen
    Default 1920x1080 displayd in a 1/4 size window
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def str2bool(v):
    """
    Checks if a string is a variation of False or True and outputs the bool value
    
    v : str
        value for checking
    """
    
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Create argument parser
parser = argparse.ArgumentParser(description='Implementação do YOLOv8 que comunica pela porta serial')
parser.add_argument('--model-path', type=str, default='./best.pt', help='caminho do modelo pre-treinado')
parser.add_argument('--capture-index', type=str, default='csi', help='caminho do video para teste | \'csi\' para csi-camera | 0 para captura da camera)')
parser.add_argument('--serial-port', type=str, default="/dev/ttyS0", help='porta serial escolhida para comunicação')
parser.add_argument('--baudrate', type=int, default=9600, help='baudrate da comunicação serial')
parser.add_argument('--show-detection', type=str2bool, default=True, help='apresenta a deteccao na tela ou nao')

# Parse arguments from terminal
args = parser.parse_args()

# Create Serial class instance
serial_port = serial.Serial(port=args.serial_port, baudrate=args.baudrate)

# Create detection class instance
detector = ObjectDetection(capture_index=args.capture_index, model_path=args.model_path, serial_port=serial_port)

# Call main
detector()
