import serial
import numpy as np
import cv2

def receive_image(port, baudrate, output_path, image_size_bytes):
    ser = serial.Serial(port, baudrate)
    image_data = bytearray()

    # Read the image data from the serial port
    while len(image_data) < image_size_bytes:
        chunk = ser.read(image_size_bytes - len(image_data))
        image_data.extend(chunk)

    # Close the serial port
    ser.close()

    # Convert the byte array to a NumPy array
    img_array = np.frombuffer(image_data, dtype=np.uint8)

    # Decode the image
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Save the image to the output path
    cv2.imwrite(output_path, img)
    print(f'Image received and saved to {output_path}')

# Example usage
port = 'COM3'  # Replace with your serial port
baudrate = 115200
output_path = 'received_image.png'
image_size_bytes = 720 * 812 * 3  # Example size; replace with actual size in bytes
receive_image(port, baudrate, output_path, image_size_bytes)
