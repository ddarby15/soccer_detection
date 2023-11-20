import argparse
import os
import time

import cv2
import numpy as np

import functions

CURRENT_DIR = os.getcwd()
DEFAULT_MODEL_DIR = 'models'
DEFAULT_MODEL = 'soccer_model_250_epochs.tflite'            # For non-edge TPU devices (MacOs, Linux, Raspberry Pi, etc.)
DEFAULT_LABELS = 'soccer_model.txt'    

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='.tflite model path',
                    default=os.path.join(CURRENT_DIR, DEFAULT_MODEL_DIR, DEFAULT_MODEL))
parser.add_argument('--labels', help='.txt label file path',
                    default=os.path.join(CURRENT_DIR, DEFAULT_MODEL_DIR, DEFAULT_LABELS))
parser.add_argument('--top_k', type=int, default=3,
                    help='number of categories with highest score to display')
parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
parser.add_argument('--threshold', type=float, default=0.3,
                    help='classifier score threshold')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
args = parser.parse_args()


use_tpu = args.edgetpu

# Convert model labels into dictionary format
labels = functions.labels_to_dict(args.labels)

# Create a new tflite interpreter instance using the given model path
interpreter = functions.make_interpreter(model_path=args.model, use_tpu=use_tpu)
interpreter.allocate_tensors()
    
# See efficientdet_lite0 input/output information here: https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Find the input size of the model as tuple (width, height) format
_, height, width, _ = input_details[0]['shape']
model_input_size = (width, height)

dtype = input_details[0]['dtype']

# Start an OpenCV video capture using the specified camera index
# cap = cv2.VideoCapture(args.camera_idx)
cap = cv2.VideoCapture('sample_videos/Screen Recording 2023-10-28 at 9.57.14 AM.mov')

# Get the camera frame resolution resolution
frame_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(f'\nFrame Resolution: {frame_res[0]} x {frame_res[1]}')

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # Convert the OpenCV frame (BGR) to RGB format for displaying in Matplotlib
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the frame to the models input size
    frame_resized = cv2.resize(frame, model_input_size)
    
    # efficientdet_lite0 based model input dimensions must be (None, height, width, 3) = (1, 320, 320, 3)
    input_data = np.expand_dims(frame_resized, axis=0)

    # Start the timer for inference time calculations
    start_time = time.time()

    # Perform the detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Stop the timer
    end_time = time.time()

    # Calculate the frame rate (frames per second)
    inference_time_ms = (end_time - start_time) * 1000.0
    frame_rate = 1.0 / (inference_time_ms / 1000.0)
    print(f"Frame Rate: {int(frame_rate)} FPS")

    # Retreive the model detection results and annotate onto frame
    output_frame = functions.get_detection_results(interpreter, labels, 
                                                   init_frame=frame, 
                                                   min_conf_threshold=args.threshold)

    cv2.imshow('Object Detection', output_frame)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()