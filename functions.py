import importlib.util
import re

import cv2
import numpy as np

def labels_to_dict(file_path):
    """
    Reads labels from a text file and returns it as a dictionary.
    
    This function supports label files with the following formats:

    + Each line contains id and description separated by colon or space.
        Example: ``0:cat`` or ``0 cat``.
    + Each line contains a description only. The returned label id's are based on
    the row number.

    Args:
        file_path (str): path to the label file.

    Returns:
        Dict of (int, string) which maps label id to description.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    labels_dict = {}
    
    for row_number, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        
        if len(pair) == 2 and pair[0].strip().isdigit():
            labels_dict[int(pair[0])] = pair[1].strip()
        else:
            labels_dict[row_number] = content.strip()
    
    return labels_dict


def make_interpreter(model_path, use_tpu=False, delegate=None):
    """

    Args:
        model_path (str): path to the model file.
        use_tpu (bool): 
        delegate (str):

    Returns:
    
    """     
    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library and the preset linux based delagate 'libedgetpu.so.1'
    pkg = importlib.util.find_spec('tflite_runtime')

    if pkg:
        print("Using tflite_runtime library")
        from tflite_runtime.interpreter import Interpreter

        if use_tpu:
            from tflite_runtime.interpreter import load_delegate

            # Set the delegate if edge tpu will be used
            preset_del = 'libedgetpu.so.1'
            delegate = [load_delegate(preset_del)]
            print(f"Loading delegate '{preset_del}'\n")

        else: 
            delegate = None
            print("No delegate is used. Tensorflow model will run on the CPU\n")

    else:
        print("Using Tensorflow 2.0 library")
        from tensorflow.lite.python.interpreter import Interpreter

        if use_tpu:
            from tensorflow.lite.python.interpreter import load_delegate

            # Set the delegate if edge tpu will be used
            preset_del = 'libedgetpu.so.1'
            delegate = [load_delegate(preset_del)]
            print(f"Loading delegate '{preset_del}'\n")

        else:
            delegate = None
            print("No delegate is used. Tensorflow model will run on the CPU\n")

    interpreter = Interpreter(model_path=model_path,
                                  experimental_delegates=delegate)
        
    print(f"Interpreter: {interpreter}")

    return interpreter


def get_detection_results(interpreter, labels, init_frame, min_conf_threshold):
    '''
    Retreives the model detection results and annotates them on the frame using OpenCV

    Args:
        interpreter ():
        labels (dict): 
        init_frame (): Initial RGB frame to annotate inference results on 
        min_conf_threshold ():

    Returns: 

    '''
    import copy

    # Copy the original image to annotate over
    frame = copy.deepcopy(init_frame)
    
    # Get the frame width and height
    imH = frame.shape[0]
    imW = frame.shape[1]

    # Define efficientdet_lite0 based model output tensor indexes. 
    # See documentation for details: https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1  
    boxes_idx, classes_idx, scores_idx, num_det_idx = 1, 3, 0, 2

    # Retrieve detection results
    boxes = interpreter.get_tensor(interpreter.get_output_details()[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(interpreter.get_output_details()[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(interpreter.get_output_details()[scores_idx]['index'])[0] # Confidence of detected objects
    num_detections = interpreter.get_tensor(interpreter.get_output_details()[num_det_idx]['index'])[0] # number of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window

            # Draw white box to put label text in
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 

            # Draw label text
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 

    return frame


def main():
    print("Hello Functions")

if __name__ == "__main__":
    main()   