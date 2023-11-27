## Run the soccer object detection model 

1.  [Set up your Coral Dev Board device](https://coral.ai/docs/dev-board/get-started/)

2.  Clone this repo onto the Coral Dev board:

    ```
    git clone https://github.com/ddarby15/soccer_player_detection.git
    ```

3.  Install the necessary dependencies in requirements.txt:

    Install individually,

    or

    ```
    pip install requirements.txt
    ```

    or

    Create a conda virtual enviuronment and install the necessary dependencies and activate the environment (11/27/23 NOTE: This file still needs to be created and added to github repo):

    ```
    conda env create -f environment.yml

    conda activate soccer_player_detection_env
    ```

5.  Run the detection model on a sample video stream:

    ```
    python detect.py
    ```

6. Run the detection model using a connected camera to the device using the 'camera_idx argument' (camera index defaults to 0. Change the index as necessary):

    ```
    python detect.py --camera_idx 0
    ```
