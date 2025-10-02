# Automated Injection

🚧 **Work in Progress**: The code is provided for reference purposes only and is not intended to be run directly. It was developed and tested on a specific hardware setup, and requires modifications to run on other systems. If you wish to adapt it, please review the code carefully and adjust configuration paths, dependencies, and hardware-specific settings accordingly.

Automated handling of injected specimens using a colaborative robot, neural networks and fiducial markers as shown 
in the article "PERCEPTION-DRIVEN EXPERIMENTAL WORKFLOWS FOR SELF-DRIVING POLYMER LABS WITH MOBILE ROBOTS AND COBOT WORKCELLS"
by Miguel Hernández-del-Valle, Alejandro Ortega, Arturo de la Escalera Hueso and Maciej Haranczyk.

The content is organized as follows:

* data: contains the logs of the experiments mentioned in the article.
* Notebooks: contains the code to run the experiment, as a Jupyter notebook, as well as the results of the YOLO training, including their metrics and weights.
* src: 
    * ctrlAX_motor: Arduino code for controlling the Ultrasonic knife.
    * ARUCO_ariadne.py: a set of functions used for reading ARuco tags and correlating them with robotic positions
    * calibration_matrix.npy and distortion_coefficients.npy: calibration data of the camera required by the OpenCV library. 
    * CalibrationPosesUR10.txt: pre-recorded poses to correlate ARuco position and robot poses.
    * CalibrationParamsUR10.txt: results of running the UR10 calibration.
    * robots_config.yaml: file with the robots available for the experiment and their IP directions.
