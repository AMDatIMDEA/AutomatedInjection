import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation 
import os
import copy, urllib3

def initAndTakePhoto(cameraID):
    video = cv2.VideoCapture(cameraID)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Ajustamos la resolución de la cámara
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    time.sleep(1)
    ret = False
    for _ in range(10):
        if video.isOpened():
            #print('Camera is connected')
            ret, frame = video.read()
            video.release()
            break
        else:
            print('Camera not connected. Retrying...')
            video.release()
            video = cv2.VideoCapture(cameraID)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
            video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            time.sleep(1)
    if not ret:
        
        for _ in range(10):
            print("WARNING! Failed to connect to camera")
            video.release()
            video = cv2.VideoCapture(cameraID)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
            video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            time.sleep(1)
            ret, frame = video.read()
            video.release()
            if ret == True:
                break
    return ret, frame

def recordUntilARUCO(cameraID, aruco_dict_type, target, robot):
    video = cv2.VideoCapture(cameraID)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Ajustamos la resolución de la cámara
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    
    found = False
    
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

             
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Convert to grayscale for better ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        markerCorners, ids, rejectedCandidates = detector.detectMarkers(gray)  

        # Draw detected markers on the frame
        if ids is not None:
            if target in ids:
                print("ArUco marker detected! Stopping recording.")
                found = True
                video.release()
                break  # Stop the loop when an ArUco marker is found
        
            
        if robot.isRobotMoving() == False:
            video.release()
            found = False
            break
    return found
    

def TakePhoto(video):
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Ajustamos la resolución de la cámara
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    ret, frame = video.read()
    return ret, frame
    
def ChooseCamara(aruco_dict_type, robotcamID = 0, show_images = False):
    videos = [item for item in os.listdir("/dev/") if item.startswith("video")]

    image_taken = False
    robotcam = 0

    for cam in videos:

        camera = cv2.VideoCapture("/dev/"+cam) 
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        return_value1, img1 = camera.read()  
        camera.release()

        if return_value1 == True:
            img = np.array(img1)
            
            if show_images == True:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
                
            dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
            parameters =  cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)

            markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)   
            if len(markerCorners) > 0:
                ids = markerIds.flatten()
                for markerID in ids:
                    print(markerID)
                    if markerID == robotcamID:
                        robotcam = '/dev/' + cam
    if robotcam == 0:
        print("Error: camera not found. Check the connection and if the ARuco is clearly visible.")
    else:
        print("Robot camera identified")
        return robotcam
    
def ChooseCamaraIP(aruco_dict_type, cameraIP = "192.168.0.0", robotcamID = 0, show_images = False):
    videos = ['8081','8082']

    image_taken = False
    robotcam = 0

    for cam in videos:

        camera = cv2.VideoCapture("http:/"+cameraIP+":"+cam) 
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        return_value1, img1 = camera.read()  
        camera.release()

        if return_value1 == True:
            img = np.array(img1)
            
            if show_images == True:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
                
            dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
            parameters =  cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)

            markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)   
            if len(markerCorners) > 0:
                ids = markerIds.flatten()
                for markerID in ids:
                    print(markerID)
                    if markerID == robotcamID:
                        robotcam = 'http://'+cameraIP+':' + cam
    if robotcam == 0:
        print("Error: camera not found. Check the connection and if the ARuco is clearly visible.")
    else:
        print("Robot camera identified")
        return robotcam
    
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, return_corners = False):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    
    detector = cv2.aruco.ArucoDetector(cv2.aruco_dict,parameters)
    
    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    #print(ids)
    
    #corners, ids, rejected_img_points = detector.detectMarkers(gray, cameraMatrix=matrix_coefficients,distCoeff=distortion_coefficients)
    rvecs = np.zeros(shape=(1,3))
    tvecs = np.zeros(shape=(1,3))
    
    markerLength=0.05
    objectCorners = np.array([[-markerLength/2, markerLength/2, 0],
                [markerLength/2, markerLength/2, 0],
                [markerLength/2, -markerLength/2, 0],
                [-markerLength/2, -markerLength/2, 0]])

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            #rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], markerLength=0.051, cameraMatrix=matrix_coefficients,
                                                                       #distCoeffs=distortion_coefficients)

            retval, rvec, tvec = cv2.solvePnP(objectCorners, corners[i], cameraMatrix=matrix_coefficients,
                                                                       distCoeffs=distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.02505)  
            
            # Draw number of arUCo
            cv2.putText(frame, f"arUCo: {ids[i][0]}", (int(corners[i][0][3][0]), int(corners[i][0][3][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            rvecs = np.append(rvecs, [np.array(rvec).flatten()], axis=0)
            tvecs = np.append(tvecs, [np.array(tvec).flatten()], axis=0)
            
    else:
        ids = []

    if return_corners == True:
        return frame, rvecs, tvecs, ids, corners
    else:
        return frame, rvecs, tvecs, ids

def get_H_cam2gripper(R_g2b, t_g2b, R_t2c, t_t2c):
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base=R_g2b, 
                                                        t_gripper2base=t_g2b,
                                                        R_target2cam=R_t2c, 
                                                        t_target2cam=t_t2c,
                                                        method=cv2.CALIB_HAND_EYE_PARK)
    M_3x4 = np.hstack((R_cam2gripper,t_cam2gripper))
    H_cam2gripper = np.vstack((M_3x4, [0,0,0,1]))
    return H_cam2gripper
    
def get_H_target2cam(arucoID, rvecs, tvecs, ids):
    flat_ids = ids.flatten()
    for i in range(len(flat_ids)):
        if ids[i] == arucoID:
            index = i+1
    dst, jacobian = cv2.Rodrigues(rvecs[index])
    H_targer2cam_3x4 = np.hstack((dst,np.reshape(tvecs[index], (3,1))))
    H_target2cam = np.vstack((H_targer2cam_3x4, [0,0,0,1]))
    return H_target2cam

def get_H_target2gripper(H_cam2gripper, H_target2cam):
    H_target2gripper = np.matmul(H_cam2gripper, H_target2cam)
    return H_target2gripper

def get_H_gripper2base(pose):
    dst, jacobian = cv2.Rodrigues(np.array(pose[3:]))
    H_gripper2base_3x4 = np.hstack((dst,np.reshape(pose[:3], (3,1))))
    H_gripper2base= np.vstack((H_gripper2base_3x4, [0,0,0,1]))
    return H_gripper2base

def get_H_target2base(H_gripper2base, H_target2gripper):
    H_target2base = np.matmul(H_gripper2base, H_target2gripper)
    return H_target2base

def getArucoXYZ(arucoID, rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose):
    pose_copy = copy.copy(pose)
    pose_copy[3] = np.radians(pose_copy[3])
    pose_copy[4] = np.radians(pose_copy[4])
    pose_copy[5] = np.radians(pose_copy[5])
    H_cam2gripper = get_H_cam2gripper(R_g2b, t_g2b, R_t2c, t_t2c)
    H_target2cam = get_H_target2cam(arucoID, rvecs, tvecs, ids)
    H_target2gripper = get_H_target2gripper(H_cam2gripper, H_target2cam)
    H_gripper2base = get_H_gripper2base(pose_copy)
    H_target2base = get_H_target2base(H_gripper2base, H_target2gripper)
    X = H_target2base[0,3]
    Y = H_target2base[1,3]
    Z = H_target2base[2,3]
    
    return X, Y, Z, H_target2base

def get_latest_mjpeg_frame(url):
    """
    Connects to an MJPEG stream and grabs the most recent full JPEG frame.

    Parameters:
        url (str): The MJPEG stream URL (e.g., "http://raspberrypi:8081").

    Returns:
        np.ndarray | None: The image frame as a NumPy array, or None if failed.
    """
    http = urllib3.PoolManager()

    try:
        response = http.request('GET', url, preload_content=False, timeout=2)
        bytes_buffer = b''

        for chunk in response.stream(1024):
            bytes_buffer += chunk
            a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
            b = bytes_buffer.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]  # discard used bytes

                img_array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                response.release_conn()
                return frame

    except Exception as e:
        print(f"Error fetching MJPEG frame: {e}")
        return None

def runCalibration(origin, robotcamID, aruco_dict_type, arucoID, posesFilename, robot, k, d, show_images=False):
    poses = []

    with open(posesFilename, 'r') as f:
        for line in f.readlines():
            pose = [float(i) for i in line.split(',')]
            poses.append(pose)
    
    R_g2b = []

    t_g2b = []

    R_t2c = []

    t_t2c = []

    robot.go_to_position(origin, speed = 0.1)
    
    #video = cv2.VideoCapture(robotcamID)
    #video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Ajustamos la resolución de la cámara
    #video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    #time.sleep(1)
    for i in range(len(poses)):
        robot.move_joints(poses[i], speed = 100)
        time.sleep(5)
        pose = robot.read_position()

        
        #flush_time = time.time()
        #while time.time() - flush_time < 1:
        #    video.read()
        
        #start_time = time.time()
        #while time.time()-start_time<2:
        #ret, frame = video.read()

        frame = get_latest_mjpeg_frame(robotcamID)
        
        #time.sleep(5.0)
        # Gripper en sistema de la base
        pose_copy = copy.copy(pose)
        pose_copy[3] = np.radians(pose_copy[3])
        pose_copy[4] = np.radians(pose_copy[4])
        pose_copy[5] = np.radians(pose_copy[5])
        r_gripper2base = np.array(pose_copy[3:])
        tr_gripper2base = np.array(pose_copy[:3])

        
        # Sacar una foto:
        output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)

        found = False
        if isinstance(ids, np.ndarray):
            #print(ids)
            flat_ids = ids.flatten()
            for i in range(len(flat_ids)):
                if ids[i] == arucoID:
                    #print("Using Aruco ", ids[i])
                    index = i+1
                    found = True
                
        if show_images==True:
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.show()
            
        if found:
            r_target2cam = rvecs[index]
            tr_target2cam = tvecs[index]
            
            R_g2b.append(r_gripper2base)
            t_g2b.append(tr_gripper2base)


            R_t2c.append(r_target2cam)
            t_t2c.append(tr_target2cam)
        
    #video.release()
    #Srobot.go_to_position(origin)
    
    return R_g2b, t_g2b, R_t2c, t_t2c

def findAllARUCOS(search_positions, origin, robotcamID, aruco_dict_type, robot, k, d, R_g2b, t_g2b, R_t2c, t_t2c, show_images = False):
    robot.go_to_position(origin)
    all_arucos_dict = dict()
    frame = get_latest_mjpeg_frame(robotcamID)

    for i in range(len(search_positions)):
        robot.move_joints(search_positions[i], speed = 100)

        frame = get_latest_mjpeg_frame(robotcamID)

        pose = robot.read_position()
        output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)
        pose = robot.read_position()

        for i in range(len(ids)):
            X, Y, Z, H = getArucoXYZ(ids[i][0], rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose)
            all_arucos_dict[ids[i][0]] = (X, Y)

        if show_images==True:
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.show()
    #video.release()
    robot.move_joints(search_positions[0], speed = 100)
    robot.go_to_position(origin)

    return all_arucos_dict

def findTargetARUCO(targetARUCO, all_arucos_dict, search_positions, origin, robotcamID, aruco_dict_type, robot, k, d, R_g2b, t_g2b, R_t2c, t_t2c, show_images = False):
    robot.go_to_position(origin)

    for i in range(len(search_positions)):
        robot.move_joints(search_positions[i], speed = 100)

        video = cv2.VideoCapture(robotcamID)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Ajustamos la resolución de la cámara
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        time.sleep(1)
        ret, frame = video.read()
        video.release()

        pose = robot.read_position()
        output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)
        pose = robot.read_position()

        for i in range(len(ids)):
            X, Y, Z, H = getArucoXYZ(ids[i][0], rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose)
            all_arucos_dict[ids[i][0]] = (X, Y)
            
        if targetARUCO in ids:
            break

        if show_images==True:
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.show()

    return all_arucos_dict

def plotAllARUCOS(all_arucos_dict):
    circle1 = plt.Circle((0, 0), 0.1, color='c', alpha = 0.5)
    plt.gca().add_patch(circle1)
    plt.annotate('Robot', (0,0))
    for aruco in all_arucos_dict.keys():
        X, Y = all_arucos_dict[aruco]
        plt.plot(X, Y, 'o')
        plt.annotate(aruco, (X, Y))
    plt.hlines([-0.25, 0.55], xmax = 0.25, xmin = -0.25, color='k')
    plt.vlines([-0.25, 0.25], ymax = 0.55, ymin = -0.25, color='k')
    plt.axis('equal')
    plt.show()
    
def reposition(robot):
    joints = copy.copy(robot.read_joints())
    pose = copy.copy(robot.read_position())
    pose[0] += -0.1*np.cos(np.radians(joints[0]))
    pose[1] += -0.1*np.sin(np.radians(joints[0]))
    pose[2] += 0.2
    robot.go_to_position(pose)
    return None
    
def goToARUCO(targetARUCO, all_arucos_dict, robot, 
              robotcam, aruco_dict_type, k, d, origin,
              R_g2b, t_g2b, R_t2c, t_t2c, search_positions):

    X, Y = all_arucos_dict[targetARUCO]
    angle = np.degrees(np.arctan2(Y,X))
    if angle < -180:
        angle += 360

    joints = copy.copy(robot.read_joints())
    joints[-1] = 180
    joints[0] = copy.copy(angle)
    robot.move_joints(joints, speed = 100)

    pose = robot.read_position()
    pose[0] = X - 0.1*np.cos(np.radians(joints[0]))
    pose[1] = Y - 0.1*np.sin(np.radians(joints[0]))
    pose[2] = 0.2
    robot.go_to_position(pose)
    
    success, frame = initAndTakePhoto(robotcam)

    pose = robot.read_position()
    output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)

    plt.imshow(output)
    plt.show()
    
    target_found = targetARUCO in ids
    reposition_needed = False
    
    for i in range(3):
        if target_found == False:
            reposition_needed = True
            reposition(robot)
            success, frame = initAndTakePhoto(robotcam)
            pose = robot.read_position()
            output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)

            plt.imshow(output)
            plt.show()

            target_found = targetARUCO in ids
    if targetARUCO in ids:
        print("Target found")
        target_found = True
        
        X, Y, Z, H = getArucoXYZ(targetARUCO, rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose)
        all_arucos_dict[targetARUCO] = X, Y
        pose = robot.read_position()
        pose[0] = X
        pose[1] = Y
        pose[2] = 0.05
        robot.go_to_position(pose)
        
        if reposition_needed:
        
            reposition(robot)
            success, frame = initAndTakePhoto(robotcam)
            pose = robot.go_to_position(robot)
            output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)

            X, Y, H, Z = getArucoXYZ(targetARUCO, rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose)
            all_arucos_dict[targetARUCO] = X, Y
            pose = robot.read_position()
            pose[0] = X
            pose[1] = Y
            pose[2] = 0.05
            robot.go_to_position(pose)
    else:
        print("Target NOT found!!!")
        print("Mapping lab again...")
        all_arucos_dict = findTargetARUCO(targetARUCO, all_arucos_dict, search_positions, origin, robotcam, aruco_dict_type, robot, k, d, R_g2b, t_g2b, R_t2c, t_t2c, show_images = False)
        if targetARUCO in all_arucos_dict.keys():
            print("Target found")
            X, Y = all_arucos_dict[targetARUCO]
            angle = np.degrees(np.arctan2(Y,X))
            if angle < 0:
                angle += 360

            joints = copy.copy(robot.read_joints())
            joints[-1] = 180
            joints[0] = copy.copy(angle)
            robot.move_joints(joints, speed = 100)

            pose = robot.read_position()
            pose[0] = X - 0.1*np.cos(np.radians(joints[0]))
            pose[1] = Y - 0.1*np.sin(np.radians(joints[0]))
            pose[2] = 0.2
            robot.go_to_position(pose)

            #success, frame = initAndTakePhoto(robotcam)
            frame = get_latest_mjpeg_frame(robotcam)

            pose = copy.copy(robot.read_position())
            output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)

            #plt.imshow(output)
            #plt.show()
            
            X, Y, Z, H = getArucoXYZ(targetARUCO, rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose)
            pose = copy.copy(robot.read_position())
            pose[0] = X
            pose[1] = Y
            pose[2] = 0.05
            robot.go_to_position(pose)
        else:
            print("Target NOT found!!!")
            print("I surrender :(")
    return X, Y, H, all_arucos_dict

def goToARUCO2(targetARUCO, all_arucos_dict, robot, 
              robotcam, aruco_dict_type, k, d, origin,
              R_g2b, t_g2b, R_t2c, t_t2c, save_image = False):

    X, Y = all_arucos_dict[targetARUCO]
    angle = np.degrees(np.arctan2(Y,X))
    if angle < -180:
        angle += 360

    joints = copy.copy(robot.read_joints())
    if robot.robot_type != 'xarm':
        joints[-1] = -180
    else:
        joints[-1] = 0
    joints[0] = copy.copy(angle)
    robot.move_joints(joints, speed = 100)

    pose = robot.read_position()
    pose[0] = X - 0.1*np.cos(np.radians(joints[0]))
    pose[1] = Y - 0.1*np.sin(np.radians(joints[0]))
    pose[2] = 0.15
    robot.go_to_position(pose)
    
    #success, frame = initAndTakePhoto(robotcam)
    frame = get_latest_mjpeg_frame(robotcam)

    
    pose = robot.read_position()
    output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)

    

    plt.imshow(output)
    if save_image == True:
        plt.savefig("plot.png")
        plt.close()
    plt.show()
    
    target_found = targetARUCO in ids
    
    i = 0
    
    reposition_needed = False

    while target_found == False:
        reposition_needed = True
        i += 1
        
        match i:
            
            case 1:
                
                joints = copy.copy(robot.read_joints())
                joints[-1] += 90
                robot.move_joints(joints, speed = 50, wait = False)
                
            case 2:
                
                joints = copy.copy(robot.read_joints())
                joints[-1] += -180
                robot.move_joints(joints, speed = 50, wait = False)
                joints = copy.copy(robot.read_joints())
                joints[-1] += 90
                robot.move_joints(joints, speed = 50, wait = False)
                
            case 3:
                
                position = copy.copy(robot.read_position())
                position[2] += 0.30
                robot.go_to_position(position, speed = 0.5, wait = False)
                
            case 4:
                
                joints = copy.copy(robot.read_joints())
                joints[-1] += 90
                robot.move_joints(joints, speed = 50, wait = False)
                
            case 5:
                
                joints = copy.copy(robot.read_joints())
                joints[-1] += -180
                robot.move_joints(joints, speed = 50, wait = False)
                joints = copy.copy(robot.read_joints())
                joints[-1] += 90
                robot.move_joints(joints, speed = 50, wait = False)
                
            case 6:
                
                position = copy.copy(robot.read_position())
                position[2] += 0.30
                robot.go_to_position(position, speed = 0.5, wait = False)
                
            case 7:
                
                joints = copy.copy(robot.read_joints())
                joints[0] += 90
                robot.move_joints(joints, speed = 50, wait = False)
                
            case 8:
                
                joints = copy.copy(robot.read_joints())
                joints[0] += -180
                robot.move_joints(joints, speed = 50, wait = False)
                joints = copy.copy(robot.read_joints())
                joints[-1] += 90
                robot.move_joints(joints, speed = 50, wait = False)
                
            case 9:
                if not target_found:
                    print("Target not found, I give up :(")
                    break
                
                
        target_found = recordUntilARUCO(robotcam, aruco_dict_type, targetARUCO, robot)
        robot.stop()
            
    #success, frame = initAndTakePhoto(robotcam)
    frame = get_latest_mjpeg_frame(robotcam)
    
    pose = robot.read_position()
    output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)  
    
    X, Y, Z, H = getArucoXYZ(targetARUCO, rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose)

    #print(Z)
    #input("continue?")
    
    pose = robot.read_position()
    pose[0] = X 
    pose[1] = Y 
    pose[2] = Z + 0.1
    
    robot.go_to_position(pose)
    
    if reposition_needed == True:
        angle = np.degrees(np.arctan2(Y,X))
        if angle < -180:
            angle += 360

        joints = copy.copy(robot.read_joints())
        joints[-1] = -180
        joints[0] = copy.copy(angle)
        robot.move_joints(joints, speed = 100)

        pose = robot.read_position()
        pose[0] = X - 0.1*np.cos(np.radians(joints[0]))
        pose[1] = Y - 0.1*np.sin(np.radians(joints[0]))
        pose[2] = 0.2
        robot.go_to_position(pose)

        #success, frame = initAndTakePhoto(robotcam)
        frame = get_latest_mjpeg_frame(robotcam)
        
        pose = robot.read_position()
        output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)

        X, Y, Z, H = getArucoXYZ(targetARUCO, rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose)
    
        pose = robot.read_position()
        pose[0] = X 
        pose[1] = Y 
        pose[2] = Z + 0.1
        robot.go_to_position(pose)
        
    all_arucos_dict[targetARUCO] = X, Y
    
    return X, Y, H, all_arucos_dict

def fineArucoPosition(targetARUCO, all_arucos_dict, robot, 
              robotcam, aruco_dict_type, k, d, origin,
              R_g2b, t_g2b, R_t2c, t_t2c, save_image = False):
    
    frame = get_latest_mjpeg_frame(robotcam)
    pose = robot.read_position()
    output, rvecs, tvecs, ids = pose_estimation(frame, aruco_dict_type, k, d)
    X, Y, Z, H = getArucoXYZ(targetARUCO, rvecs, tvecs, ids, R_g2b, t_g2b, R_t2c, t_t2c, pose)
    all_arucos_dict[targetARUCO] = X, Y

    return X, Y, H, all_arucos_dict