import os
import cv2
import dlib
from imutils import face_utils
import numpy as np
import torch
import csv
import argparse
from torchvision import transforms
from model import gaze_network

from head_pose import HeadPoseEstimator

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)

    return image_out

def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (224, 224)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped

def generate_camera_calibration(image):
    # Generate a new camera calibration for each input image
    height, width, _ = image.shape
    camera_matrix = np.array([[width, 0, width / 2],
                              [0, width, height / 2],
                              [0, 0, 1]])
    distortion = np.zeros((4, 1))
    return camera_matrix, distortion

def save_camera_calibration(camera_matrix, distortion, filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write('Camera_Matrix', camera_matrix)
    fs.write('Distortion_Coefficients', distortion)
    fs.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaze Estimation Script')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--csv_name', type=str,default='result.csv', help='Name of the output csv file')
    parser.add_argument('--output', type=str,default='output.mp4', help='Output video file')

    args = parser.parse_args()

    #video_file_name = '/home/kpit/Desktop/XGaze/ETH-XGaze/steering_wheel/vp1/run1b_2018-05-29-14-02-47.ids_4.mp4'
    print('load input video: ', args.video)
    cap = cv2.VideoCapture(args.video)

    csv_file = open(args.csv_name, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Gaze_X', 'Gaze_Y'])
    
    # Check if video file was opened successfully
    if not cap.isOpened():
        print("Error opening video  file")
        exit()

    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()

    # Initialize video writer
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (width, height))
    
    frame_count = 1  # added this line
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # cv2.imshow("frame",frame)
        
        detected_faces = face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
        if len(detected_faces) == 0:
            print('warning: no detected face')
            csv_writer.writerow([frame_count, 0,0])
            frame_count += 1
            continue

        shape = predictor(frame, detected_faces[0])
        shape = face_utils.shape_to_np(shape)
        landmarks = []
        for (x, y) in shape:
            landmarks.append((x, y))
        landmarks = np.asarray(landmarks)

        # Generate a new camera calibration for each input frame
        camera_matrix, distortion = generate_camera_calibration(frame)
        
        # Save the camera calibration to a file
        # filename = os.path.splitext(video_file_name)[0] + '_frame_' + str(frame_count) + '.xml'  # modified this line
        # save_camera_calibration(camera_matrix, distortion, filename)
        # print('Camera calibration file saved as:', filename)

        print('estimate head pose')
        face_model_load = np.loadtxt('face_model.txt')
        landmark_use = [20, 23, 26, 29, 15, 19]
        face_model = face_model_load[landmark_use, :]
        facePts = face_model.reshape(6, 1, 3)
        landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
        landmarks_sub = landmarks_sub.astype(float)
        landmarks_sub = landmarks_sub.reshape(6, 1, 2)
        hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, distortion)

        print('data normalization, i.e. crop the face image')
        img_normalized, landmarks_normalized = normalizeData_face(frame, face_model, landmarks_sub, hr, ht, camera_matrix)

        print('load gaze estimator')
        model = gaze_network()
        # model.cuda()
        pre_trained_model_path = './ckpt/epoch_24_ckpt.pth.tar'
        if not os.path.isfile(pre_trained_model_path):
            print('the pre-trained gaze estimation model does not exist.')
            exit(0)
        ckpt = torch.load(pre_trained_model_path)
        model.load_state_dict(ckpt['model_state'], strict=True)
        model.eval()
        input_var = img_normalized[:, :, [2, 1, 0]]
        input_var = trans(input_var)
        input_var = torch.autograd.Variable(input_var.float())
        input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))
        pred_gaze = model(input_var)
        pred_gaze = pred_gaze[0]
        pred_gaze_np = pred_gaze.cpu().data.numpy()
        print(f"predicted_gaze for frame {frame_count} is {pred_gaze_np}")
        csv_writer.writerow([frame_count, pred_gaze_np[0], pred_gaze_np[1]])

        print('prepare the output')
        landmarks_normalized = landmarks_normalized.astype(int)
        for (x, y) in landmarks_normalized:
            cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
        face_patch_gaze = draw_gaze(img_normalized, pred_gaze_np)
        
        # Draw gaze direction on the original frame
        #(h, w) = frame.shape[:2]
        #length = np.min([h, w]) / 2.0
        #pos = (int(w / 2.0), int(h / 2.0))
        #dx = -length * np.sin(pred_gaze_np[1]) * np.cos(pred_gaze_np[0])
        #dy = -length * np.sin(pred_gaze_np[0])
        #cv2.arrowedLine(frame, tuple(np.round(pos).astype(np.int32)),
        #               tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), (0, 0, 255),
        #               2, cv2.LINE_AA, tipLength=0.2)

        #out.write(frame)
        frame_count += 1  # added this line

    cap.release()
    out.release()
    cv2.destroyAllWindows()

