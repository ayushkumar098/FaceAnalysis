import os
import cv2
import pkg_resources

# My libs
import spiga.demo.analyze.track.get_tracker as tr
import spiga.demo.analyze.extract.spiga_processor as pr_spiga
from spiga.demo.analyze.analyzer import VideoAnalyzer
from spiga.demo.visualize.viewer import Viewer

# Paths
video_out_path_dft = pkg_resources.resource_filename('spiga', 'demo/outputs')
if not os.path.exists(video_out_path_dft):
    os.makedirs(video_out_path_dft)

import csv
 
# Define the CSV file name
default_csv_filename = "landmarks_headpose.csv"

# Function to write frame-wise data to CSV
def write_frame_data(csv_filename,frame_number, bbox, landmarks, headpose):
    """
    Writes a single frame's landmark and headpose data to a CSV file.
 
    :param frame_number: The frame index (integer)
    :param landmarks: List of 68 (x, y) tuples
    :param headpose: Tuple (yaw, pitch, roll)
    """
    
    # Flatten the landmarks list [(x1, y1), (x2, y2), ...] -> [x1, x2, x3, .... x68, y1, y2, ...., y68]
    landmark_data = []
    for i in landmarks:    
        landmark_data.append(i[0])
    for i in landmarks:    
        landmark_data.append(i[1])
    
    # Combine frame number, landmarks, and headpose
    row = [frame_number] + list(bbox) + landmark_data + list(headpose)
    
    # Append data to the CSV file
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
 
# Initialize CSV file with a header
def initialize_csv(csv_filename):
    """Creates the CSV file with the header if it doesn't exist."""
    header = ["Frame"] + ["Bbox_x", "Bbox_y", "Bbox_w", "Bbox_h"] + [f"x{i+1}" for i in range(68)] + [f"y{i+1}" for i in range(68)] + ["Yaw", "Pitch", "Roll"]
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def main():
    import argparse
    pars = argparse.ArgumentParser(description='Face App')
    pars.add_argument('-i', '--input', type=str, default='0', help='Video input')
    pars.add_argument('-d', '--dataset', type=str, default='wflw',
                      choices=['wflw', '300wpublic', '300wprivate', 'merlrav'],
                      help='SPIGA pretrained weights per dataset')
    pars.add_argument('-t', '--tracker', type=str, default='RetinaSort',
                      choices=['RetinaSort', 'RetinaSort_Res50'], help='Tracker name')
    pars.add_argument('-sh', '--show',  nargs='+', type=str, default=['fps', 'face_id', 'landmarks', 'headpose','bbox'],
                      choices=['fps', 'bbox', 'face_id', 'landmarks', 'headpose'],
                      help='Select the attributes of the face to be displayed ')
    pars.add_argument('-s', '--save', action='store_true', help='Save record')
    pars.add_argument('-nv', '--noview', action='store_false', help='Do not visualize the window')
    pars.add_argument('--outpath', type=str, default=video_out_path_dft, help='Video output directory')
    pars.add_argument('--fps', type=int, default=30, help='Frames per second')
    pars.add_argument('--shape', nargs='+', type=int, help='Visualizer shape (W,H)')
    pars.add_argument('--csvName', type=str, default=default_csv_filename)
    args = pars.parse_args()

    if args.shape:
        if len(args.shape) != 2:
            raise ValueError('--shape requires two values: width and height. Ej: --shape 256 256')
        else:
            video_shape = tuple(args.shape)
    else:
        video_shape = None

    # if not args.noview and not args.save:
    #     raise ValueError('No results will be saved neither shown')

    video_app(args.input, spiga_dataset=args.dataset, tracker=args.tracker, fps=args.fps,
              save=args.save, output_path=args.outpath, video_shape=video_shape, visualize=args.noview, plot=args.show, csvName=args.csvName)


def video_app(input_name, spiga_dataset=None, tracker=None, fps=30, save=False,
              output_path=video_out_path_dft, video_shape=None, visualize=True, plot=(),csvName=default_csv_filename):

    # Load video
    try:
        capture = cv2.VideoCapture(int(input_name))
        video_name = None
        if not visualize:
            print('WARNING: Webcam must be visualized in order to close the app')
        visualize = True

    except:
        try:
            capture = cv2.VideoCapture(input_name)
            video_name = input_name.split('/')[-1][:-4]
        except:
            raise ValueError('Input video path %s not valid' % input_name)

    if capture is not None:
        # Initialize viewer
        if video_shape is not None:
            vid_w, vid_h = video_shape
        else:
            vid_w, vid_h = capture.get(3), capture.get(4)
        viewer = Viewer('face_app', width=vid_w, height=vid_h, fps=fps)
        if save:
            viewer.record_video(output_path, video_name)

        # Initialize face tracker
        faces_tracker = tr.get_tracker(tracker)
        faces_tracker.detector.set_input_shape(capture.get(4), capture.get(3))
        # Initialize processors
        processor = pr_spiga.SPIGAProcessor(dataset=spiga_dataset)
        # Initialize Analyzer
        faces_analyzer = VideoAnalyzer(faces_tracker, processor=processor)

        # Convert FPS to the amount of milliseconds that each frame will be displayed
        if visualize:
            viewer.start_view()

        initialize_csv(csvName)  # Create CSV with header
        i = 0
        while capture.isOpened():
            i+=1
            ret, frame = capture.read()
            if ret:
                # Process frame
                objId = faces_analyzer.process_frame(frame)
                write_frame_data(csvName,i,objId[0].bbox[:4], objId[0].landmarks,objId[0].headpose[:3])
                # Show results
                key = viewer.process_image(frame, drawers=[faces_analyzer], show_attributes=plot)
                if key:
                    break
            else:
                break

        capture.release()
        viewer.close()


if __name__ == '__main__':
    main()
