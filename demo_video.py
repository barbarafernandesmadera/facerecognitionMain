import cv2
import time
import pandas as pd
import argparse
from facerecognition.inference import FaceRecognition
from spoofing.inference import Spoofing
from facerecognition.comparison import One2Many


def main(video_path, output_video):
    """
    Main script for performing face recognition and spoofing detection on video input.
    It captures video frames, detects faces, evaluates for spoofing, and recognizes identified faces.
    
    Parameters:
    video_path (str or int): The path to the input video file or 0 for webcam.
    output_video (str): The name of the output video file.
    """
    # Initialize face recognition
    face_recognition = FaceRecognition(image_size=160, margin=0,
                                       min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                                       factor=0.709, post_process=True)
    # Initialize spoofing detection
    anti_spoofing = Spoofing(model_dir='spoofing/anti_spoof_models')
    # Initialize face comparison
    one2many = One2Many('bray', 'data/db')
    # Load database control file
    db = pd.read_csv('data/db_control.csv')

    print("Classes loaded")

    # Database update time interval
    start_time = time.time()
    update_time = 900  # seconds

    if not isinstance(video_path, int):
        spoofing_th = 0.9798
        identification_th = 0.0872
    else:        
        spoofing_th = 0.75
        identification_th = 0.3

    cap = cv2.VideoCapture(video_path)

    # Check if video capture was successfully initialized
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        exit()

    # Set output video parameters
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG codec
    fps = 5.0  # Constant FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - start_time
        if update_time < current_time:
            print("Updating DB")
            one2many.update_db()
            start_time = time.time()  # Reset start time after updating

        # Detect faces
        faces, bboxes = face_recognition.detect_faces(frame)
        
        if bboxes is not None:
            for face, bbox in zip(faces, bboxes):
                color = (0, 255, 0)  # Green for valid recognition
                x1, y1, x2, y2 = [int(coord) for coord in bbox]

                # Spoofing detection
                if not anti_spoofing.is_spoofing(frame, bbox, spoofing_th):
                    # Extract face embedding
                    try:
                        embedding = face_recognition.image_embedding(frame, face)

                        # Compare with the database
                        distance, subject_id = one2many.compare(embedding)[0]
                        # Check if the distance is below the threshold
                        if distance < identification_th:
                            image_name = db[db['subject_id'] == subject_id]['image_name'].values[0].split('.')[0]
                            label = f"ID: {image_name}"
                            color = (0, 255, 0)  # Green for valid recognition
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        # else:
                        #     label = "Unknown"
                        #     color = (0, 0, 255)  # Red for unknown
                        #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    except Exception as e:
                        pass
                else:
                    label = "Spoofing Detected"
                    color = (0, 0, 255)  # Red for spoofing
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Write frame to output video
        out.write(frame)

        # Display the frame with detections
        cv2.imshow('Face Recognition and Spoofing Detection', frame)

        # Stop the loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer, and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Recognition and Spoofing Detection')
    parser.add_argument('--video_path', type=str, default=0, help='Path to the input video file or 0 for webcam')
    parser.add_argument('--output_video', type=str, default='output.mp4', help='Name of the output video file')
    args = parser.parse_args()
    
    main(args.video_path, args.output_video)


# python demo_video.py --video_path data/videos/real/real_client001_android_SD_scene01.mp4 --output_video data/results/video/real_video_client1.mp4
# python demo_video.py --video_path data/videos/real/real_client032_android_SD_scene01.mp4 --output_video data/results/video/real_video_client32.mp4
# python demo_video.py --video_path data/videos/fake/attack_client001_android_SD_printed_photo_scene01.mp4 --output_video data/results/video/fake_video_client1.mp4
# python demo_video.py --video_path data/videos/fake/attack_client032_android_SD_ipad_video_scene01.mp4 --output_video data/results/video/fake_video_client32.mp4
# python demo_video.py --output_video data/results/video/katia.mp4
# python demo_video.py --output_video data/results/video/live_demo.mp4