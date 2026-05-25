import numpy as np
import cv2
from Plotter import Plotter
from utilities import detect_faces, get_primary_faces, visualize_eyes, get_eye_aspect_ratio, calculate_distance, play

try:
    from pygame import mixer
except ModuleNotFoundError:
    mixer = None

#------------------------------------------------------------------------------
# 1. Initializations.
#------------------------------------------------------------------------------

BLINK = 0

CONFIG_PATH = '../data/model/deploy.prototxt'
MODEL_PATH = '../data/model/res10_300x300_ssd_iter_140000.caffemodel'
LBF_PATH = '../data/model/lbfmodel.yaml'

net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

landmarkDetector = cv2.face.createFacemarkLBF()
landmarkDetector.loadModel(LBF_PATH)

VIDEO_SOURCE = '../data/images/input-video.mp4'
cap = cv2.VideoCapture(0)
state_prev = state_curr = 'open'

#------------------------------------------------------------------------------
# 2. Execution Logic
#------------------------------------------------------------------------------

if __name__ == "__main__":
    frame_count = 0
    frame_calib = 30
    sum_ear = 0

    ret, frame = cap.read()

    frame_h, frame_w = frame.shape[:2]

    plot_width, plot_height = 800, 400
    p = Plotter(plot_width, plot_height, sample_buffer=200, scale_value=100)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret != True:
            break

        faces = detect_faces(frame, net, detection_threshold=0.90)

        if len(faces) > 0:
            primary_face = get_primary_faces(faces, frame_h, frame_w)

            if primary_face is not None:
                cv2.rectangle(frame, primary_face, (0, 255, 0), 3)

                retval, landmarksList = landmarkDetector.fit(frame, np.expand_dims(primary_face, 0))

                if retval:
                    landmarks = landmarksList[0][0]
                    # display detections
                    visualize_eyes(landmarks, frame)
                    # get eye aspect ratio
                    ear = get_eye_aspect_ratio(landmarks)

                    if frame_count < frame_calib:
                        frame_count += 1
                        sum_ear += ear
                    elif frame_count == frame_calib:
                        frame_count += 1
                        avg_ear = sum_ear/frame_count
                        HIGHER_TH = 0.90 * avg_ear
                        LOWER_TH = 0.70 * HIGHER_TH
                        print(f'SET EAR HIGH: {HIGHER_TH}')
                        print(f'SET EAR LOW: {LOWER_TH}')
                    else:
                        p.plot(ear, label='EAR')
                        # We register a blink when the eye status transitions from "closed" to "open"
                        if ear < LOWER_TH:
                            state_curr = 'closed'
                            print("state-closed (EAR): ",ear)
                        elif ear > HIGHER_TH:
                            state_curr = 'open'
                        if state_prev == 'closed' and state_curr == 'open':
                            BLINK += 1
                            print("state-open   (EAR): ", ear)
                            print("BLINK DETECTED\n")
                            if mixer:
                                play('../data/images/click.wav')

                        # Update the previous state.
                        state_prev = state_curr

                        cv2.putText(frame, "Blink Counter: {}".format(BLINK), (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.imshow('Output', frame)
                        k = cv2.waitKey(1)
                        if k == ord('q'):
                            cv2.destroyAllWindows()
                            break
            else:
                print('No valid face detected.')

    cap.release()