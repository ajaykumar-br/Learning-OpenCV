import numpy as np
import cv2

try:
    from pygame import mixer
except ModuleNotFoundError:
    mixer = None
    pass

def detect_faces(image, net, detection_threshold = 0.70):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()

    faces = []

    img_h = image.shape[0]
    img_w = image.shape[1]

    for detection in detections[0][0]:
        if detection[2] >= detection_threshold:
            left   = detection[3] * img_w
            top    = detection[4] * img_h
            right  = detection[5] * img_w
            bottom = detection[6] * img_h

            face_w = right - left
            face_h = bottom - top

            face_roi = (left, top, face_w, face_h)
            faces.append(face_roi)

    return np.array(faces).astype(int)

def get_primary_faces(faces, frame_h, frame_w):
    primary_face_index = None
    face_height_max = 0

    for idx in range(len(faces)):
        face = faces[idx]
        x1 = face[0]
        y1 = face[1]
        x2 = x1 + face[2]
        y2 = y1 + face[3]
        if x1 > frame_w or y1 > frame_h or x2 > frame_w or y2 > frame_h:
            continue
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            continue

        if face[3] > face_height_max:
            primary_face_index = idx
            face_height_max = face[3]

    if primary_face_index is not None:
        primary_face_index = faces[primary_face_index]
    else:
        primary_face_index = None
    
    return primary_face_index

def visualize_eyes(landmarks, frame):
    for i in range(36, 48):
        cv2.circle(frame, tuple(landmarks[i].astype('int')), 2, (0, 255, 0), -1)

def get_eye_aspect_ratio(landmarks):
    # Compute the Euclidean distances between the two sets of
    # vertical eye landmarks.
    vert_dist_1right = calculate_distance(landmarks[37], landmarks[41])
    vert_dist_2right = calculate_distance(landmarks[38], landmarks[40])
    vert_dist_1left  = calculate_distance(landmarks[43], landmarks[47])
    vert_dist_2left  = calculate_distance(landmarks[44], landmarks[46])

    # Compute the Euclidean distance between the horizontal
    # eye landmark coordinates.
    horz_dist_right  = calculate_distance(landmarks[36], landmarks[39])
    horz_dist_left = calculate_distance(landmarks[42], landmarks[45])

    # Compute the eye aspect ratio.
    EAR_left = (vert_dist_1left + vert_dist_2left) / (2.0 * horz_dist_left)
    EAR_right = (vert_dist_1right + vert_dist_2right) / (2.0 * horz_dist_right)

    ear = (EAR_left + EAR_right )/2
    # Return the eye aspect ratio.
    return ear

def calculate_distance(A, B):
    distance = ((A[0] - B[0])**2+(A[1] - B[1])**2)**0.5
    return distance

def play(file):
    mixer.init()
    sound = mixer.Sound(file)
    sound.play()