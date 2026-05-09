import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

st.title('OpenCV Deep Learning based Face Detection')
img_file_buffer = st.file_uploader('Choose a File', type=['jpg', 'jpeg', 'png'])

if 'file_uploaded_name' not in st.session_state:
    st.session_state.file_uploaded_name = None
if 'detections' not in st.session_state:
    st.session_state.detections = None

def detectFaceOpenCVDnn(net, image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections

def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h, frame_w = frame.shape[0], frame.shape[1]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h/200)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_AA)

    return frame, bboxes

@st.cache_resource()
def load_model():
    modelFile = "data/model/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "data/model/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

net = load_model()

if img_file_buffer is not None:
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    file_name = img_file_buffer.name

    placeholder = st.columns(2)

    placeholder[0].image(img, channels='BGR')
    placeholder[1].text("Input Image")

    conf_threshold = st.slider('SET Confidence Threshold', min_value = 0.0, max_value=1.0, step=0.01, value=0.5)

    if file_name != st.session_state.file_uploaded_name:
        st.session_state.file_uploaded_name = file_name
        st.session_state.detections = detectFaceOpenCVDnn(net, img)
        st.write("New Image Uploaded, calling the face detection model.")
    else:
        st.write("Same Image Uploaded, processing with the previous detections.")

    out_image, _ = process_detections(img, st.session_state.detections, conf_threshold)

    placeholder[1].image(out_image, channels='BGR')
    placeholder[1].text("Output Image")

    out_image = Image.fromarray(out_image[...,::-1])
    st.markdown(get_image_download_link(out_image, 'face_output.jpg', 'Download Processed Image'), unsafe_allow_html=True)
