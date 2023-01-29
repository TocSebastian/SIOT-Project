import streamlit as st
# import mediapipe as mp
import cv2
# from cv2 import VideoCapture
import numpy as np
import tempfile
import time
from PIL import Image
import torch

@st.cache
def upload_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path= 'best.pt', force_reload=True)
    return model

model = upload_model()



DEMO_IMAGE = "demo.jpg"
DEMO_VIDEO = "demo.mp4"

st.title('Face mask detection')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div: first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div: first-child{
        width: 350px
        margin-left: -350px
    }

    <style>   
    """, unsafe_allow_html=True,
)

st.sidebar.title('Mask Detection Sidebar')
st.sidebar.subheader('parameters')


@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = width / float(w)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


app_mode = st.sidebar.selectbox('Choose the App mode',
                                ['About App',  'Run on Video'])

if app_mode == 'About App':
    st.markdown('This is a maks recognition application.')
    st.markdown("---")
    st.markdown("The model used for recognition is YOLOV5, that has been trained on 1000 pictures of people with and without the mask.")

# if app_mode == 'Run on Image':
#     detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
#     st.markdown("---")
#
#     st.markdown(
#         """
#         <style>
#         [data-testid="stSidebar"][aria-expanded="true"] > div: first-child{
#             width: 350px
#         }
#         [data-testid="stSidebar"][aria-expanded="false"] > div: first-child{
#             width: 350px
#             margin-left: -350px
#         }
#
#         <style>
#         """, unsafe_allow_html=True,
#     )
#
#     img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])
#
#     if img_file_buffer is not None:
#         image = np.array(Image.open(img_file_buffer))
#
#     else:
#         demo_image = DEMO_IMAGE
#         image = np.array(Image.open(demo_image))
#
#     st.sidebar.text("Original Image")
#     st.sidebar.image(image)

if app_mode == 'Run on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)

    use_webcam = st.sidebar.button("Use Webcam")
    # record = st.sidebar.checkbox("Record Video")
    #
    # if record:
    #     st.checkbox('Recording', value=True)

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div: first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div: first-child{
            width: 350px
            margin-left: -350px
        }

        <style>   
        """, unsafe_allow_html=True,
    )

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.markdown("---")

    model.conf = detection_confidence
    counter = 0

    st.markdown("## Output")

    stframe = st.empty()
    video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    # We get our input video here

    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)

        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))


    # Recording

    # codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # out = cv2.VideoWriter('output.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**Detection**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Counter**")
        kpi2_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    prevTime = 0

    while vid.isOpened():
        ret, frame = vid.read()
        detector = model(frame)

        info = detector.pandas().xyxy[0].to_dict(orient="records")

        if len(info) != 0:
            for result in info:
                conf = result['confidence']
                clasa = result['class']
                nume = result['name']

                if conf > 0.30 and nume == 'cu_masca':
                    counter += 1
                    if counter > 5:
                        counter = 10
                        if 5 < counter <=10:
                            cv2.putText(frame, 'Masca detectata', (50, 50), cv2.LINE_AA, 1, (0, 255, 0), 2)
                            cv2.putText(frame, 'Acces autorizat', (50, 450), cv2.LINE_AA, 1, (0, 255, 0), 2)

                elif conf > 0.15 and nume == 'fara_masca':
                    counter += -1
                    if counter < -5:
                        counter = -10
                        cv2.putText(frame, 'Masca nedetectata', (50, 50), cv2.LINE_AA, 1, (0, 0, 255), 2)
                        cv2.putText(frame, 'Acces neautorizat', (50, 450), cv2.LINE_AA, 1, (0, 0, 255), 2)
                print(counter)

        else:
            cv2.putText(frame, 'Procesam datele', (50, 50), cv2.LINE_AA, 1, (255, 255, 255), 2)
            nume = None


        cv2.imshow('Detector de masca yolo v5', np.squeeze(detector.render()))

        if not ret:
            continue


        frame.flags.writeable = True

        currentTime = time.time()
        fps = 1/ (currentTime - prevTime)
        prevTime = currentTime


        # if record:
        #     out.write(frame)

        if nume is None:
            nume = 'No detection'

        kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{nume}</h1>", unsafe_allow_html=True)
        kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{int(counter)}</h1>", unsafe_allow_html=True)

        frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
        frame = image_resize(image = frame, width = 640)
        stframe.image(frame,channels = 'BGR',use_column_width=True)

    st.text('Video Processed')

    # output_video = open('output1.mp4','rb')
    # out_bytes = output_video.read()
    # st.video(out_bytes)

    vid.release()
    # out. release()
   