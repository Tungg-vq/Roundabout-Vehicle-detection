import streamlit as st
import cv2
import tempfile
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Roundabout detecting system", layout="wide")

@st.cache_resource
def load_model(model_path='best.pt'):

    return YOLO(model_path)


st.sidebar.title("System configuration")
model_path = st.sidebar.text_input("Model path (.pt)", "best.pt")
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
task = st.sidebar.selectbox("Select operation mode", ["Image Detection", "Video Tracking"])

# Load model
try:
    model = load_model(model_path)
    st.sidebar.success("model loaded sucessfully")
except Exception as e:
    st.sidebar.error(f"Error loading model: Check the path again!\n{e}")
    st.stop()

st.title("Roundabout Detecting System")


#  Xử lí ảnh
if task == "Image Detection":
    st.header("Image Detection")
    uploaded_img = st.file_uploader("Upload Image (jpg, png, jpeg)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            
        if st.button("Start Detection"):
            with st.spinner('Processing...'):
                results = model.predict(image, conf=conf_thres)
                res_img = results[0].plot(conf=False)
                
            with col2:
                st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="Detection Results", use_container_width=True)

# Xử lí video
elif task == "Video Tracking":
    st.header("Video Tracking")
    uploaded_vid = st.file_uploader("Upload Video (mp4, avi)", type=['mp4', 'avi', 'mov'])
    
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_vid.read())
        
        if st.button("Start Tracking"):
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty() 
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model.predict(frame, conf=conf_thres)
                res_frame = results[0].plot(conf= False)
                
                stframe.image(cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            cap.release()
            st.success("Completed video processing!")
