import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import cv2
import pandas as pd
import face_recognition
import os

global PATH_DATA

with st.sidebar:
    choose = option_menu(
        "App Gallery",
        ["About", "Contact", "Run on picture"],
        icons=["house", "person lines fill", "camera fill"],
        menu_icon="app-indicator",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#02ab21"},
        },
    )

if choose == "About":
    col1, col2 = st.columns([0.8, 0.2])
    with col1:  # To display the header text using css style
        st.markdown(
            """ <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="font">About the Application</p>', unsafe_allow_html=True)

        st.markdown(
            "We used **OpenCV** and the **face_recognition** module in this application to create a **face recognition** application. **Streamlit** is used to create the web **Graphical User Interface (GUI)**."
        )
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="font">About Me</p>', unsafe_allow_html=True)
    st.markdown(
        """
            Hey, I'm **Aham Gupta** . \n
            
            Check me out on:
            - [LinkedIn](https://www.linkedin.com/in/aham-gupta-18a02a202/)
            - [Orcid](https://orcid.org/0009-0009-9274-8078)
            - [GitHub](https://github.com/aham-18113) 
            """
    )

elif choose == "Contact":
    st.markdown(
        """ <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """,
        unsafe_allow_html=True,
    )
    st.markdown('<p class="font">Contact Form</p>', unsafe_allow_html=True)
    with st.form(
        key="columns_in_form2", clear_on_submit=True
    ):  # set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        Name = st.text_input(label="Please enter your name")  # Collect user feedback
        Email = st.text_input(label="Please enter your email")  # Collect user feedback
        Message = st.text_input(
            label="Please enter your message"
        )  # Collect user feedback
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(
                "Thanks for contacting us. We will respond to your questions or inquiries as soon as possible!"
            )

elif choose == "Run on picture":
    col1, col2 = st.columns([0.8, 0.2])
    with col1:  # To display the header text using css style
        st.markdown(
            """ <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """,
            unsafe_allow_html=True,
        )
        st.markdown('<p class="font">Face Recognition App</p>', unsafe_allow_html=True)

    # CONSTANTS
    PATH_DATA = "./data/DB.csv"
    COLOR_DARK = (0, 0, 153)
    COLOR_WHITE = (255, 255, 255)
    COLS_INFO = ["name", "description"]
    COLS_ENCODE = [f"v{i}" for i in range(128)]
    WEBCAMNUM = 0  # from videocapture_index_check.py


def init_data(data_path=PATH_DATA):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)


# convert image from opened file to np.array
def byte_to_array(image_in_byte):
    return cv2.imdecode(np.frombuffer(image_in_byte.read(), np.uint8), cv2.IMREAD_COLOR)


# convert opencv BRG to regular RGB mode
def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)


# convert face distance to similirity likelyhood
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = 1.0 - face_match_threshold
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))


if __name__ == "__main__":
    # disable warning signs:
    st.set_option("deprecation.showfileUploaderEncoding", False)

    # displays a file uploader widget and return to BytesIO
    image_byte = st.file_uploader(
        label="Select a picture containing faces:", type=["jpg", "png", "jpeg"]
    )
    # detect faces in the loaded image
    max_faces = 0
    rois = []  # region of interests (arrays of face areas)
    if image_byte is not None:
        image_array = byte_to_array(image_byte)
        face_locations = face_recognition.face_locations(image_array)
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            # save face region of interest to list
            rois.append(image_array[top:bottom, left:right].copy())

            # Draw a box around the face and lable it
            cv2.rectangle(image_array, (left, top), (right, bottom), COLOR_DARK, 2)
            cv2.rectangle(
                image_array,
                (left, bottom + 35),
                (right, bottom),
                COLOR_DARK,
                cv2.FILLED,
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                image_array,
                f"#{idx}",
                (left + 5, bottom + 25),
                font,
                0.55,
                COLOR_WHITE,
                1,
            )

        st.image(BGR_to_RGB(image_array), width=720)
        max_faces = len(face_locations)

    if max_faces > 0:
        # select interested face in picture
        face_idx = st.selectbox("Select face#", range(max_faces))
        roi = rois[face_idx]
        st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

        # initial database for known faces
        DB = init_data()
        face_encodings = DB[COLS_ENCODE].values
        dataframe = DB[COLS_INFO]

        # compare roi to known faces, show distances and similarities
        face_to_compare = face_recognition.face_encodings(roi)[0]
        dataframe["distance"] = face_recognition.face_distance(
            face_encodings, face_to_compare
        )
        dataframe["similarity"] = dataframe.distance.apply(
            lambda distance: f"{face_distance_to_conf(distance):0.2%}"
        )
        st.dataframe(dataframe.sort_values("distance").iloc[:5].set_index("name"))

        # add roi to known database
        if st.checkbox("add it to knonwn faces"):
            face_name = st.text_input("Name:", "")
            face_des = st.text_input("Desciption:", "")
            if st.button("add"):
                encoding = face_to_compare.tolist()
                DB.loc[len(DB)] = [face_name, face_des] + encoding
                DB.to_csv(PATH_DATA, index=False)
    else:
        st.write("No human face detected.")

    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.markdown(
            '<p class="font">Webcam Face Recognition</p>', unsafe_allow_html=True
        )

FRAME_WINDOW = st.image([])


@st.cache_data
def load_known_data():
    DB = pd.read_csv(PATH_DATA)
    return (DB["name"].values, DB[COLS_ENCODE].values)


def capture_face(video_capture):
    # got 3 frames to auto adjust webcam light
    for i in range(3):
        video_capture.read()

    while True:
        ret, frame = video_capture.read()
        FRAME_WINDOW.image(frame[:, :, ::-1])
        # face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if len(face_locations) > 0:
            video_capture.release()
            return frame


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = 1.0 - face_match_threshold
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))


def recognize_frame(frame):
    # convert COLOR_BGR2RGB
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        # Draw a box around the face
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )

        best_match_index = np.argmin(face_distances)
        name = known_face_names[best_match_index]
        similarity = face_distance_to_conf(face_distances[best_match_index], 0.5)
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_DARK, 2)
        return name, similarity, np.ascontiguousarray(frame[:, :, ::-1])


if __name__ == "__main__":
    while True:
        known_face_names, known_face_encodings = load_known_data()
        video_capture = cv2.VideoCapture(WEBCAMNUM)
        frame = capture_face(video_capture)
        name, similarity, frame = recognize_frame(frame)
        FRAME_WINDOW.image(frame)
        if similarity > 0.75:
            label = f"**{name}**: *{similarity:.2%} likely*"
            st.markdown(label)
            break
    # press to restart the scripts
    st.button("Continue...")
