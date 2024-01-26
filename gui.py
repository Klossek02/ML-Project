from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QDesktopWidget,
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QFont, QIcon
from PyQt5.QtCore import QTimer, Qt, QFile, QIODevice

model = load_model('age_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

win_init_height = 80
win_init_width = 1050

# function for detecting faces and eyes in a frame 
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converting image to grayscale for much better detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4) # detecting faces in the image 
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # drawing rectangle around a face, thickness 2
        
        face_roi = frame[y:y+h, x:x+w]
        # apply necessary preprocessing (resizing, scaling)
        face_roi = cv2.resize(face_roi, (128, 128))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        # predict age and put a label
        predicted_age = model.predict(face_roi)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y-30) 
        fontScale = 1
        color = (0, 255, 0) 
        thickness = 2
        cv2.putText(frame, "Age: "+str(int(predicted_age)), org, font, fontScale, color, thickness, cv2.LINE_AA)

    return frame

# function for detecting faces and eyes in an image  
def image_face_detection(image_path):
    # cv2.destroyAllWindows()
    img = cv2.imread(str(image_path)) # reading images from the path (given by the user)
    
    # error handling for image (checking if it exists)
    if img is None:
        print(f"Error: Unable to read the image at {image_path}. Check if the file exists and is accessible.")
        return

    # setting width and height of the images (as it will be displayed in a new window)
    image_width = 1000
    image_height = 1024

    # calculating the aspect ratio (to be able to resize the window on sides - horizontally )
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # calculating dimensions
    if width > height:
        dim_width = image_width
        dim_height = int(image_width / aspect_ratio)
    else:
        dim_height = image_height
        dim_width = int(image_height * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (dim_width, dim_height))

    resized_img = detect_faces(resized_img)
    
    return resized_img

# function for detecting faces and eyes on a webcam 
def webcam_face_detection(point):
    cv2.destroyAllWindows()
    cap = cv2.VideoCapture(0) # start capturing video from the webcam

    cv2.namedWindow('Face and Eye Detection - Web Camera', cv2.WINDOW_NORMAL)
    h, w = cap.get(4), cap.get(3)
    aspect_ratio = w / h
    cv2.resizeWindow('Face and Eye Detection - Web Camera', win_init_width, int(win_init_width / aspect_ratio) + 100)
    cv2.moveWindow('Face and Eye Detection - Web Camera', point.x()-8, point.y()+5)
    
    # error handling for webcam (checking if it opens)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read() # reading frame from the webcam
        if not ret:
            break
        frame = detect_faces(frame) # detecting faces in a frame 
        cv2.imshow('Face and Eye Detection - Web Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # hit 'q' on the keyboard to quit 
            break
        if cv2.getWindowProperty('Face and Eye Detection - Web Camera', cv2.WND_PROP_VISIBLE) <1: # if window is closed
            break
    cap.release()
    cv2.destroyAllWindows() # closing displayed window 

# function for detecting faces and eyes in a video file  
def video_face_detection(cap, point):
    cv2.namedWindow('Face and Eye Detection - Video', cv2.WINDOW_NORMAL)
    h, w = cap.get(4), cap.get(3)
    aspect_ratio = w / h
    cv2.resizeWindow('Face and Eye Detection - Video', win_init_width, int(win_init_width / aspect_ratio) + 100)
    cv2.moveWindow('Face and Eye Detection - Video', point.x()-8, point.y()+5)
    while cap.isOpened():
        ret, frame = cap.read() # reading frame from the video
        if not ret:
            break
        frame = detect_faces(frame) # detecting faces in the frame 
        cv2.imshow('Face and Eye Detection - Video', frame) # displaing frame with detections 
        key = cv2.waitKey(1)

        if key == 32:
            cv2.waitKey()
        elif key == ord('q'): # hit 'q' on the keyboard to quit 
            break
        if cv2.getWindowProperty('Face and Eye Detection - Video', cv2.WND_PROP_VISIBLE) <1: # if window is closed
            break
    cap.release()
    cv2.destroyAllWindows() # closing displayed window 

class AgeDetectionUI(QWidget):
    def __init__(self):
        super().__init__()
        # set window title and size
        self.setWindowTitle("Face and Age Detection")
        self.setGeometry(0, 0, win_init_width, win_init_height)
        self.setWindowIcon(
            QIcon("images.png")
        )  # path needs to be adjusted, depending on the user's one
        self.age_detection_active = False
        self.age_model = load_model('age_model.h5') # can be adjusted 

        # get the center point of the screen
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        centerPoint.setY(centerPoint.y() - 450)
        qtRectangle.moveCenter(centerPoint)

        # move the window to the center
        self.move(qtRectangle.topLeft())

        # create labels and buttons
        button_style = """
            QPushButton {
                background-color: #797EF6; /* Cornflower Blue */
                color: white; 
                padding: 4px; /* padding around text */
                border-style: outset; /* none, outset, inset */
                border-width: 0px; 
                border-radius: 10px; /* rounded corners, set to 0 for square */
                border-color: beige; 
                font: bold 14px; 
                min-width: 10em; 
                padding: 3px; /* padding inside button */
            }
            QPushButton:pressed {
                background-color: #545AA7; /* Liberty Blue */
            }
            QPushButton:hover {
                background-color: #545AA7; /* Liberty Blue */
            }
        """
        button_width = 150  # width
        button_height = 30  # height
        button_spacing = 0  # spacing

        # calculate starting x position for centering the buttons
        total_buttons_width = (button_width * 5) + (button_spacing * 4)
        start_x = (win_init_width - total_buttons_width) // 2

        self.select_image_button = QPushButton("Select Image", self)
        self.select_image_button.setGeometry(100, 20, 100, 30)
        self.select_image_button.setStyleSheet(button_style)
        self.select_image_button.setGeometry(start_x, 20, button_width, button_height)

        self.select_video_button = QPushButton("Select Video", self)
        self.select_video_button.setGeometry(250, 20, 100, 30)
        self.select_video_button.setStyleSheet(button_style)
        self.select_video_button.setGeometry(
            start_x + (button_width + button_spacing), 20, button_width, button_height
        )

        self.select_camera_button = QPushButton("Select Camera", self)
        self.select_camera_button.setGeometry(400, 20, 100, 30)
        self.select_camera_button.setStyleSheet(button_style)
        self.select_camera_button.setGeometry(
            start_x + (button_width + button_spacing) * 2,
            20,
            button_width,
            button_height,
        )

        self.save_image_button = QPushButton("Save Image", self)
        self.save_image_button.setGeometry(550, 20, 100, 30)
        self.save_image_button.setStyleSheet(button_style)
        self.save_image_button.setGeometry(
            start_x + (button_width + button_spacing) * 3,
            20,
            button_width,
            button_height,
        )

        self.save_video_button = QPushButton("Save Video", self)
        self.save_video_button.setGeometry(700, 20, 100, 30)
        self.save_video_button.setStyleSheet(button_style)
        self.save_video_button.setGeometry(
            start_x + (button_width + button_spacing) * 4,
            20,
            button_width,
            button_height,
        )

        self.select_image_button.clicked.connect(self.select_image)
        self.select_video_button.clicked.connect(self.select_video)
        self.select_camera_button.clicked.connect(self.select_camera)
        self.save_image_button.clicked.connect(self.save_image)
        self.save_video_button.clicked.connect(self.save_video)
        
        self.cap = None

    def repositionButtons(self):
        # calculate new dimensions and button positions
        button_width = 150
        button_height = 30
        button_spacing = 10

        # calculate starting x position for centering the buttons
        total_buttons_width = (button_width * 5) + (button_spacing * 4)
        start_x = (self.width() - total_buttons_width) // 2

        self.select_image_button.setGeometry(start_x, 20, button_width, button_height)
        self.select_video_button.setGeometry(
            start_x + (button_width + button_spacing), 20, button_width, button_height
        )
        self.select_camera_button.setGeometry(
            start_x + (button_width + button_spacing) * 2,
            20,
            button_width,
            button_height,
        )
        self.save_image_button.setGeometry(
            start_x + (button_width + button_spacing) * 3,
            20,
            button_width,
            button_height,
        )
        self.save_video_button.setGeometry(
            start_x + (button_width + button_spacing) * 4,
            20,
            button_width,
            button_height,
        )

    def select_image(self):
        # open file dialog to select image
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "All Files (*);;Image Files (*.png *.jpg *.jpeg)",
            options=options,
        )

        if file_path:
            self.img = image_face_detection(file_path)
            cv2.namedWindow('Face and Eye Detection - Image', cv2.WINDOW_NORMAL)
            h, w = self.img.shape[:2]
            aspect_ratio = w / h
            cv2.resizeWindow('Face and Eye Detection - Image', win_init_width, int(win_init_width / aspect_ratio) + 100)
            # get center
            qtRectangle = self.frameGeometry()
            point = qtRectangle.bottomLeft()
            cv2.moveWindow('Face and Eye Detection - Image', point.x()-8, point.y()+5)
            cv2.imshow('Face and Eye Detection - Image', self.img)
            cv2.waitKey(0)
            if cv2.getWindowProperty('Face and Eye Detection - Image', cv2.WND_PROP_VISIBLE) <1:# if window is closed
                cv2.destroyAllWindows()

    def select_video(self):
        # open file dialog to select video
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "All Files (*);;Video Files (*.mp4 *.avi *.mov)",
            options=options,
        )
        if file_path:
            cv2.destroyAllWindows()
            cap = cv2.VideoCapture(str(file_path)) # opening the video file 
            self.video_size = (int(cap.get(3)), int(cap.get(4)))

            # error handling for video file (checking if it opens)
            if not cap.isOpened():
                print(f"Error: Could not open video file at {file_path}.")
                return
            
            self.video_path = file_path
            # get center
            qtRectangle = self.frameGeometry()
            point = qtRectangle.bottomLeft()
            video_face_detection(cap, point)

    def select_camera(self):
        # get center
        qtRectangle = self.frameGeometry()
        point = qtRectangle.bottomLeft()
        webcam_face_detection(point)

    # helper function to convert to grayscale and apply histogram equalization
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    # function for saving images
    def save_image(self):
        if self.img is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "PNG Image (*.png);"
            )
            if file_path:
                cv2.imwrite(str(file_path), self.img)

    # function for saving videos
    def save_video(self):
        if self.video_path is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Video", "", "MP4 Video (*.mp4)"
            )
            if file_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    file_path,
                    fourcc,
                    30,
                    self.video_size,
                )

                # restart video capture from the beginning
                self.cap = cv2.VideoCapture(self.video_path)

                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    frame_with_detection = detect_faces(frame)

                    out.write(frame_with_detection)

                out.release()

    def keyPressEvent(self, event):
        # pause the video when the spacebar is pressed
        if event.key() == 32 and self.video_playing:
            self.video_playing = False
            self.timer.stop()

    # turn off the camera if it was on when closing
    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.age_detection_active = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.repositionButtons()

if __name__ == "__main__":
    app = QApplication([])
    age_detection_ui = AgeDetectionUI()
    age_detection_ui.show()
    app.exec_()