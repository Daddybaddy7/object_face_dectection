# Object & Face Detection

This project demonstrates real-time object and face detection using OpenCV and YOLOv3. It includes features like time-stamped recordings, facial recognition, and snapshots upon detection.

## Setup Instructions

1. Clone the repository:  
   ```bash
   git clone https://github.com/Daddybaddy7/object_face_dectection.git
   cd object_face_dectection
   ```

2. Create two folders inside the project directory:  
   - `videos`: This folder will store the time-stamped recordings.  
   - `unknown_faces`: This folder will store images of unrecognized faces.  

   You can create them manually or by running the following commands:  
   ```bash
   mkdir videos
   mkdir unknown_faces
   ```

3. Download the YOLOv3 weights and configuration files if not already included:  
   - [YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights)  
   - [YOLOv3 config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

4. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

5. Run the project:  
   ```bash
   python object_face_detection.py
   ```

## Features

- **Real-time object detection**  
- **Facial recognition**  
- **Time-stamped recordings**  
- **Snapshots upon detection**  

## Results

- **Detected Objects:** Stored in `videos/`  
- **Unrecognized Faces:** Stored in `unknown_faces/`

## Contributing

Feel free to fork this repository, create a new branch, and submit a pull request!

---
