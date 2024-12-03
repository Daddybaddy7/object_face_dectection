import cv2
import numpy as np
from datetime import datetime
import os

# Create folders if they don't exist
if not os.path.exists('unknown_faces'):
    os.makedirs('unknown_faces')

if not os.path.exists('videos'):
    os.makedirs('videos')

# Load pre-trained object detection model and labels
net = cv2.dnn.readNet("C:/Users/91738/Downloads/yolov3.weights", "C:/Users/91738/Downloads/yolov3.cfg")

with open("C:/Users/91738/Downloads/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam and video writer
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out_video = None
recording = False

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape
    
    # Prepare image for model
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected boxes, confidences, and class IDs
    class_ids = []
    confidences = []
    boxes = []
    detected_person = False  # Flag to check if any person is detected
    
    # Loop over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Detect all objects but check if it's a person
            if confidence > 0.5:  # Minimum confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # Check if it's a person (class_id = 0)
                if class_id == 0:  # Person detected
                    detected_person = True

    # Non-max suppression to eliminate duplicate detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {round(confidence * 100, 2)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # If a person is detected, capture image and start/continue recording video
    if detected_person:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save a snapshot of the current frame in the 'unknown_faces' folder
        cv2.imwrite(f"unknown_faces/detected_person_{timestamp}.jpg", frame)
        
        # Start recording if not already recording
        if not recording:
            video_filename = f"videos/detected_persons_{timestamp}.avi"
            out_video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))
            recording = True
            print(f"Started recording: {video_filename}")
        
        # Write the frame to the video file
        if out_video is not None:
            out_video.write(frame)
    else:
        # Stop recording if no persons are detected and we were recording
        if recording:
            out_video.release()
            recording = False
            print("Stopped recording.")

    # Display the frame with detection
    cv2.imshow("Object Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
if out_video is not None:
    out_video.release()
cv2.destroyAllWindows()
