import cv2
import numpy as np

def detect_and_draw_bounding_box(video_path):
    # Load YOLO model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    classes = []
    with open("9k.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Get output layer names
    output_layers_names = net.getUnconnectedOutLayersNames()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # Convert the frame to blob for the YOLO model
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # Run forward pass and get predictions
        detections = net.forward(output_layers_names)

        # Loop over the detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "basketball":
                    # YOLO returns coordinates normalized between 0 and 1, so we need to scale them
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)

                    # Calculate coordinates for the bounding box
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Basketball Detection', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Replace 'your_video_path.mp4' with the path to your video file
detect_and_draw_bounding_box('videos/IMG_0654.MOV')
