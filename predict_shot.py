import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import cvzone
from cvzone.ColorModule import ColorFinder
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def tensorflow_init():
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return detection_graph, image_tensor, boxes, scores, classes, num_detections

def detect_API(img):
    response = []
    height, width = img.shape[:2]
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    with tf.Session(graph=detection_graph) as sess:
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})

        for i, box in enumerate(boxes[0]):
            if (scores[0][i] > 0.5):
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                if(classes[0][i] == 1):  # basketball
                    print("basketball found")
                    response.append({
                        'class': 'Basketball',
                        'detection_detail': {
                            'confidence': float(scores[0][i]),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })
                if(classes[0][i] == 2):  # Rim
                    print("rim found")
                    response.append({
                        'class': 'Hoop',
                        'detection_detail': {
                            'confidence': float(scores[0][i]),
                            'center_coordinate': {'x': xCoor, 'y': yCoor},
                            'box_boundary': {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
                        }
                    })

# Mapping dictionary to map keypoints from Mediapipe to our Classifier model
lm_dict = {
  0:0 , 1:10, 2:12, 3:14, 4:16, 5:11, 6:13, 7:15, 8:24, 9:26, 10:28, 11:23, 12:25, 13:27, 14:5, 15:2, 16:8, 17:7,
}



def set_pose_parameters():
    mode = False 
    complexity = 1
    smooth_landmarks = True
    enable_segmentation = False
    smooth_segmentation = True
    detectionCon = 0.5
    trackCon = 0.5
    mpPose = mp.solutions.pose
    return mode,complexity,smooth_landmarks,enable_segmentation,smooth_segmentation,detectionCon,trackCon,mpPose


def get_pose (img, results, draw=True):        
        if results.pose_landmarks:
            if draw:
                mpDraw = mp.solutions.drawing_utils
                mpDraw.draw_landmarks(img,results.pose_landmarks,
                                           mpPose.POSE_CONNECTIONS) 
        return img

def get_position(img, results, height, width, draw=True ):
        landmark_list = []
        if results.pose_landmarks:
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                #finding height, width of the image printed
                height, width, c = img.shape
                #Determining the pixels of the landmarks
                landmark_pixel_x, landmark_pixel_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, landmark_pixel_x, landmark_pixel_y])
                if draw:
                    cv2.circle(img, (landmark_pixel_x, landmark_pixel_y), 5, (255,0,0), cv2.FILLED)
        return landmark_list    

# 11, 23, 25
def get_angle(img, landmark_list, point1, point2, point3, lefty, draw=True):   
        #Retrieve landmark coordinates from point identifiers

        point1setlefty = {11, 23, 13}
        point2setlefty = {13, 25, 23, 11}
        point3setlefty = {15, 27, 25, 23}

        point1setrighty = {12, 14, 24}
        point2setrighty = {12, 14, 24, 26}
        point3setrighty = {16, 24, 26, 28}

        point1set = None
        point2set = None
        point3set = None
    
        x1, y1 = landmark_list[point1][1:]
        x2, y2 = landmark_list[point2][1:]
        x3, y3 = landmark_list[point3][1:]
            
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                             math.atan2(y1-y2, x1-x2))
        
        #Handling angle edge cases: Obtuse and negative angles
        if angle < 0:
            angle += 360
            if angle > 180:
                angle = 360 - angle
        elif angle > 180:
            angle = 360 - angle

        """# Handling hip angle
        if point1 == 11 and point2 == 23 and point3 == 25:
            x1, y1 = landmark_list[point1][1:]
            x2, y2 = landmark_list[point2][1:]
            x3, y3 = x2, 0
            
            angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))"""

        if draw:
            match lefty:
                case True:
                    point1set = point1setlefty
                    point2set = point2setlefty
                    point3set = point3setlefty
                case False:
                    point1set = point1setrighty
                    point2set = point2setrighty
                    point3set = point3setrighty

            if point1 in point1set and point2 in point2set and point3 in point3set:
                cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255,255,255), 3)

                #Drawing circles at intersection points of lines
                cv2.circle(img, (x1, y1), 5, (75,0,130), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (75,0,130), 2)
                cv2.circle(img, (x2, y2), 5, (75,0,130), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (75,0,130), 2)
                cv2.circle(img, (x3, y3), 5, (75,0,130), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (75,0,130), 2)
                
                #Show angles between lines
                cv2.putText(img, str(int(angle)), (x2-50, y2+50), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        return angle


# Setting variables for video feed
def set_video_feed_variables():
    cap = cv2.VideoCapture("videos/IMG_0654.MOV")
    count = 0
    direction = 0
    form = 0
    feedback = "Bad Form."
    frame_queue = deque(maxlen=250)
    return cap,count,direction,form,feedback,frame_queue


def set_percentage_bar_and_text(elbow_angle, knee_angle):
    success_percentage = np.interp(knee_angle, (90, 160), (0, 100))
    progress_bar = np.interp(knee_angle, (90, 160), (380, 30))
    return success_percentage,progress_bar

def set_body_angles_from_keypoints(get_angle, img, landmark_list, lefty):
    elbow_angle = get_angle(img, landmark_list, 11, 13, 15, lefty)
    shoulder_angle = get_angle(img, landmark_list, 13, 11, 23, lefty)
    hip_angle = get_angle(img, landmark_list, 11, 23, 25, lefty)
    elbow_angle_right = get_angle(img, landmark_list, 12, 14, 16, lefty)
    shoulder_angle_right = get_angle(img, landmark_list, 14, 12, 24, lefty)
    hip_angle_right = get_angle(img, landmark_list, 12, 24, 26, lefty)
    knee_angle = get_angle(img, landmark_list, 23, 25, 27, lefty)
    return elbow_angle,shoulder_angle,hip_angle,elbow_angle_right,shoulder_angle_right,hip_angle_right,knee_angle

def draw_percentage_progress_bar(knee_form, img, success_percentage, progress_bar):
    xd, yd, wd, hd = 10, 175, 50, 200
    
    cv2.rectangle(img, (xd,30), (xd+wd, yd+hd), (0, 255, 0), 3)
    cv2.putText(img, f'{0}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)
    if knee_form == 1:
        cv2.rectangle(img, (xd, int(progress_bar)), (xd+wd, yd+hd), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(success_percentage)}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)

def show_workout_feedback(feedback, img):    
    xf, yf = 85, 70
    cv2.putText(img, feedback, (xf, yf), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0,0,0), 2)

def display_workout_stats(count, form, feedback, draw_percentage_progress_bar, show_workout_feedback, img, success_percentage, progress_bar):
    #Draw the progress bar
    draw_percentage_progress_bar(form, img, success_percentage, progress_bar)
        
    #Show the feedback 
    show_workout_feedback(feedback, img)


def main():
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks,
                                enable_segmentation, smooth_segmentation,
                                detectionCon, trackCon)


    # Setting video feed variables
    cap, count, direction, form, feedback, frame_queue = set_video_feed_variables()

    # shooter handedness
    lefty = False

    #Start video feed and run workout
    knee_form = 0
    while cap.isOpened():
        #Getting image from camera
        ret, img = cap.read()
        
        #Getting video dimensions
        width  = cap.get(3)  
        height = cap.get(4)  
        
        #Convert from BGR (used by cv2) to RGB (used by Mediapipe)
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        #Get pose and draw landmarks
        img = get_pose(img, results, False)
        
        # Get landmark list from mediapipe
        landmark_list = get_position(img, results, height, width, False)
        
        #If landmarks exist, get the relevant workout body angles and run workout. The points used are identifiers for specific joints
        if len(landmark_list) != 0:
            elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle = set_body_angles_from_keypoints(get_angle, img, landmark_list, lefty)

            #print("shoulder angle: ", shoulder_angle)
            #print("the hip angle is:", hip_angle)

            #Is the form correct at the start?
            success_percentage, progress_bar = set_percentage_bar_and_text(elbow_angle, knee_angle)
            
            #Full workout motion
            if knee_angle < 100 and knee_form == 0:
                knee_form = 1
                print(type(success_percentage), progress_bar)
            if elbow_angle > 45 and elbow_angle < 60:
                feedback = "Feedback: Correct posture for a perfect shot"
            elif elbow_angle > 60 and elbow_angle < 90:
                feedback = "Feedback: Correct posture for a good shot"
            elif elbow_angle > 120:
                feedback = "Feedback: Bend your elbows to make a 45 degree"

            #Display workout stats        
            display_workout_stats(count, knee_form, feedback, draw_percentage_progress_bar, show_workout_feedback, img, success_percentage, progress_bar)
            
            
        # Transparent Overlay
        overlay = img.copy()
        x, y, w, h = 75, 10, 900, 100
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)      
        alpha = 0.75  # Transparency factor.
        # Following line overlays transparent rectangle over the image
        image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)          
            
        cv2.imshow('Basketball Form GOAT', image_new)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
