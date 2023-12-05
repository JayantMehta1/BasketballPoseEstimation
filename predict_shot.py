from collections import deque
import math
import cvzone
from cvzone.ColorModule import ColorFinder
import time
from absl import app, logging
import cv2
import mediapipe as mp
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statistics import mean
tf.disable_v2_behavior()


shooting_result = {
    'attempts': 0,
    'made': 0,
    'miss': 0,
    'avg_elbow_angle': 0,
    'avg_knee_angle': 0,
    'avg_release_angle': 0,
    'avg_ballInHand_time': 0
}

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
    cap = cv2.VideoCapture("analysis/Videos/one_score_one_miss.mp4")
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
    knee_angle_right = get_angle(img, landmark_list, 24, 26, 28, lefty)
    ankle_angle = get_angle(img, landmark_list, 25, 27, 31, lefty)
    ankle_angle_right = get_angle(img, landmark_list, 26, 28, 32, lefty)
    return elbow_angle,shoulder_angle,hip_angle,elbow_angle_right,shoulder_angle_right,hip_angle_right,knee_angle, knee_angle_right, ankle_angle, ankle_angle_right

def draw_percentage_progress_bar(knee_form, img, success_percentage, progress_bar):
    xd, yd, wd, hd = 10, 175, 50, 200
    
    cv2.rectangle(img, (xd,30), (xd+wd, yd+hd), (0, 255, 0), 3)
    cv2.putText(img, f'{0}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)
    if knee_form == 1:
        cv2.rectangle(img, (xd, int(progress_bar)), (xd+wd, yd+hd), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(success_percentage)}%', (xd, yd+hd+50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 0), 2)

'''def show_workout_feedback(feedback, img):    
    xf, yf = 85, 70
    cv2.putText(img, feedback, (xf, yf), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0,0,0), 2)

def display_workout_stats(count, form, feedback, draw_percentage_progress_bar, show_workout_feedback, img, success_percentage, progress_bar):
    #Draw the progress bar
    draw_percentage_progress_bar(form, img, success_percentage, progress_bar)
        
    #Show the feedback 
    show_workout_feedback(feedback, img)'''


def fit_func(x, a, b, c):
    return a*(x ** 2) + b * x + c


def trajectory_fit(balls, height, width, shotJudgement, fig):
    x = [ball[0] for ball in balls]
    y = [height - ball[1] for ball in balls]

    try:
        params = curve_fit(fit_func, x, y)
        [a, b, c] = params[0]   
    except:
        print("fitting error")
        a = 0
        b = 0
        c = 0
    x_pos = np.arange(0, width, 1)
    y_pos = [(a * (x_val ** 2)) + (b * x_val) + c for x_val in x_pos]

    if(shotJudgement == "MISS"):
        plt.plot(x, y, 'ro', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='red',
                 alpha=0.4, linewidth=5, figure=fig)
    else:
        plt.plot(x, y, 'go', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='green',
                 alpha=0.4, linewidth=5, figure=fig)

def distance(x, y):
    return ((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2) ** (1/2)


def calculateAngle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return round(np.degrees(angle), 2)



def detect_shot(frame, trace, width, height, sess, image_tensor, boxes, scores, classes,
                num_detections, previous, during_shooting, shot_result, fig, shooting_pose,
                head_landmark, right_hand_landmark, elbow_angle, elbow_angle_right, knee_angle, 
                knee_angle_right, right_elbow_landmark, left_elbow_landmark, right_knee_landmark, 
                left_knee_landmark, left_hand_landmark, landmark_list, hip_angle, hip_angle_right, 
                ankle_angle, ankle_angle_right, shoulder_angle, shoulder_angle_right, right_shoulder_landmark,
                left_shoulder_landmark, kneeToAnkle, heelToToe, lefty):

    
    global shooting_result

    if(shot_result['displayFrames'] > 0):
        shot_result['displayFrames'] -= 1
    if(shot_result['release_displayFrames'] > 0):
        shot_result['release_displayFrames'] -= 1
    if(shooting_pose['ball_in_hand']):
        shooting_pose['ballInHand_frames'] += 1
    #     # print("ball in hand")


    headX, headY = head_landmark[1:]
    handX, handY = left_hand_landmark[1:]
    elbowAngle = elbow_angle
    kneeAngle = knee_angle
    elbowX, elbowY = right_elbow_landmark[1:]
    kneeX, kneeY = right_knee_landmark[1:]
    shoulderX, shoulderY = left_shoulder_landmark[1:]
    toeX, toeY = 0,0
    heelX, heelY = 0,0

    frame_expanded = np.expand_dims(frame, axis=0)
    # main tensorflow detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # displaying MediaPipe keypoints, joint angle and release angle
    # frame = results.pose_landmarks.render(frame, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    #cv2.putText(frame, 'Elbow: ' + str(elbowAngle) + ' deg',
                #(elbowX + 65, elbowY), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 0), 3)
    #cv2.putText(frame, 'Knee: ' + str(kneeAngle) + ' deg',
                #(kneeX + 65, kneeY), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 0), 3)

    # ball height is ankle y - ball y
    # elbow height is ankle y - elbow y


    if lefty == True:
        ankleX, ankleY = landmark_list[29][1:]
        handX, handY = left_hand_landmark[1:]
        elbowX, elbowY = left_elbow_landmark[1:]
        elbowAngle = elbow_angle
        knee_angle = knee_angle
        kneeX, kneeY = left_knee_landmark[1:]
        shoulderX, shoulderY = left_shoulder_landmark[1:]
        toeX, toeY = landmark_list[31][1:]
        heelX, heelY = landmark_list[29][1:]
    else:
        ankleX, ankleY = landmark_list[29][1:]
        handX, handY = right_hand_landmark[1:]
        elbowX, elbowY = right_elbow_landmark[1:]
        elbowAngle = elbow_angle_right
        knee_angle = knee_angle_right
        kneeX, kneeY = right_knee_landmark[1:]
        shoulderX, shoulderY = right_shoulder_landmark[1:]
        toeX, toeY = landmark_list[32][1:]
        heelX, heelY = landmark_list[20][1:]

    #actual distance knee to ankle / pixel distance knee to ankle
    kneeToAnklePixel = math.sqrt((kneeY - ankleY) ** 2 + (kneeX - ankleX) ** 2)
    verticalRatio = 0.002025

    

    #actual distance heel to toe / pixel distance heel to toe
    heelToToePixel = math.sqrt((heelY - toeY) ** 2 + (heelX - toeX) ** 2)
    horizontalRatio = 0.002025

    #ball position:
    #print("x: " + str(previous['ball'][0]) + " y: " + str(previous['ball'][1]))


    #pre shot angles:
    '''if lefty == True:
        print("release hand height is: ", verticalRatio * (toeY - handY))

        if handY - shoulderY <= 10 and during_shooting['isShooting'] == False:
            print(str(hip_angle) + ", " + str(knee_angle) + ", " + str(ankle_angle) + ", " + str(elbow_angle) + ", " + str(shoulder_angle) + ", " + "0,0,0")
            print("pre shot elbow height is: ", verticalRatio * (toeY - elbowY))

    else:
        print("release hand height is: ", verticalRatio * (toeY - handY))

        if handY - shoulderY <= 10 and during_shooting['isShooting'] == False:
            print(str(hip_angle_right) + ", " + str(knee_angle_right) + ", " + str(ankle_angle_right) + ", " + str(elbow_angle_right) + ", " + str(shoulder_angle_right) + ", " + "0,0,0")
            print("pre shot elbow height is: ", verticalRatio * (toeY - elbowY))'''


    if(shot_result['release_displayFrames']):
        cv2.putText(frame, 'Release: ' + str(during_shooting['release_angle_list'][-1]) + ' deg',
                    (during_shooting['release_point'][0] - 80, during_shooting['release_point'][1] + 80), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 255), 3)

        '''# ball height at release
        ballHeight = ankleY - during_shooting['release_point'][1]

        cv2.putText(frame, "Release Height: " + str(ballHeight) + ' pixels', (during_shooting['release_point'][0] - 80, during_shooting['release_point'][1] - 80), cv2.FONT_HERSHEY_PLAIN, 2,
                    (82, 168, 50), 3)
        
        elbowHeight = ankleY - elbowY

        # elbow height at release
        cv2.putText(frame, 'Elbow: ' + str(elbowHeight) + ' pixels',
                (elbowX + 65, elbowY), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 0), 3)
'''
    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.2):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            # Basketball (not head)
            if(classes[0][i] == 1 and (distance([headX, headY], [xCoor, yCoor]) > 30)):

                # recording shooting pose
                if(distance([xCoor, yCoor], [handX, handY]) < 120):
                    shooting_pose['ball_in_hand'] = True
                    shooting_pose['knee_angle'] = min(
                        shooting_pose['knee_angle'], kneeAngle)
                    shooting_pose['elbow_angle'] = min(
                        shooting_pose['elbow_angle'], elbowAngle)
                else:
                    shooting_pose['ball_in_hand'] = False

                # During Shooting
                if(ymin < (previous['hoop_height'])):
                    if(not during_shooting['isShooting']):
                        during_shooting['isShooting'] = True

                    during_shooting['balls_during_shooting'].append(
                        [xCoor, yCoor])

                    #calculating release angle
                    if(len(during_shooting['balls_during_shooting']) == 2):
                        first_shooting_point = during_shooting['balls_during_shooting'][0]
                        release_angle = calculateAngle(np.array(during_shooting['balls_during_shooting'][1]), np.array(
                            first_shooting_point), np.array([first_shooting_point[0] + 1, first_shooting_point[1]]))
                        if(release_angle > 90):
                            release_angle = 180 - release_angle
                        during_shooting['release_angle_list'].append(
                            release_angle)
                        during_shooting['release_point'] = first_shooting_point
                        shot_result['release_displayFrames'] = 30
                        #print("release angle:", release_angle)
                        ballHeight = verticalRatio * (toeY - during_shooting['release_point'][1])
                        #print("ball to floor height: ", ballHeight)

                        ballHorizontalDist = horizontalRatio * (toeX - during_shooting['release_point'][0])
                        #print("ball horizontal dist: ", ballHorizontalDist)

                        #elbowHeight = ankleY - elbowY
                        #print("elbow height: " + str(elbowHeight) + " pixels")

                    #draw purple circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)
                    
                    #ball position:
                    print(str(xCoor) + ", " + str(yCoor))

                # Not shooting
                elif(ymin >= (previous['hoop_height'] - 30) and (distance([xCoor, yCoor], previous['ball']) < 100)):
                    # the moment when ball go below basket
                    if(during_shooting['isShooting']):
                        if(xCoor >= previous['hoop'][0] and xCoor <= previous['hoop'][2]):  # shot
                            shooting_result['attempts'] += 1
                            shooting_result['made'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "SCORE"
                            print("SCORE")
                            # draw green trace when miss
                            points = np.asarray(
                                during_shooting['balls_during_shooting'], dtype=np.int32)
                            cv2.polylines(trace, [points], False, color=(
                                82, 168, 50), thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(82, 168, 50), thickness=-1)
                        else:  # miss
                            shooting_result['attempts'] += 1
                            shooting_result['miss'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "MISS"
                            print("miss")
                            # draw red trace when miss
                            points = np.asarray(
                                during_shooting['balls_during_shooting'], dtype=np.int32)
                            cv2.polylines(trace, [points], color=(
                                0, 0, 255), isClosed=False, thickness=2, lineType=cv2.LINE_AA)
                            for ballCoor in during_shooting['balls_during_shooting']:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(0, 0, 255), thickness=-1)

                        # reset all variables
                        trajectory_fit(
                            during_shooting['balls_during_shooting'], height, width, shot_result['judgement'], fig)
                        during_shooting['balls_during_shooting'].clear()
                        during_shooting['isShooting'] = False
                        shooting_pose['ballInHand_frames_list'].append(
                            shooting_pose['ballInHand_frames'])
                        print("ball in hand frames: ",
                              shooting_pose['ballInHand_frames'])
                        shooting_pose['ballInHand_frames'] = 0

                        print("elbow angle: ", shooting_pose['elbow_angle'])
                        print("knee angle: ", shooting_pose['knee_angle'])
                        shooting_pose['elbow_angle_list'].append(
                            shooting_pose['elbow_angle'])
                        shooting_pose['knee_angle_list'].append(
                            shooting_pose['knee_angle'])
                        shooting_pose['elbow_angle'] = 370
                        shooting_pose['knee_angle'] = 370

                    #draw blue circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)

                previous['ball'][0] = xCoor
                previous['ball'][1] = yCoor

            if(classes[0][i] == 2):  # Rim
                # cover previous hoop with white rectangle
                cv2.rectangle(
                    trace, (previous['hoop'][0], previous['hoop'][1]), (previous['hoop'][2], previous['hoop'][3]), (255, 255, 255), 5)
                cv2.rectangle(frame, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)
                cv2.rectangle(trace, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)

                #display judgement after shot
                if(shot_result['displayFrames']):
                    if(shot_result['judgement'] == "MISS"):
                        cv2.putText(frame, shot_result['judgement'], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 8)
                    else:
                        cv2.putText(frame, shot_result['judgement'], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (82, 168, 50), 8)

                previous['hoop'][0] = xmin
                previous['hoop'][1] = ymax
                previous['hoop'][2] = xmax
                previous['hoop'][3] = ymin
                previous['hoop_height'] = max(ymin, previous['hoop_height'])

    combined = np.concatenate((frame, trace), axis=1)
    return combined, trace


def main():
    mode, complexity, smooth_landmarks, enable_segmentation, smooth_segmentation, detectionCon, trackCon, mpPose = set_pose_parameters()
    pose = mpPose.Pose(mode, complexity, smooth_landmarks,
                                enable_segmentation, smooth_segmentation,
                                detectionCon, trackCon)


    # Setting video feed variables
    cap, count, direction, form, feedback, frame_queue = set_video_feed_variables()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    # Tensorflow initialization for ball tracking
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    # shooter handedness
    lefty = True

    # knee to ankle distance (m)
    kneeToAnkle = 0.51

    # heel to toe distance (m)
    heelToToe = 0.24

    #Start video feed and run workout
    knee_form = 0

    fig = plt.figure()
    #objects to store detection status
    previous = {
    'ball': np.array([0, 0]),  # x, y
    'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
        'hoop_height': 0
    }
    during_shooting = {
        'isShooting': False,
        'balls_during_shooting': [],
        'release_angle_list': [],
        'release_point': []
    }
    shooting_pose = {
        'ball_in_hand': False,
        'elbow_angle': 370,
        'knee_angle': 370,
        'ballInHand_frames': 0,
        'elbow_angle_list': [],
        'knee_angle_list': [],
        'ballInHand_frames_list': []
    }
    shot_result = {
        'displayFrames': 0,
        'release_displayFrames': 0,
        'judgement': ""
    }

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.36

    skip_count = 0
    with tf.Session(graph=detection_graph, config=config) as sess:
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
                elbow_angle, shoulder_angle, hip_angle, elbow_angle_right, shoulder_angle_right, hip_angle_right, knee_angle, knee_angle_right, ankle_angle, ankle_angle_right = set_body_angles_from_keypoints(get_angle, img, landmark_list, lefty)

                # Elbow, knee, head, and hand coordinates
                left_elbow_index = 13
                left_knee_index = 25
                right_elbow_index = 14
                right_knee_index = 26
                head_index = 0
                right_hand_index = 16
                left_hand_index = 15
                right_shoulder_index = 12
                left_shoulder_index = 11
                left_elbow_landmark = landmark_list[left_elbow_index]
                left_knee_landmark = landmark_list[left_knee_index]
                right_elbow_landmark = landmark_list[right_elbow_index]
                right_knee_landmark = landmark_list[right_knee_index]
                head_landmark = landmark_list[head_index]
                right_hand_landmark = landmark_list[right_hand_index]
                left_hand_landmark = landmark_list[left_hand_index]
                right_shoulder_landmark = landmark_list[right_shoulder_index]
                left_shoulder_landmark = landmark_list[left_shoulder_index]

                # print(right_knee_landmark)

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


                # # Start the detection here
                detection, trace = detect_shot(img, trace, width, height, sess, image_tensor, boxes, scores, classes,
                                            num_detections, previous, during_shooting, shot_result, fig, shooting_pose,
                                            head_landmark, right_hand_landmark, elbow_angle, elbow_angle_right, knee_angle, 
                                            knee_angle_right, right_elbow_landmark, left_elbow_landmark, right_knee_landmark, 
                                            left_knee_landmark, left_hand_landmark, landmark_list, hip_angle, hip_angle_right, 
                                            ankle_angle, ankle_angle_right, shoulder_angle, shoulder_angle_right, right_shoulder_landmark, left_shoulder_landmark, kneeToAnkle, heelToToe, lefty)

                ## Finish the detection here

                #Display workout stats        
                #display_workout_stats(count, knee_form, feedback, draw_percentage_progress_bar, show_workout_feedback, img, success_percentage, progress_bar)
                
                
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