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

def getVideoStream(video_path='analysis/Videos/vid (4).mp4'):
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

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
        while True:
            ret, img = cap.read()
            if ret == False:
                break
            skip_count += 1
            if(skip_count < 4):
                continue
            skip_count = 0
            detection, trace = detect_shot(img, trace, width, height, sess, image_tensor, boxes, scores, classes,
                                        num_detections, previous, during_shooting, shot_result, fig, shooting_pose)
            
            print("dope")
            detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83)
            frame = cv2.imencode('.jpg', detection)[1].tobytes()
            result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # yield result
    
    # print("yeaaaaa")
    # getting average shooting angle
    shooting_result['avg_elbow_angle'] = round(mean(shooting_pose['elbow_angle_list']), 2)
    shooting_result['avg_knee_angle'] = round(mean(shooting_pose['knee_angle_list']), 2)
    shooting_result['avg_release_angle'] = round(mean(during_shooting['release_angle_list']), 2)
    shooting_result['avg_ballInHand_time'] = round(mean(shooting_pose['ballInHand_frames_list']) * (4 / fps), 2)

    print("avg", shooting_result['avg_elbow_angle'])
    print("avg", shooting_result['avg_knee_angle'])
    print("avg", shooting_result['avg_release_angle'])
    print("avg", shooting_result['avg_ballInHand_time'])

    plt.title("Trajectory Fitting", figure=fig)
    plt.ylim(bottom=0, top=height)
    trajectory_path = os.path.join(
        os.getcwd(), "static/detections/trajectory_fitting.jpg")
    fig.savefig(trajectory_path)
    fig.clear()
    trace_path = os.path.join(os.getcwd(), "static/detections/basketball_trace.jpg")
    cv2.imwrite(trace_path, trace)


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


# def getAngleFromDatum(datum):
#     hipX, hipY, _ = datum.poseKeypoints[0][9]
#     kneeX, kneeY, _ = datum.poseKeypoints[0][10]
#     ankleX, ankleY, _ = datum.poseKeypoints[0][11]

#     shoulderX, shoulderY, _ = datum.poseKeypoints[0][2]
#     elbowX, elbowY, _ = datum.poseKeypoints[0][3]
#     wristX, wristY, _ = datum.poseKeypoints[0][4]

#     kneeAngle = calculateAngle(np.array([hipX, hipY]), np.array(
#         [kneeX, kneeY]), np.array([ankleX, ankleY]))
#     elbowAngle = calculateAngle(np.array([shoulderX, shoulderY]), np.array(
#         [elbowX, elbowY]), np.array([wristX, wristY]))

#     elbowCoord = np.array([int(elbowX), int(elbowY)])
#     kneeCoord = np.array([int(kneeX), int(kneeY)])
#     return elbowAngle, kneeAngle, elbowCoord, kneeCoord

def getAngleFromLandmarks(landmark1, landmark2):
    x1, y1, _ = landmark1.x, landmark1.y, landmark1.z
    x2, y2, _ = landmark2.x, landmark2.y, landmark2.z
    angle = calculateAngle(np.array([x1, y1]), np.array([x2, y2]), np.array([x2 + 1, y2]))
    return angle

def detect_shot(frame, trace, width, height, sess, image_tensor, boxes, scores, classes, num_detections, previous, during_shooting, shot_result, fig, shooting_pose):
    global shooting_result

    if(shot_result['displayFrames'] > 0):
        shot_result['displayFrames'] -= 1
    if(shot_result['release_displayFrames'] > 0):
        shot_result['release_displayFrames'] -= 1
    if(shooting_pose['ball_in_hand']):
        shooting_pose['ballInHand_frames'] += 1
    #     # print("ball in hand")
    #     ############################################################################### Replace with MediaPipe
    # # getting openpose keypoints
    # datum.cvInputData = frame
    # opWrapper.emplaceAndPop([datum])
    # try:
    #     headX, headY, headConf = datum.poseKeypoints[0][0]
    #     handX, handY, handConf = datum.poseKeypoints[0][4]
    #     elbowAngle, kneeAngle, elbowCoord, kneeCoord = getAngleFromDatum(datum)
    # except:
    #     print("Something went wrong with OpenPose")
    #     headX = 0
    #     headY = 0
    #     handX = 0
    #     handY = 0
    #     elbowAngle = 0
    #     kneeAngle = 0
    #     elbowCoord = np.array([0, 0])
    #     kneeCoord = np.array([0, 0])

    # #########################################################################################################

    ############################################################################### Replace with MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    headX = 0
    headY = 0
    handX = 0
    handY = 0
    elbowAngle = 0
    kneeAngle = 0
    elbowX = 0
    elbowY = 0
    kneeX = 0
    kneeY = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame and get the pose detection results
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extracting relevant keypoints from MediaPipe results
            head_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            hand_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            elbow_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            knee_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

            headX, headY = int(head_landmark.x * width), int(head_landmark.y * height)
            handX, handY = int(hand_landmark.x * width), int(hand_landmark.y * height)
            elbowX, elbowY = int(elbow_landmark.x * width), int(elbow_landmark.y * height)
            kneeX, kneeY = int(knee_landmark.x * width), int(knee_landmark.y * height)

            elbowAngle, kneeAngle = getAngleFromLandmarks(elbow_landmark, knee_landmark)

            # Render the landmarks on the frame
            frame = mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
            print("ya didnt work something wrong with mnpose")

            # ... Continue with the rest of the code

    #########################################################################################################

    frame_expanded = np.expand_dims(frame, axis=0)
    # main tensorflow detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # displaying openpose, joint angle and release angle
    # frame = datum.cvOutputData

    # displaying MediaPipe keypoints, joint angle and release angle
    # frame = results.pose_landmarks.render(frame, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    cv2.putText(frame, 'Elbow: ' + str(elbowAngle) + ' deg',
                (elbowX + 65, elbowY), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 0), 3)
    cv2.putText(frame, 'Knee: ' + str(kneeAngle) + ' deg',
                (kneeX + 65, kneeY), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 0), 3)
    if(shot_result['release_displayFrames']):
        cv2.putText(frame, 'Release: ' + str(during_shooting['release_angle_list'][-1]) + ' deg',
                    (during_shooting['release_point'][0] - 80, during_shooting['release_point'][1] + 80), cv2.FONT_HERSHEY_COMPLEX, 1.3, (102, 255, 255), 3)

    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
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
                        print("release angle:", release_angle)

                    #draw purple circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=7,
                               color=(235, 103, 193), thickness=3)

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




getVideoStream()
