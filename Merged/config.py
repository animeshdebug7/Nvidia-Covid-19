## import requirements
import math
import random
import os
import cv2
import numpy as np
import time
from itertools import combinations
from random import randrange

# =============================Default Values==========================================
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 90

# ===========================Calculation and Detection==================================
def cal_distance(p1, p2):
    """This function measures the 
       Euclidean distance"""
    x_dist = p1[0]-p2[0]
    y_dist = p1[1]-p2[1]
    dist = np.sqrt(x_dist**2 + y_dist**2) 
    return dist

def detect_people(frame, net, ln, personIdx=0, min_conf=MIN_CONF, nms_thre=NMS_THRESH):
    """Function for detecting person in a frame.
       it will return detected person's coordinates with its centroid """
    # grab the dimensions of the frame and initialize the list of
    (H, W) = frame.shape[:2]
    results = []   

    # detection
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    centroid_dict = dict()  # Function creates a dictionary and calls it centroid_dict

    # results
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            # extract the class ID and confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > min_conf:
                # scale the bounding box coordinates back 
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
        
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_thre)

    # ensure at least one detection exists
    if len(idxs) > 0:
        objectId = 0
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            centroid_dict[objectId] = (centroids[i][0], centroids[i][1], x, y, x + w, y + h)
            objectId += 1 #Increment the index for each detection 

    return centroid_dict

def violation(centroid_dict, min_dist= MIN_DISTANCE):
    """This function calculate if two or more person is
       closer than threshold distance"""
    red_zone = {} # List containing which Object id is in under threshold distance condition. 
    yellow_zone = {}
    for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections
        distance = cal_distance(p1, p2) 			# Calculates the Euclidean distance
        if distance < min_dist:				# Set our social distance threshold - If they meet this condition then..
            if id1 not in red_zone:
                red_zone[id1] = p1[0:2]   
            if id2 not in red_zone:
                red_zone[id2] = p2[0:2]
                
        if distance < min_dist+30:
            if id1 not in yellow_zone:
                yellow_zone[id1] = p1[0:2]   
            if id2 not in yellow_zone:
                yellow_zone[id2] = p2[0:2]
    return red_zone, yellow_zone

def zone_detector(centroid_dict, min_dist=MIN_DISTANCE):
    ce = centroid_dict
    zone = {x: 0 for x in ce.keys()}
    for (id1, p1), (id2, p2) in combinations(ce.items(), 2):
        distance = cal_distance(p1[:2], p2[:2]) 
        if distance < min_dist:	
            zone[id1] += 1
            zone[id2] += 1  
    c = [(0,255,0), (255,255,0), (255,0,0)]
    # l = ['Closer to 0', 'Closer to < 4', 'Closer to > 4']

    # zone = {x: ((y, c[0], ce[x][:2]) if y==0  else (c[1]) if y < 4 else (c[2])) for x, y in zone.items()}
    zone = {x: ((y, c[0], ce[x][-2:]) if y==0 else
                (y, c[1], ce[x][-2:]) if y<3 else
                (y, c[2], ce[x][-2:])) 
                for x, y in zone.items()}
    return zone
# ==================================Plotting=============================================
def plot_zone(centroid_dict, min_dist, img):
    "Function for plotting the respective zone to the person distance"
    zone = zone_detector(centroid_dict, min_dist)
    for idx, values in zone.items():
        color = values[1]
        x, y = values[2]
        cv2.circle(img, (x, y), 7, color, -1)
    return img, zone

def bbox(red_zone, centroid_dict, img):
    """Function for plotting bounding box around detected person"""
    for idx, box in centroid_dict.items():  
        if idx in red_zone:   
            cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)
        else:
            cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) 
    return img

def riskLine(zones, img):
    """Function for plotting lines between two points if they are in red/yellow zone"""
    color = [(255,0,0), (255,255,0)]
    i = 0
    for zone in zones:
        for check in range(0, len(zone)-1):
            points = list(zone.values())

            start_point = points[check] 
            end_point = points[check+1]

            check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
            check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y

            if (check_line_x < 75) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
                cv2.line(img, start_point, end_point, color[i], 2)      # Only above the threshold lines are displayed. 
    return img

def plotImg(centroid_dict, min_dist, img):
    red_zone, yellow_zone = violation(centroid_dict, min_dist)

    ## plot bounding box
    # img = bbox(red_zone, centroid_dict, img)

    ## plot zone dots
    img, zone = plot_zone(centroid_dict, min_dist, img)

    # preprocess zone for output
    
    ## summary      
    text = "People at Risk: %s" % str(len(red_zone))
    location = (10,25)
    cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA) 
    
    ## plot lines
    # img = riskLine([red_zone, yellow_zone], img)

    return img, zone

# =======================================================================================