from detect_mask_video import *
from detect_social_distance import *
import pandas as pd

# ======================Requirement Mask===================================
# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model(r"face_detector\mask_detectorFINAL.model")

# =================Requirement Social Distancing===========================
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(["yolo-coco/coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco/yolov4.weights"])
configPath = os.path.sep.join(["yolo-coco/yolov4.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# ===================Variable from Social Distancing=======================
## thresholds
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 150

# =====================Combined model prediction===========================
def mask_social(output, video=0, show_frame=1):
    # video input
    if type(video) is int:
        print("[INFO] sampling frames from webcam using thread...")
        cap = WebcamVideoStream(src=video).start()
    else:
        print("[INFO] sampling frames from video file using thread...")
        cap = FileVideoStream(video, queue_size=256).start()

    # video meta data
    frame_width = int(cap.stream.get(3))
    frame_height = int(cap.stream.get(4))
 
   # output details
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (frame_width, frame_height))
    zone_data = pd.DataFrame()
    mask_data = pd.DataFrame()

    fps = FPS().start()
   # loops through all frames
    while True:
        frame_read = cap.read()

        # Check if frame present 
        if type(video) == int:
            if cap.grabbed==False:
                print('failed to grab frame')
                break
        else: 
            if cap.more() == False:
                print('failed to grab frame')
                break
    
        # processing frame
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
       
        ## social distancing detection
        detect = detect_people(frame_resized, net, ln, personIdx=LABELS.index("person"),  min_conf=MIN_CONF, nms_thre=NMS_THRESH)
        image, zone = plotImg(centroid_dict=detect, min_dist= MIN_DISTANCE, img=frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ## detect mask
        (locs, preds) = detect_and_predict_mask(image, faceNet, maskNet)
        mask = mask_plot(locs, preds, image)

        # update dataframe
        zone_color = {'red':0, 'yellow':0, 'green':0}
        for i, y in zone.items():
            color = y[1]
            
            # convert rgb code to color
            b = {(0,255,0):'green', (255,255,0):'yellow', (255,0,0):'red'}

            c = {(0,255,0), (255,255,0), (255,0,0)}
            zone_color[b[color]] += 1

        zone_data = zone_data.append(zone_color, ignore_index=True)
        mask_data = mask_data.append(mask, ignore_index=True)
        
        # display frame
        if show_frame > 0:
            cv2.imshow("Output Frames", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # update the FPS counter
        fps.update()

        # writing changes
        out.write(image)
        if __name__ != "__main__":
            ret,buffer=cv2.imencode('.jpg',image)
            frame_=buffer.tobytes()
            print("hello")
            yield(b'--frame_\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_ + b'\r\n')
    
    out.release()
    cap.stop()
    fps.stop()
    cv2.destroyAllWindows()

    # output message
    print(":::Video Write Completed")
    print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
    
    # # # export analytics data                                                    
    zone_data.to_csv('zone_data.csv', index=False)
    mask_data.to_csv('mask_data.csv', index=False)

if __name__ == "__main__":
    # Analyze a video 
    mask_social(video='1.mp4', show_frame=1, output='test_output_t.avi')
    
