# our config module
from config import *
from imutils.video import WebcamVideoStream, FileVideoStream, FPS

## thresholds
# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 150


# =============Pretrained Model==============================
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

# ===================Video Input OpenCV=================================
def social_distancing(output, video=0, show_frame=1):
    print("[INFO] sampling frames from webcam...")
    cap = cv2.VideoCapture(video)

    # video meta data
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2

    # output details
    out = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (new_width, new_height))
    
    fps = FPS().start()
    # loops through all frames
    while True:
        ret, frame_read = cap.read()

        # Check if frame present 
        if not ret:
            print('failed to grab frame')
            break
        
        # processing frame
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
        detect = detect_people(frame_resized, net, ln, personIdx=LABELS.index("person"),  min_conf=MIN_CONF, nms_thre=NMS_THRESH)
        image, zone = plotImg(centroid_dict=detect, min_dist= MIN_DISTANCE, img=frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        ret,buffer=cv2.imencode('.jpg', image)
        frame_ = buffer.tobytes()
        yield(b'--frame_\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_ + b'\r\n')
    
    out.release()
    cap.release()
    fps.stop()
    cv2.destroyAllWindows()

    # output message
    print(":::Video Write Completed")
    print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# ===================Video Input Threading==============================
def social_distancing_thread(output, video=0, show_frame=1):
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
        detect = detect_people(frame_resized, net, ln, personIdx=LABELS.index("person"),  min_conf=MIN_CONF, nms_thre=NMS_THRESH)
        image, zone = plotImg(centroid_dict=detect, min_dist= MIN_DISTANCE, img=frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
    
    out.release()
    cap.stop()
    fps.stop()
    cv2.destroyAllWindows()

    # output message
    print(":::Video Write Completed")
    print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

if __name__ == "__main__":
    # social_distancing(video=0, show_frame=1,
    #                   output='test_output_cv.avi')
    social_distancing_thread(video='1.mp4', show_frame=1,
                             output='test_output_thread.avi')