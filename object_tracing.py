import torch
import time
import cv2
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np

import super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

def detection_img(img_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_nas = super_gradients.training.models.get("yolo_nas_s", pretrained_weights="coco")
    model_predictions  = yolo_nas.predict(img_path).show()
    # out.save('out.jpg')

def loading_models():
    deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
    tracker = DeepSort(model_path=deep_sort_weights, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1, max_iou_distance=0.8, max_age=70,)    
    yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").cuda()
    return tracker,yolo_nas

def tracking(video_path,):
    tracker,yolo_nas = loading_models()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video stream or file")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_path = r'output/output.mp4'
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frames = []
    i = 0
    counter, fps, elasped = 0,0,0
    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (1280,720))

            og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = og_frame.copy()

            with torch.no_grad():
                conf_threshold = 0.7
                device = 0

                detection_pred = list(yolo_nas.predict(frame,conf =  conf_threshold, iou = 0.7)._images_prediction_lst)
                # print(detection_pred[0].prediction.labels.tolist())
                # exit()
                bboxes_xyxy = detection_pred[0].prediction.bboxes_xyxy.tolist()
                confidence = detection_pred[0].prediction.confidence.tolist()

                lables = [label for label in detection_pred[0].prediction.labels.tolist() if label == 0.0]
                class_names = ['person']

                labels = [int(label) for label in lables]
                class_name = [class_names[label] for label in labels]

                bboxes_xywh = []
                for bbox in bboxes_xyxy:
                    bbox_xywh = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    bboxes_xywh.append(bbox_xywh)
                
                bboxes_xywh = np.array(bboxes_xywh)

                tracks = tracker.update(bboxes_xywh, confidence,og_frame)
                w = h = 0

                for track in tracker.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    track_id = track.track_id
                    hits = track.hits
                    bbox = track.to_tlbr()
                    # class_name = track.get_class()
                    x1,y1,x2,y2 =bbox
                    w = x2 -x1
                    h = y2 -y1

                    shift_percent = 0.50
                    y_shift = int(h*shift_percent)
                    x_shift = int(w*shift_percent)
                    y1 += y_shift
                    y2 += y_shift
                    x1 += x_shift
                    x2 += x_shift

                    bbox_xywh = (x1,y1,x2,y2)
                    color = (255,0,0)

                    cv2.rectangle(og_frame,(int(x1),int(y1)),(int(x1+w),int(y1+h)),color,2)

                    text_color = (0,0,0)
                    cv2.putText(og_frame,f'{class_name[0]} - {str(track_id)}',(int(x1)+10,int(y1) - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5, text_color,2)
                
                current_time = time.perf_counter()
                elasped = (current_time - start_time)
                counter +=1
                if elasped > 1:
                    fps = counter/elasped
                    counter = 0
                    start_time = current_time
                
                cv2.putText(og_frame,
                            f'FPS: {str(round(fps,2))}',
                            (10,int(h)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,255,255),
                            2,
                            cv2.LINE_AA)
                
                frames.append(og_frame)

                # out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

                # cv2.imshow('Video', cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
    
    cap.release()
    # out.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
    #change this
    video_path = r'/home/ksuser/LS/Yolonas-deepsort/cctv_footage/locker_pathway/locker_hall.mp4'
    tracking(video_path)
    

