import json
import torch
import time
import cv2
import cvzone
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
import time
import datetime
from datetime import timedelta

import super_gradients
from super_gradients.training import models
from super_gradients.common.object_names import Models

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from deep_sort.sort.tracker import Tracker

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def load_model():
    tracker = DeepSort(model_path='deep_sort/deep/checkpoint/ckpt.t7',
                       max_dist=0.2,
                       min_confidence=0.5,
                       nms_max_overlap=1,
                       max_iou_distance=0.3,
                       n_init=3,
                       max_age=30, # changed
                       nn_budget=100,
                       use_cuda=True)
    tracker_2 = DeepSort(model_path='deep_sort/deep/checkpoint/ckpt.t7',
                       max_dist=0.2,
                       min_confidence=0.5,
                       nms_max_overlap=1,
                       max_iou_distance=0.3,
                       n_init=3,
                       max_age=30, # changed
                       nn_budget=100,
                       use_cuda=True)
    yolonas = models.get('yolo_nas_l',pretrained_weights = 'coco').cuda()
    return tracker,tracker_2,yolonas

def detection(path_1,path_2,output_path,mask_path):
    tracker,tracker_2, yolonas = load_model()
    cap = cv2.VideoCapture(path_1)
    cap_2 = cv2.VideoCapture(path_2)
    # mask = cv2.imread(mask_path)
    if not cap.isOpened() and not cap_2.isOpened():
        print('Error opning video stream or file')
        
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_height_2 = int(cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width_2 = int(cap_2.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps_2 = cap_2.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path + 'locker_path__1.mp4',
                          fourcc,
                          fps,
                          (frame_width,frame_height))
    
    out_2= cv2.VideoWriter(output_path + 'entrance_clip__1.mp4',
                          fourcc,
                          fps_2,
                          (frame_width_2,frame_height_2))
    
    # mask = cv2.resize(mask,(frame_width,frame_height))
    entrance_limits = [916*2,596*2,930*2,248*2] #  (916*2, 596*2),(930*2,248*2
    locker_hall_entry_limits = [490,400,900,400]
    locker_hall_exit_limits = [477,174,657,149]
    pathway_2_entry_limits = [75,190,178,163]
    pathway_2_exit_limits = [788,223,1076,447]
    pathway_3_entry_limits = [200,461,372,383]
    ids_stack = []
    ids_stack_2 = []
    total_data = []
    
    i = 0
    counter, fps, elasped = 0,0,0
    counter_2, fps_2, elasped_2 = 0,0,0
    start_time = time.process_time()
    start_time_2 = time.process_time()
    
    
    while True:
        ret,og_frame_locker = cap.read()
        ret_2, og_frame_entrance = cap_2.read()

        if ret and ret_2:
            frame_locker = og_frame_locker.copy()
            frame_entrance = og_frame_entrance.copy()
            # frame_locker = cv2.bitwise_and(og_frame_locker,mask)
            # # frame_locker = cv2.resize(frame_locker, (1280,720))
            # # frame_locker = cv2.cvtColor(frame_locker, cv2.COLOR_RGB2BGR)
            results = list(yolonas.predict(frame_locker,conf = 0.4,iou = 0.7)._images_prediction_lst)
            results_2 = list(yolonas.predict(frame_entrance,conf = 0.4,iou = 0.7)._images_prediction_lst)
            
            
            # Camera 1
            bboxes_xyxy = results[0].prediction.bboxes_xyxy.tolist()
            confidence = results[0].prediction.confidence.tolist()
            
            labels = results[0].prediction.labels.tolist()
            
            person_bboxes_xyxy = [bbox for i, bbox in enumerate(bboxes_xyxy) if labels[i] == 0]
            person_confidence = [conf for i, conf in enumerate(confidence) if labels[i] == 0]
            
            bboxes_xywh = []
            for bbox in person_bboxes_xyxy:
                bbox_xywh = [int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])]
                bboxes_xywh.append(bbox_xywh)
            
            bboxes_xywh = np.array(bboxes_xywh)                
            
            
            resultsTracker = tracker.update(bboxes_xywh, person_confidence, frame_locker)
            
            cv2.line(og_frame_locker, (locker_hall_exit_limits[0]*2, locker_hall_entry_limits[1]*2), (locker_hall_entry_limits[2]*2, locker_hall_entry_limits[3]*2), (0,0,255), 5)
            
            
            
            for track in tracker.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                track_id = track.track_id
                hits = track.hits
                bbox = track.to_tlbr()
                x1,y1,x2,y2 = bbox
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                w = x2 - x1
                h = y2 - y1
                
                shift_percent = 0.50
                y_shift = int(h*shift_percent)
                x_shift = int(w*shift_percent)
                y1 += y_shift
                y2 += y_shift
                x1 += x_shift
                x2 += x_shift
                
                cvzone.cornerRect(og_frame_locker, 
                                  (x1,y1,w,h),
                                  l=40,
                                  t=8,
                                  rt=2,
                                  colorR = (255,0,0),
                                  colorC = (50,50,200))
                
                cvzone.putTextRect(og_frame_locker, f'track_id: {track_id}', (max(0,x1), max(35, y1)), scale=2, thickness=2)
                
                cx, cy = x1 + w // 2 , y1 + h // 2
                
                cv2.circle(og_frame_locker,(cx,cy), 5, (255,0,255), cv2.FILLED)
            
                # if entrance_limits[0] < cx < entrance_limits[2] and entrance_limits[1] - 15 < cy < entrance_limits[3] + 15:
                if locker_hall_entry_limits[0]*2 < cx < locker_hall_entry_limits[2]*2 and locker_hall_entry_limits[1]*2 - 15< cy < locker_hall_entry_limits[3]*2 + 15:
                    cvzone.putTextRect(og_frame_locker, f"track_id_passed: {track_id}", (0,20))
                    if track_id not in ids_stack:
                        ids_stack.insert(0,track_id)
                    # cv2.line(og_frame_locker , (entrance_limits[0], entrance_limits[1]),(entrance_limits[2],entrance_limits[3]), (0,255,0),5)
                    cv2.line(og_frame_locker, (locker_hall_entry_limits[0]*2, locker_hall_entry_limits[1]*2), (locker_hall_entry_limits[2]*2,locker_hall_entry_limits[3]*2), (255,0,0),5)
                
                
                y_c = 50
                for ids in ids_stack:
                    cvzone.putTextRect(og_frame_locker, f"track_id : {ids}", (0,y_c))
                    y_c += 50
            
            current_time = time.process_time()
            elasped = (current_time - start_time)
            counter +=1
            
            if elasped > 1:
                fps = counter/elasped
                counter = 0
                start_time = current_time
            
            cv2.putText(og_frame_locker,
                        f'FPS: {str(round(fps,2))}',
                        (10,int(frame_height//2)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,255,255),
                        2,
                        cv2.LINE_AA)
            
            
            
            
            
            # Camera 2
            bboxes_xyxy_2 = results_2[0].prediction.bboxes_xyxy.tolist()
            confidence_2 = results_2[0].prediction.confidence.tolist()
            
            labels_2 = results_2[0].prediction.labels.tolist()
            
            person_bboxes_xyxy_2 = [bbox for i, bbox in enumerate(bboxes_xyxy_2) if labels_2[i] == 0]
            person_confidence_2 = [conf for i, conf in enumerate(confidence_2) if labels_2[i] == 0]
            
            # person_labels = [label for label in labels if label == 0]

            bboxes_xywh_2 = []
            
            for bbox in person_bboxes_xyxy_2:
                bbox_xywh_2 = [int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])]
                bboxes_xywh_2.append(bbox_xywh_2)
            
            
            bboxes_xywh_2 = np.array(bboxes_xywh_2)
            
            
            # data = {"Shape" : f"{bboxes_xywh.shape}",
            #         "Values" : f"{bboxes_xywh}"}
            # total_data.append(data)
            # with open('bboxes.json', 'w') as j:
            #     json.dump(total_data, j, indent = 2)
                
            # print(bboxes_xywh_2,person_confidence_2,frame_entrance)
            
            resultsTracker_2 = tracker_2.update(bboxes_xywh_2, person_confidence_2, frame_entrance)
            cv2.line(og_frame_entrance , (916*2, 596*2),(930*2,248*2), (0,0,255),5)
            
            # print(resultsTracker)
                 
            for track in tracker_2.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                track_id = track.track_id
                hits = track.hits
                bbox = track.to_tlbr()
                x1,y1,x2,y2 = bbox
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                w = x2 - x1
                h = y2 - y1
                
                shift_percent = 0.50
                y_shift = int(h*shift_percent)
                x_shift = int(w*shift_percent)
                y1 += y_shift
                y2 += y_shift
                x1 += x_shift
                x2 += x_shift
                
                cvzone.cornerRect(og_frame_entrance,
                                  (x1,y1,w,h),
                                #   l=40,
                                #   t=8,
                                #   rt=2,
                                  colorR = (255,0,0),
                                  colorC = (50,50,200))
                
                cvzone.putTextRect(og_frame_entrance, f'track_id: {track_id}', (max(0,x1), max(35, y1)), scale=2, thickness=2)
                
                cx, cy = x1 + w // 2 , y1 + h // 2
                
                cv2.circle(og_frame_entrance,(cx,cy), 5, (255,0,255), cv2.FILLED)
            
                if entrance_limits[0] < cx < entrance_limits[2] and entrance_limits[1] > cy > entrance_limits[3]:
                # if locker_hall_entry_limits[0]*2 < cx < locker_hall_entry_limits[2]*2 and locker_hall_entry_limits[1]*2 - 15< cy < locker_hall_entry_limits[3]*2 + 15:
                    cvzone.putTextRect(og_frame_entrance, f"track_id_passed: {track_id}", (0,20))
                    if track_id not in ids_stack_2:
                        ids_stack_2.insert(0,track_id)
                    cv2.line(og_frame_entrance , (entrance_limits[0], entrance_limits[1]),(entrance_limits[2],entrance_limits[3]), (0,255,0),5)
                    # cv2.line(og_frame_locker, (locker_hall_entry_limits[0]*2, locker_hall_entry_limits[1]*2), (locker_hall_entry_limits[2]*2,locker_hall_entry_limits[3]*2), (255,0,0),5)
                
                
                y_c = 50
                for ids in ids_stack_2:
                    cvzone.putTextRect(og_frame_entrance, f"track_id : {ids}", (0,y_c))
                    y_c += 50
                        
                    
                    
            current_time = time.process_time()
            elasped_2 = (current_time - start_time_2)
            counter_2 +=1
            
            if elasped_2> 1:
                fps_2 = counter_2/elasped
                counter_2 = 0
                start_time_2 = current_time
            
            cv2.putText(og_frame_entrance,
                        f'FPS: {str(round(fps_2,2))}',
                        (10,int(frame_height_2//2)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,255,255),
                        2,
                        cv2.LINE_AA)
            # og_frame_locker = cv2.cvtColor(og_frame_locker, cv2.COLOR_BGR2RGB)
            out.write(og_frame_locker)
            out_2.write(og_frame_entrance)
            # cv2.imshow('imgRegion', cv2.resize(frame_locker, (1280,720)))     
            # cv2.imshow('Camrea 1', cv2.resize(og_frame_entrance, (1280,720)))
            # cv2.imshow('frame_locker', cv2.resize(og_frame_locker, (1280,720)))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break   
        else:
            break
    
    print(ids_stack)
    cap.release()
    cap_2.release()
    out.release()
    out_2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_path_1 = f'/home/ksuser/LS/Yolonas-deepsort/cctv_footage/locker_pathway/locker_hall.mp4'
    input_path_2 = f'/home/ksuser/LS/Yolonas-deepsort/cctv_footage/Entrance_Video/entrance_clip.mp4'
    mask_path = f'/home/ksuser/LS/Yolonas-deepsort/cctv_footage/pathway_3/mask_pathway_3.png'
    
    # input_path = f'/home/ksuser/LS/Yolonas-deepsort/cctv_footage/pathway_2/pathway_2.mp4'
    output_path = f'output/'
    # mask_path = f'/home/ksuser/LS/Yolonas-deepsort/cctv_footage/pathway_2/mask_pathway_2.png'
    # output_path = str(input('Enter Output path - '))    
    start= time.time()
    detection(input_path_1,input_path_2,output_path, mask_path)
    print('Time Taken for tacking : ',str(timedelta(seconds = time.time() - start)))