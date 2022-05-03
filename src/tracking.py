import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import ast
import glob
import shutil
import sys
import numpy as np
import imagesize
import cv2
from tqdm import tqdm
from typing import List
import torch
from torchvision.ops import box_iou
from typing import List
import torch
from torchvision.ops import box_iou
import warnings
warnings.filterwarnings('ignore')

import cv2
import torch
from PIL import Image
from scipy.stats import multivariate_normal

import torch
import torchvision.ops.boxes as bops

# load best model
model = torch.hub.load('/media/peter/2TB/julien/mlproject/great-barrier-reef/yolov5', 'custom', 
                       path='/media/peter/2TB/julien/mlproject/great-barrier-reef/yolov5/runs/train/exp9/weights/best.pt', source='local')  # local repo

video_num = 2
img_num = 0
bool = True
path_image = f'/media/peter/2TB/julien/mlproject/great-barrier-reef/full_videos/train_images/video_{video_num}/{img_num}.jpg'
img_loaded = cv2.imread(path_image)
img_loaded = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)

res_model = model(img_loaded).pred[0].cpu().numpy().tolist()

cmp = 0
last_bounding_boxes = []
times_last = []
all_bounding_boxes = []
for l in res_model:
    l.append(cmp)
    l.append(img_num)
    last_bounding_boxes.append(np.asarray(l))
    times_last.append(img_num)
    all_bounding_boxes.append(l)
    cmp+=1

for idx in tqdm(np.arange(1, 10000)):
    img_num_bis = img_num + idx
    path_image = f'/media/peter/2TB/julien/mlproject/great-barrier-reef/full_videos/train_images/video_{video_num}/{img_num_bis}.jpg'
    if os.path.exists(path_image):
        img_loaded = cv2.imread(path_image)
        img_loaded = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
        res_model = model(img_loaded).pred[0].cpu().numpy().tolist()
        idx_time = np.where(np.asarray(times_last) > img_num_bis-25)[0] #look at 1 second
        for l_bis in res_model:
            assigned = False
            for t in idx_time[::-1]:

                previous_box = last_bounding_boxes[t][:4]
                iou = bops.box_iou(torch.tensor([l_bis[:4]]), torch.tensor([previous_box]))
                if iou > 0.3: #half overlaps 
                    assigned = True
                    break;
            if assigned:
                assignment = last_bounding_boxes[t][6]
                l_bis.append(assignment)
                l_bis.append(img_num_bis)
                all_bounding_boxes.append(l_bis)
                last_bounding_boxes[t] = l_bis
                times_last[t] = img_num_bis
            else:
                l_bis.append(cmp)
                cmp += 1
                l_bis.append(img_num_bis)
                all_bounding_boxes.append(l_bis)
                last_bounding_boxes.append(np.asarray(l_bis))
                times_last.append(img_num_bis)

results = np.array(all_bounding_boxes)
np.save(f'results_video_{video_num}_03iou.npy', results)


cluster_ids = np.unique(results[:, 6]).astype('int')
num_times = np.zeros(cluster_ids.max()+1)
mean_prob = np.zeros(cluster_ids.max()+1)
for cluster in cluster_ids:
    num_times[cluster] = (results[:, 6] == cluster).sum()
    mean_prob[cluster] = results[results[:, 6] == cluster, 4].mean()


good_clusters = np.where((num_times > 2))[0]
results_good_clusters = results[np.isin(results[:, 6], good_clusters)]
results_bad_clusters = results[~np.isin(results[:, 6], good_clusters)]


# Conversion des annotations entre coco et yolo
def coco2yolo(image_height, image_width, bboxes):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # normalizinig
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]/ image_height
    
    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]/2
    
    return bboxes

def yolo2coco(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # denormalizing
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]* image_height
    
    # converstion (xmid, ymid) => (xmin, ymin) 
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    
    return bboxes

def load_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) #, cv2.COLOR_BGR2RGB


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_bboxes(img, bbox_np, classes, class_ids, labels_colors, colors = None, show_classes = None, bbox_format = 'yolo', class_name = False, line_thickness = 2):  
     
    image = img.copy()
    show_classes = classes if show_classes is None else show_classes
    colors = (0, 255 ,0) if colors is None else colors
    
    if bbox_format == 'yolo':
        
        for idx in range(len(bbox_np)):  
            
            bbox  = bbox_np[idx]
            cls   = classes[idx]
            cls_id = class_ids[idx]
            color = colors[labels_colors[idx]] if type(colors) is list else colors
            
            if cls in show_classes:
            
                x1 = round(float(bbox[0])*image.shape[1])
                y1 = round(float(bbox[1])*image.shape[0])
                w  = round(float(bbox[2])*image.shape[1]/2) #w/2 
                h  = round(float(bbox[3])*image.shape[0]/2)

                voc_bbox = (x1, y1, x1+2*w, y1+2*h)
                plot_one_box(voc_bbox, 
                             image,
                             color = color,
                             label = cls if class_name else str(get_label(cls)),
                             line_thickness = line_thickness)
    else:
        raise ValueError('wrong bbox format')

    return image

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

def get_imgsize(row):
    row['width'], row['height'] = imagesize.get(row['image_path'])
    return row


for im_num in tqdm(range(10000)):
    image_path = f'/media/peter/2TB/julien/mlproject/great-barrier-reef/full_videos/train_images/video_{video_num}/{im_num}.jpg'
    labels_path = f'/media/peter/2TB/julien/mlproject/great-barrier-reef/datasets/labels/video_{video_num}_{im_num}.txt'
    video_path = f'tracked_video_{video_num}/{im_num}.jpg'
#     if not os.path.exists(video_path):
    if os.path.exists(image_path):

        img_loaded = load_image(image_path)

        bbox_np = []

        labeled= False
        if os.path.exists(labels_path):
            labeled= True
            labels_loaded = np.genfromtxt(labels_path)

#         list_bbox = model(img_loaded).pred[0].cpu().numpy().tolist()
        list_bbox = []
        bbox_good = results_good_clusters[results_good_clusters[:, 7] == im_num]
        bbox_bad = results_bad_clusters[results_bad_clusters[:, 7] == im_num]

        for i in range(len(bbox_good)):
            sub_list = bbox_good[i]
            bbox_np.append([sub_list[0]/1280, sub_list[1]/720, (sub_list[2]-sub_list[0])/1280, (sub_list[3]-sub_list[1])/720])
        for i in range(len(bbox_bad)):
            sub_list = bbox_bad[i]
            bbox_np.append([sub_list[0]/1280, sub_list[1]/720, (sub_list[2]-sub_list[0])/1280, (sub_list[3]-sub_list[1])/720])
        if labeled:
            if labels_loaded.ndim == 1:
                sub_list = labels_loaded[1:].tolist()
                bbox_np.append([(sub_list[0]-sub_list[2]/2), sub_list[1]-sub_list[3]/2, sub_list[2], sub_list[3]])
            else:
                for i in range(labels_loaded.shape[0]):
                    sub_list = labels_loaded[i, 1:].tolist()
                    bbox_np.append([(sub_list[0]-sub_list[2]/2), sub_list[1]-sub_list[3]/2, sub_list[2], sub_list[3]])

        labels_true_model = []
        for i in range(len(bbox_good)):
            labels_true_model.append(0)
        for i in range(len(bbox_bad)):
            labels_true_model.append(2)
        if labeled:
            for i in np.arange(len(list_bbox), len(bbox_np)):
                labels_true_model.append(1)

        names = ['starfish']*len(bbox_np)
        labels = [0]*len(bbox_np)

        plt.figure(figsize = (20, 12))
        plt.imshow(draw_bboxes(img = img_loaded,
                               bbox_np = bbox_np, 
                               classes = names,
                               class_ids = labels,
                               labels_colors = labels_true_model,
                               class_name = True, 
                               colors = [(255,0,255), (255,255,0), (0,255,255)], 
                               bbox_format = 'yolo',
                               line_thickness = 2))
        # plt.axis('OFF')
        plt.savefig(video_path)
#         plt.show()
        plt.close()


# Make Video 


import cv2
import numpy as np
import glob
from tqdm.notebook import tqdm
import os

NUM_IMAGES = 10000
img_array = []
im_num = 20
video_path = f'tracked_video_{video_num}/{im_num}.jpg'
img = cv2.imread(video_path)
height, width, layers = img.shape
size = (width,height)
out = cv2.VideoWriter(f'tracked_video_{video_num}_iou03.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for im_num in tqdm(range(NUM_IMAGES)):
    video_path = f'tracked_video_{video_num}/{im_num}.jpg'
    if os.path.exists(video_path):
        img = cv2.imread(video_path)
        height, width, layers = img.shape
        out.write(img)

out.release()
