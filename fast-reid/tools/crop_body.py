#coding=utf-8
import os, sys, time
import os.path as osp
import numpy as np
import glob
import cv2
import json
import logging
import argparse
import multiprocessing as mp
import pdb
pdb.set_trace()

def save_crop_simple(image, frame_id, video_name, box, output_dir, pid2label=None):
    if image is None:
      return
    for item in box:
        x,y,w,h = item[0]
        # if w*1./h > 2 or h*1./w > 6:
        #   continue # remove bizzare body shape
        pid = item[1]
        if pid2label:
          pid = pid2label[pid]
        pid = int(pid)
        pid = ('%05d' % pid)
        img_subfold = os.path.join(output_dir, pid)
        image_path = os.path.join(output_dir, pid, '%s_%s_%s.jpg' % (video_name.split('.')[0], pid, frame_id.zfill(8)))
        if not os.path.exists(img_subfold):
            os.makedirs(img_subfold)
        try:
            cv2.imwrite(image_path, image[y:y+h,x:x+w])
            print('save img to ' + image_path)
        except:
            print('failed to save image')

def get_logger(filename):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    filehandler = logging.FileHandler(filename=filename, mode='w')
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger

def mkdir_if(dirname):
    if not osp.exists(dirname):
        os.makedirs(dirname)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='/ssd/zphe/data/MOT20_train')
    parser.add_argument('--json_dir', type=str, default='/ssd/zphe/data/MOT20_train')
    parser.add_argument('--save_dir', type=str, default='/ssd/zphe/data/reid/MOT20_reid/')
    parser.add_argument('--test_video', type=str, default='')
    args = parser.parse_args()
    return args

def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    frame_dict = {}
    pid_frame_dict = {}
    for frame_id in data.keys():
        frame_dict[int(frame_id)] = data[frame_id]
        for box in data[frame_id]:
            bbox = box[0]
            pid = int(box[1])
            if pid not in pid_frame_dict:
                pid_frame_dict[pid] = {}
            pid_frame_dict[pid][int(frame_id)] = bbox
    return frame_dict, pid_frame_dict

def get_iou(bbx1, bbx2):
    x1 = bbx1[0]
    y1 = bbx1[1]
    x2 = bbx2[0]
    y2 = bbx2[1]
    area1 = bbx1[2] * bbx1[3]
    area2 = bbx2[2] * bbx2[3]
    if area1 > 2 * area2 or area2 > 2 * area1:
        return 0
    xmax1 = bbx1[0] + bbx1[2]
    ymax1 = bbx1[1] + bbx1[3]
    xmax2 = bbx2[0] + bbx2[2]
    ymax2 = bbx2[1] + bbx2[3]

    x = max(x1, x2)
    y = max(y1, y2)
    xmax = min(xmax2, xmax1)
    ymax = min(ymax2, ymax1)
    if xmax <= x or ymax <= y:
        return 0
    iou_area = (xmax - x) * (ymax - y)
    iou = float(iou_area) / (area1 + area2 - iou_area)
    return iou

def get_target_pid_frame_dict(pid_frame_dict, frame_interval=2, disk_save = True, thresh=0.75):
    # get pics every 25 frames
    new_pid_frame_dict = {}
    for pid in pid_frame_dict:
        frame_set = set([])
        frame_list = sorted(list(set(pid_frame_dict[pid].keys())))
        frame_list = frame_list[::frame_interval]
        if disk_save:
            ### only save those images with enough movement, we don't save body crops that stay at the same place
            first_frame = frame_list[0]
            prev_bbx = pid_frame_dict[pid][first_frame]
            filtered_frame_list = [first_frame]
            for fidx in frame_list[1:]:
                this_bbx = pid_frame_dict[pid][fidx]
                iou = get_iou(this_bbx, prev_bbx)
                if iou > thresh:
                    # this box overlaps with the previous box a lot, skip 
                    continue
                else:
                    filtered_frame_list.append(fidx)
                    prev_bbx = this_bbx
            new_pid_frame_dict[pid] = filtered_frame_list
        else:
            new_pid_frame_dict[pid] = frame_list
    frame_dict = {}
    for pid in new_pid_frame_dict:
        for frame in new_pid_frame_dict[pid]:
            bbox = pid_frame_dict[pid][frame]
            if frame not in frame_dict:
                frame_dict[frame] = []
            frame_dict[frame].append([bbox, pid])
    return frame_dict

def get_pids(frame_dict):
    pids = set([])
    for frame_id in frame_dict.keys():
        for (box, pid) in frame_dict[frame_id]:
            pids.add(pid)
    return list(pids)
        
args = get_args()
logger = get_logger('mot17_test.log')
json_files = glob.glob(osp.join(args.json_dir, '*.json'))
videos = glob.glob(osp.join(args.video_dir, '*.mp4'))

global total_pid
total_pid = 0
total_img = 0

def generate_frame_data(inputs):
    video_path, output_fold, total_pid, total_img = inputs
    video_name = osp.basename(video_path) 
    TEST = (video_name == args.test_video)
    logger.info('start processing ' + video_name)
    json_file = osp.join(args.json_dir, video_name + '.final.reduced.json')     
    if not osp.exists(json_file):
        return total_pid, total_img

    frame_dict, pid_frame_dict = load_json(json_file)
    frame_dict = get_target_pid_frame_dict(pid_frame_dict, frame_interval=2, disk_save = True, thresh=0.85)
    pids = get_pids(frame_dict)
    #if not TEST:
    pid2label = {pid : str(idx + total_pid) for idx, pid in enumerate(pids)}
    total_pid += len(pids)

    frame_list = map(int, frame_dict.keys())
    frame_list = sorted(list(frame_list))
    frame_list = frame_list[::2] # set sampling rate to 4
    vidcap = cv2.VideoCapture(video_path)
    success = True
    frame_num = 0
    num_count = 0 
    len_frame_list = len(frame_list)
    while success:
        if frame_list[num_count] < frame_num + 100:
            success = vidcap.grab()
            if frame_num in frame_dict:
                success, image = vidcap.retrieve()
                frame_info = frame_dict[frame_num]
                #if TEST:
                save_crop_simple(image, str(frame_num), video_name, frame_info, osp.join(output_fold, 'train'), pid2label)
                #else:
                #    save_crop_simple(image, str(frame_num), video_name, frame_info, osp.join(output_fold, 'train'), pid2label)
                num_count += 1
                total_img += 1

        else: 
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_list[num_count])
            frame_num = frame_list[num_count]
            success, image = vidcap.read()
            frame_info = frame_dict[frame_num]
            #if TEST:
            save_crop_simple(image, str(frame_num), video_name, frame_info, osp.join(output_fold, 'train'), pid2label)
            #else:
            #    save_crop_simple(image, str(frame_num), video_name, frame_info, osp.join(output_fold, 'train'), pid2label)
            num_count += 1
            total_img += 1
        frame_num += 1
        if num_count == len_frame_list:
            success = False
    logger.info('processing ' + video_name + ' done.')
    return total_pid, total_img

#multi_input = []
#for video in videos:
#   multi_input.append((video, args.save_dir))
#
#pool = mp.Pool(4)
#pool.map(generate_frame_data, multi_input)
#pool.close()
#pool.join()

for video in videos:
    total_pid, total_img = generate_frame_data((video, args.save_dir, total_pid, total_img))
logger.info('All done, total collected {} pids, {} imgs'.format(total_pid, total_img))
