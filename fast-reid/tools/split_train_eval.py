import os
import os.path as osp
import sys
import random
from collections import defaultdict

label_file = '/ssd/zphe/data/NAIC/1/A/train/label.txt'
val_ratio = 0.2
pid_imgs = defaultdict(list)

with open(label_file, 'r') as rf:
    for line in rf:
        items = line.strip().split(':')
        imgname = items[0]
        pid = items[1]
        pid_imgs[pid].append(imgname)

total_pids = list(pid_imgs.keys())
num = len(total_pids)
val_num = int(num * val_ratio)

val_pids = random.sample(total_pids, val_num)
train_pids = list(set(total_pids) - set(val_pids))

wf = open('NAIC_train_val.txt', 'w')
for pid in train_pids:
    for img in pid_imgs[pid]:
        line = img + ' ' + pid  + ' 0\n'
        wf.write(line)

for pid in val_pids:
    imgs = pid_imgs[pid]
    img_num = len(imgs)
    qry_num = int(img_num * 0.5)
    random.shuffle(imgs)
    for idx, img in enumerate(imgs):
        if idx < qry_num:
            line = img + ' ' + pid + ' 1\n'
        else:
            line = img + ' ' + pid + ' 2\n'
        wf.write(line)