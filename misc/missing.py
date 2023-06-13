import os
import glob
import re
#import cv2
import shutil
from collections import Counter
from functools import partial
from PIL import Image
from PIL.ImageStat import Stat
import numpy as np
import pandas as pd
import multiprocessing
from collections import OrderedDict


BASE_DIR = '/data/in21k_resized/'
EXCLUDE_DIR = '/data/in21k_excluded/'
VAL_DIR = os.path.join(BASE_DIR, 'validation')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')

os.makedirs(EXCLUDE_DIR, exist_ok=True)

val_set = pd.read_csv('meta/val_12k.csv')

valid_classes = set(val_set.cls)

#for f in glob.glob(BASE_DIR + '*'):
#    basename = os.path.basename(f)
#    if os.path.basename(f) not in valid_classes:
#        shutil.move(f, EXCLUDE_DIR)

missing = 0
for c, f in zip(val_set.cls, val_set.filename):
    src = os.path.join(BASE_DIR, c, f)
    if os.path.exists(src):
        dst_dir = os.path.join(VAL_DIR, c)
        dst_file = os.path.join(dst_dir, f)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, dst_file)
    else:
        missing += 1

print('missing', missing)

os.makedirs(TRAIN_DIR, exist_ok=True)
for c in valid_classes:
    src = os.path.join(BASE_DIR, c)
    if os.path.exists(src):
        shutil.move(os.path.join(BASE_DIR, c), os.path.join(TRAIN_DIR, c))
