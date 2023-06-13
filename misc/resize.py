import os
import glob
import numpy as np
import multiprocessing
from PIL import Image


BASE_DIR = '/data/old_in21k/resized/'
EXCLUDE_DIR = '/data/in21k_resized/'


samplings = {
    (1, 1, 1, 1, 1, 1): 0,
    (2, 1, 1, 1, 1, 1): 1,
    (2, 2, 1, 1, 1, 1): 2,
}


def process(f):
    dest = os.path.relpath(f, BASE_DIR)
    dest = os.path.join(EXCLUDE_DIR, 'xresized', dest)
    try:
        img = Image.open(f)
        fmt = img.format
        if hasattr(img, 'layer') and len(img.layer) == 3:
            sampling = img.layer[0][1:3] + img.layer[1][1:3] + img.layer[2][1:3]
            s = samplings.get(sampling, -1)
        else:
            s = -1
        if max(img.size) > 896:
            w, h = img.size
            if w > h:
                ratio = h / w
                wn = 896
                hn = int(wn * ratio)
            else:
                ratio = w / h
                hn = 896
                wn = int(hn * ratio)
            print(f'resizing {w}, {h} to {wn}, {hn}, sampling: {s} ')
            img = img.resize((wn, hn), Image.BICUBIC, reducing_gap=3)
            extra = {}
            if 'icc_profile' in img.info:
                extra['icc_profile'] = img.info['icc_profile']
            if 'exif' in img.info:
                extra['exif'] = img.info['exif']

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            img.save(dest, 'jpeg', subsampling=1, quality=95, **extra)

    except Exception as e:
        print(e)


def process_imgs(filenames):
    for f in filenames:
        process(f)


num_processes = 20
filenames = list(glob.glob(BASE_DIR + '**/*.JPEG', recursive=True))
input_slices = [x.tolist() for x in np.array_split(filenames, num_processes)]
pool = multiprocessing.Pool(num_processes)
pool.map(process_imgs, input_slices)

