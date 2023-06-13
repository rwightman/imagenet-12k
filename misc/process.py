import os
import re
import shutil
from collections import Counter
from functools import partial
from PIL import Image
from PIL.ImageStat import Stat
import numpy as np
import pandas as pd
import multiprocessing
from collections import OrderedDict

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images(folder, types=IMG_EXTENSIONS):
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))

    filenames = list(sorted(filenames))
    return filenames


def img_info(f):
    img = Image.open(f)
    fmt = img.format
    img = img.convert('RGB')
    s = Stat(img)
    mean = s.mean
    std = [x ** 0.5 for x in s.var]

    return img, (img.size, mean, std, fmt)


def process_imgs(filnames, base_dir):
    cls_list = []
    id_list = []
    size_list = []
    mean_list = []
    std_list = []
    format_list = []
    error_list = []
    for f in filnames:
        id_only = os.path.splitext(os.path.basename(f))[0]
        img_cls, img_id = id_only.split('_')
        try:
            img = Image.open(f)
            fmt = img.format
           
            mode_changed = False
            if img.mode not in ("RGB", "L"):
                print('Mode', img.mode)
                mode_changed = True
                img = img.convert('RGB')

            resized = False
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
                print(f'resizing {w}, {h} to {wn}, {hn} ')
                img = img.resize((wn, hn), Image.BICUBIC, reducing_gap=3)
                resized = True

            if min(img.size) < 56:
                dest = os.path.relpath(f, base_dir)
                dest = os.path.join(base_dir, 'discard', dest)
                #os.makedirs(os.path.dirname(dest), exist_ok=True)
                #shutil.move(f, dest)
                #continue
                print('small', img.size)
            elif resized or mode_changed:
                print('re-writing image')
                dest = os.path.relpath(f, base_dir)
                dest = os.path.join(base_dir, 'resized', dest)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(f, dest)  # move original
                extra = {}
                if 'icc_profile' in img.info:
                    extra['icc_profile'] = img.info['icc_profile']
                if 'exif' in img.info:
                    extra['exif'] = img.info['exif']
                img.save(f, 'jpeg', subsampling=1, quality=95, **extra)

            if not mode_changed:
                img = img.convert('RGB')
            s = Stat(img)
            mean = s.mean
            std = [x ** 0.5 for x in s.var]
        except Exception as e:
            print('Exception:', e)
            dest = os.path.relpath(f, base_dir)
            dest = os.path.join(base_dir, 'error', dest)
            #os.makedirs(os.path.dirname(dest), exist_ok=True)
            #shutil.move(f, dest)
            error_list.append((id_only, str(e)))
            continue

        cls_list.append(img_cls)
        id_list.append(img_id)
        size_list.append(img.size)
        mean_list.append(mean)
        std_list.append(std)
        format_list.append(fmt)
    # end for

    return cls_list, id_list, np.array(size_list), np.array(mean_list), np.array(std_list), format_list, error_list


def process_dataset(name, filenames, base_dir, num_processes=20,):
    input_slices = [x.tolist() for x in np.array_split(filenames, num_processes)]
    pool = multiprocessing.Pool(num_processes)
    img_cls = []
    img_ids = []
    sizes = []
    means = []
    stds = []
    formats = []
    errors = []
    for m_cls, m_ids, m_sizes, m_means, m_stds, m_fmts, m_err in pool.map(
            partial(process_imgs, base_dir=base_dir), input_slices):
        print(len(m_ids))
        img_cls.extend(m_cls)
        img_ids.extend(m_ids)
        sizes.append(m_sizes)
        means.append(m_means)
        stds.append(m_stds)
        formats.extend(m_fmts)
        errors.extend(m_err)
    pool.close()

    print(errors)
    sizes = np.concatenate(sizes)
    means = np.concatenate(means)
    stds = np.concatenate(stds)

    num_s = sizes[(sizes < 56).any(axis=1)]
    num_l = sizes[(sizes > 896).any(axis=1)]
    print('small:', len(num_s), 'large:', len(num_l))

    min_w = np.min(sizes[:, 0])
    min_h = np.min(sizes[:, 1])
    print('Minimum width, height:', min_w, min_h)
    print('Mean width, height', np.mean(sizes[:, 0]), np.mean(sizes[:, 1]))
    print('Med width, height', np.median(sizes[:, 0]), np.median(sizes[:, 1]))
    max_w = np.max(sizes[:, 0])
    max_h = np.max(sizes[:, 1])
    print('Max width, height:', max_w, max_h)

    avg_mean = np.mean(means, axis=0)
    avg_std = np.mean(stds, axis=0)
    print('Dataset mean:', avg_mean)
    print('Dataset std:', avg_std)

    fh = Counter(formats)
    print(fh)

    df = pd.DataFrame(data=OrderedDict(
        cls=img_cls, id=img_ids, width=sizes[:, 0], height=sizes[:, 1],
        mean_r=means[:, 0], mean_g=means[:, 1], mean_b=means[:, 2],
        std_r=stds[:, 0], std_g=stds[:, 1], std_b=stds[:, 2],
        fmt=formats,
    ))
    output_csv = '%s-info.csv' % name
    print('Writing %s with %d entries' % (output_csv, len(df.index)))
    df.to_csv(output_csv, index=False)


BASE_DIR = '/data/in21k/'
images_test = find_images(BASE_DIR)
process_dataset('pp', images_test, BASE_DIR)
#process_dataset('validation', images_validation)


aa