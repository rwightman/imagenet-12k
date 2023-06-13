import multiprocessing
import io
import math
import os
import json
from functools import partial
from itertools import accumulate, chain, repeat, tee

import numpy as np
import pandas as pd
import webdataset as wds
from PIL import Image
from PIL.ImageStat import Stat

from common.load_records import load_records


def chunk(xs, n):                                                                                                                                                                                            
    assert n > 0                                                                                                                                                                                             
    L = len(xs)                                                                                                                                                                                              
    s, r = divmod(L, n)                                                                                                                                                                                      
    widths = chain(repeat(s + 1, r), repeat(s, n - r))
    offsets = accumulate(chain((0,), widths))                                                                                                                                                                
    b, e = tee(offsets)                                                                                                                                                                                      
    next(e)                                                                                                                                                                                                  
    return [xs[s] for s in map(slice, b, e)]   


def write_shard(
        shard,
        input_dir,
        output_pattern,
        alt_label='',
        resize_short=False,
        max_img_size=None,
        min_img_size=56,
        include_stats=False,
):
    shard_id, input_records = shard
    ss = open(output_pattern % shard_id, 'wb')
    writer = wds.TarWriter(ss)
    if alt_label:
        alt_label = '_'.join(['label', alt_label])

    success = (shard_id, [])
    error_list = []
    for r in input_records:
        label = r['label']
        label = label.item() if isinstance(label, np.integer) else int(label)
        output_record = dict(label=label)
        if alt_label:
            alt_label_val = r[alt_label]
            alt_label_val = alt_label_val.item() if isinstance(alt_label_val, np.integer) else int(alt_label_val)
            output_record[alt_label] = alt_label_val
        output_record['class_name'] = r['cls']
        f = os.path.join(input_dir, r['filename'])
        try:
            img_stream = io.BytesIO(open(f, 'rb').read())
            img = Image.open(img_stream)
            format_changed = img.format != 'JPEG'
            mode_changed = False
            if img.mode not in ("RGB", "L"):
                print('Mode', img.mode)
                mode_changed = True
                img = img.convert('RGB')

            w, h = img.size
            output_record['width'] = w
            output_record['height'] = h
            resized = False
            if max_img_size is not None:
                cmp_fn = min if resize_short else max
                cmp_size = cmp_fn(img.size)
                if cmp_size > max_img_size:
                    scale = max_img_size / float(cmp_size)
                    if scale != 1.0:
                        wn, hn = tuple(round(d * scale) for d in (w, h))
                        print(f'resizing {w}, {h} to {wn}, {hn} ')
                        img = img.resize((wn, hn), Image.BICUBIC, reducing_gap=3)
                        resized = True
                        output_record['orig_width'] = int(w)
                        output_record['orig_height'] = int(h)
                        output_record['width'] = int(wn)
                        output_record['height'] = int(hn)

            if min_img_size and min(img.size) < min_img_size:
                print('skipping small', img.size)
                raise RuntimeError(f'Skipped. Image ({f}) too small ({img.size})')
            elif resized or mode_changed or format_changed:
                print('re-writing image', resized, mode_changed, format_changed)
                extra = {}
                if 'icc_profile' in img.info:
                    extra['icc_profile'] = img.info['icc_profile']
                if 'exif' in img.info:
                    extra['exif'] = img.info['exif']
                del img_stream
                img_stream = io.BytesIO()
                img.save(img_stream, 'JPEG', subsampling=1, quality=90, **extra)

            img_stream.seek(0)

            if include_stats:
                if not mode_changed:
                    img = img.convert('RGB')
                s = Stat(img)
                output_record['mean'] = s.mean
                output_record['std'] = [x ** 0.5 for x in s.var]

            if 'key' in output_record:
                key = output_record.pop('key')
            else:
                key = os.path.splitext(os.path.basename(f))[0]

            writer.write(dict(__key__=key, jpg=img_stream.read(), json=output_record, cls=label))

        except Exception as e:
            print('Exception:', e)
            error_list.append((r['filename'], str(e)))
            continue

        output_record['filename'] = os.path.basename(f)
        success[1].append(output_record)
    # end for

    writer.close()
    return success, error_list


def process_dataset(
        records,
        dataset_name,
        split_name,
        input_dir,
        output_dir,
        alt_label='',
        num_processes=16,
        num_shards=2048,
        resize_short=True,
        max_img_size=None,
        min_img_size=56
):
    sharded_images_to_write = list(enumerate(chunk(records, num_shards)))
    shard_digits = math.ceil(math.log(num_shards) / math.log(10))
    pattern = f'{dataset_name}-{split_name}-%0{shard_digits}d.tar'
    abs_pattern = os.path.join(output_dir, pattern)

    pool = multiprocessing.Pool(num_processes)
    success = []
    errors = []

    for m_success, m_err in pool.imap_unordered(
            partial(
                write_shard,
                input_dir=input_dir,
                output_pattern=abs_pattern,
                alt_label=alt_label,
                resize_short=resize_short,
                max_img_size=max_img_size,
                min_img_size=min_img_size,
            ), sharded_images_to_write):

        success.append(m_success)
        errors.extend(m_err)

    pool.close()
    print(errors)

    df = pd.DataFrame(success)
    output_csv = f'{dataset_name}-{split_name}-info.csv'
    output_csv = os.path.join(output_dir, output_csv)
    print('Writing %s with %d entries' % (output_csv, len(df.index)))
    df.to_csv(output_csv, index=False)

    success = sorted(success, key=lambda x: x[0])
    filenames = [pattern % s[0] for s in success]
    info = dict(
        name=dataset_name,
        splits=dict(),
    )
    split = dict(
        name=split_name,
        filenames=filenames,
        shard_lengths=[len(s[1]) for s in success],
        alt_label=None,
    )
    split['num_samples'] = sum(split['shard_lengths'])
    info['splits'][split_name] = split
    if alt_label:
        alt_split = '_'.join([split_name, alt_label])
        alt_label = '_'.join(['label', alt_label])
        split = dict(
            name=alt_split,
            filenames=filenames,
            shard_lengths=[sum([ss[alt_label] >= 0 for ss in s[1]]) for s in success],
            alt_label=alt_label,
        )
        split['num_samples'] = sum(split['shard_lengths'])
        info['splits'][alt_split] = split
    return info


INPUT_DIR = '/data/in21k/'
OUTPUT_DIR = '/data/in12k-wds'
TRAIN_CSV = 'train_full.csv'
VAL_CSV = 'val_12k.csv'
SYNSET = 'imagenet-22k.txt'
SYNSET_12k = 'imagenet-12k.txt'
MIN_DIM = 56
INCLUDE_12k = False

train_records, val_records = load_records(
    train_csv=TRAIN_CSV,
    validation_csv=VAL_CSV,
    label_file=SYNSET,
    alt_label_name='label_12k' if INCLUDE_12k else '',
    alt_label_file=SYNSET_12k if INCLUDE_12k else '',
    min_img_size=MIN_DIM,
)

train_info = process_dataset(
    records=train_records,
    dataset_name='imagenet22k',
    split_name='train',
    alt_label='12k',
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    num_shards=4096,
    min_img_size=MIN_DIM,
    max_img_size=600,
)


val_info = process_dataset(
    records=val_records,
    dataset_name='imagenet22k',
    split_name='validation',
    alt_label='12k',
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    num_shards=512,
    min_img_size=MIN_DIM,
    max_img_size=600,
)

for k, v in val_info['splits'].items():
    train_info['splits'][k] = v

with open(os.path.join(OUTPUT_DIR, 'info.json'), 'w') as f:
    json.dump(train_info, f, indent=4)

with open(os.path.join(OUTPUT_DIR, 'info.yaml'), 'w') as f:
    import yaml
    yaml.safe_dump(train_info, f)
