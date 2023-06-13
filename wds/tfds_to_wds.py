
import io
import json
import math
import multiprocessing
import os
import re
import numpy as np
import tensorflow as tf
import webdataset as wds
import pandas as pd
from PIL import Image
from glob import glob
from functools import partial


def write_shard(
        shard,
        output_pattern,
        map_synsets=None,
):
    shard_id, shard_file = shard

    raw_dataset = tf.data.TFRecordDataset(shard_file)

    image_feature_desc = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_desc)

    parsed_image_dataset = raw_dataset.map(_parse_image_function)

    ss = open(output_pattern % shard_id, 'wb')
    writer = wds.TarWriter(ss)

    success = (shard_id, [])
    error_list = []
    for r in parsed_image_dataset:
        label = r['label'].numpy()
        label = label.item() if isinstance(label, np.integer) else int(label)
        filename = r['file_name'].numpy().decode('utf-8')
        filename = os.path.basename(filename)
        key = os.path.splitext(filename)[0]
        print(filename)
        if map_synsets is not None:
            synset = map_synsets.get(filename)
        else:
            synset = filename.split('_')[0]
        #filename = os.path.join(synset, filename)  # of form synset/filename

        print(filename, key)

        image_bytes = r['image'].numpy()
        output_record = dict(label=label)
        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            output_record['width'] = w
            output_record['height'] = h
            output_record['filename'] = filename
            #print(output_record)

            writer.write(dict(__key__=key, jpg=image_bytes, json=output_record, cls=label))

        except Exception as e:
            print('Exception:', e)
            error_list.append((filename, str(e)))
            continue

        success[1].append(output_record)
    # end for

    writer.close()
    return success, error_list


def process_dataset(
        dataset_name,
        split_name,
        input_dir,
        output_dir,
        num_processes=16,
        map_synsets=None,
):
    input_pattern = f'**-{split_name}.tfrecord**'
    path_pattern = os.path.join(input_dir, input_pattern)
    files = glob(path_pattern)

    shards = []
    num_shards = 0
    for f in sorted(files):
        mm = re.search(r'\btfrecord-(\d+)-of-(\d+)', f)
        shard_id = int(mm.group(1))
        num_shards = int(mm.group(2))
        shards.append((shard_id, f))
    assert len(shards) == num_shards

    shard_digits = math.ceil(math.log(len(shards)) / math.log(10))
    pattern = f'{dataset_name}-{split_name}-%0{shard_digits}d.tar'
    abs_pattern = os.path.join(output_dir, pattern)

    pool = multiprocessing.Pool(num_processes)
    success = []
    errors = []

    for m_success, m_err in pool.imap_unordered(
            partial(
                write_shard,
                output_pattern=abs_pattern,
                map_synsets=map_synsets,
            ), shards):

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
    )
    split['num_samples'] = sum(split['shard_lengths'])
    info['splits'][split_name] = split
    return info


INPUT_DIR = '/data/in12k-tfds/imagenet12k/1.0.0'
OUTPUT_DIR = '/data/temp'

#synset_map = pd.read_csv('./test.csv')
#synset_map = dict(zip(synset_map.iloc[:, 0], synset_map.iloc[:, 1]))
#print(synset_map)

validation_info = process_dataset(
    dataset_name='imagenet1k',
    split_name='validation',
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
 #   map_synsets=synset_map,
)

train_info = process_dataset(
    dataset_name='imagenet1k',
    split_name='train',
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
)

print(train_info)



for k, v in validation_info['splits'].items():
    train_info['splits'][k] = v

with open(os.path.join(OUTPUT_DIR, 'info.json'), 'w') as f:
    json.dump(train_info, f, indent=4)
