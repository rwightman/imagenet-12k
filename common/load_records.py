import os
import time
import re
import operator
from itertools import chain

import numpy as np
import pandas as pd

MIN_DIM = 56

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


def distribute(sequence):
    """
    Enumerate the sequence evenly over the interval (0, 1).

    >>> list(distribute('abc'))
    [(0.25, 'a'), (0.5, 'b'), (0.75, 'c')]
    """
    m = len(sequence) + 1
    for i, x in enumerate(sequence, 1):
        yield i/m, x


def intersperse(*sequences):
    """
    Evenly intersperse the sequences.

    Based on https://stackoverflow.com/a/19293603/4518341

    >>> list(intersperse(range(10), 'abc'))
    [0, 1, 'a', 2, 3, 4, 'b', 5, 6, 7, 'c', 8, 9]
    >>> list(intersperse('XY', range(10), 'abc'))
    [0, 1, 'a', 2, 'X', 3, 4, 'b', 5, 6, 'Y', 7, 'c', 8, 9]
    >>> ''.join(intersperse('hlwl', 'eood', 'l r!'))
    'hello world!'
    """
    distributions = map(distribute, sequences)
    get0 = operator.itemgetter(0)
    for _, x in sorted(chain(*distributions), key=get0):
        yield x


def load_records(
        train_csv,
        validation_csv,
        label_file,
        alt_label_name='',
        alt_label_file='',
        min_img_size=MIN_DIM,
):
    train_record_df = pd.read_csv(train_csv, index_col=None)
    train_record_df = train_record_df[
        ~((train_record_df.height < min_img_size) | (train_record_df.width < min_img_size))]

    val_record_df = pd.read_csv(validation_csv, index_col=None)
    val_record_df = val_record_df[
        ~((val_record_df.height < min_img_size) | (val_record_df.width < min_img_size))]

    if label_file and os.path.exists(label_file):
        print(f'loading {label_file}')
        with open(label_file, 'r') as sf:
            class_to_idx = {s.strip(): i for i, s in enumerate(sf.readlines())}
    else:
        class_to_idx = sorted(train_record_df['cls'].unique(), key=natural_key)
        class_to_idx = {k: i for i, k in enumerate(class_to_idx)}
        with open(label_file, 'w') as sf:
            sf.writelines(c + '\n' for c in class_to_idx.keys())
    print('class to idx', len(class_to_idx))

    train_record_df['label'] = train_record_df['cls'].map(class_to_idx).astype(int)
    val_record_df['label'] = val_record_df['cls'].map(class_to_idx).astype(int)

    if alt_label_file:
        assert alt_label_name
        with open(alt_label_file, 'r') as sf:
            print(f'loading {alt_label_file}')
            class_to_idx_alt = {s.strip(): i for i, s in enumerate(sf.readlines())}
        print('alt class to idx', len(class_to_idx_alt))

        train_record_df[alt_label_name] = train_record_df['cls'].map(class_to_idx_alt).fillna(-1).astype(int)
        train_record_df = train_record_df[['filename', 'label', alt_label_name, 'cls']]
        val_record_df[alt_label_name] = val_record_df['cls'].map(class_to_idx_alt).fillna(-1).astype(int)
        val_record_df = val_record_df[['filename', 'label', alt_label_name, 'cls']]

        # intersperse alt label samples with others more evenly
        np.random.seed(42)
        record_df_alt = train_record_df[train_record_df.cls.isin(class_to_idx_alt)].index.to_numpy()
        record_df_rest = train_record_df[~train_record_df.cls.isin(class_to_idx_alt)].index.to_numpy()
        np.random.shuffle(record_df_alt)
        np.random.shuffle(record_df_rest)
        mixed = intersperse(record_df_alt, record_df_rest)
        train_record_df = train_record_df.loc[mixed]
    else:
        train_record_df = train_record_df[['filename', 'label', 'cls']]
        val_record_df = train_record_df[['filename', 'label', 'cls']]
        train_record_df = train_record_df.sample(frac=1, random_state=42)

    val_record_df = val_record_df.sample(frac=1, random_state=42)
    print('num train records:', len(train_record_df.index))
    print('num val records:', len(val_record_df.index))
    time.sleep(5)

    train_records = train_record_df.to_records(index=False)
    val_records = val_record_df.to_records(index=False)
    return train_records, val_records
