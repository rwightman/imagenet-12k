# ImageNet-12k processing and split info

This is some very hacky code and metadata for the ImageNet-12k splits used for training the 12k models in [`timm`](https://github.com/huggingface/pytorch-image-models)

This ImageNet subset is based on the original `fall11_whole.tar` and is not compatible with the newer Winter 2021 release of ImageNet.

## Code
Included:
* TFDS helpers (in `tfds/`) to build 12k train + val or 22k (w/ 12k alternate annotation) and 12k val
* Webdataset (in `wds/`) to build webdataset tars from scratch or convert from tfds for same shuffle
* Misc scripts used to clean/analyze the original images in `misc/`

*NOTE:* This code is not being maintained and is 'as is', it requires modifications to work in different environments.

## Metadata
* `meta/train_12k.csv` the list of samples in 12k train split
* `meta/train_full.csv` the list of samples in full 22k train split (but w/ held out val samples)
* `meta/val_12k.csv` the list of samples in 12k validation split

Note the validation set is the same for both and only covers the 12k subset. The 12k (11821) synsets were chosen based on being able to have 40 samples per synset for validation w/ at least 400 samples remaining for train. See `meta/frequency.json` for per-synset sample counts. 