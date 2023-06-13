"""imagenet22k dataset."""

import tensorflow_datasets as tfds

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """
"""

_LABELS_FNAME = 'imagenet_22k_labels.txt'
_LABELS_ALT_FNAME = 'imagenet_12k_labels.txt'
_TRAIN_CSV = 'train_full.csv'
_VALIDATION_CSV = 'val_12k.csv'


class Imagenet22k(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for imagenet22k dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(imagenet22k): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=['no', 'yes']),
            'label_12k': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    # TODO(imagenet22k): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(),
        'validation': self._generate_examples(),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(imagenet22k): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      yield 'key', {
          'image': f,
          'label': 'yes',
      }
