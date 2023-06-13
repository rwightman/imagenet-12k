"""imagenet12k dataset."""

import tensorflow_datasets as tfds
from . import imagenet22k


class Imagenet12kTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for imagenet12k dataset."""
  # TODO(imagenet12k):
  DATASET_CLASS = imagenet12k.Imagenet12k
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
