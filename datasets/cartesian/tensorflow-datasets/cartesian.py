"""cartesian."""

import sys

import tensorflow as tf
import tensorflow_datasets as tfds

sys.path.append('../../datasets')
from datasets.common_files.example_generator import example_generator


_DESCRIPTION = """
Original and iterative decoding data for the cartesian product dataset.
"""

_CITATION = """
"""


class Cartesian(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cartesian dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Run datasets/cartesian/data_generation.py with the desired values for the flags 
  predict_row, copy_output and short_input, and save the output files in 
  `manual_dir/data`.
  """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          'source': tfds.features.Text(),
          'target': tfds.features.Text(),
          'op': tf.int32,
        }),
        supervised_keys=('source', 'target'), 
        disable_shuffling=True,
        homepage='',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    archive_path = dl_manager.manual_dir / 'data'
    extracted_path = dl_manager.extract(archive_path)
    return {
      # Original data.
      'train': self._generate_examples(
          source_path=extracted_path / 'train.src',
          target_path=extracted_path / 'train.tgt',
      ),
      'test_easy': self._generate_examples(
          source_path=extracted_path / 'test_easy.src',
          target_path=extracted_path / 'test_easy.tgt',
      ),
      'test_hard': self._generate_examples(
          source_path=extracted_path / 'test_hard.src',
          target_path=extracted_path / 'test_hard.tgt',
      ),
      # Iterative decoding data.
      'it_dec_train': self._generate_examples(
          source_path=extracted_path / 'it_dec_train.src',
          target_path=extracted_path / 'it_dec_train.tgt',
      ),
      # Val is the split used to check standard generalization to unseen data.
      'it_dec_val_easy': self._generate_examples(
          source_path=extracted_path / 'it_dec_val_easy.src',
          target_path=extracted_path / 'it_dec_val_easy.tgt',
      ),
      'it_dec_val_hard': self._generate_examples(
          source_path=extracted_path / 'it_dec_val_hard.src',
          target_path=extracted_path / 'it_dec_val_hard.tgt',
      ),
      # Test is the split used to check iterative decoding generalization.
      'it_dec_test_easy': self._generate_examples(
          source_path=extracted_path / 'it_dec_test_easy.src',
          target_path=extracted_path / 'it_dec_test_easy.tgt',
          ops_path=extracted_path / 'it_dec_test_easy.ops',
      ),
      'it_dec_test_hard': self._generate_examples(
          source_path=extracted_path / 'it_dec_test_hard.src',
          target_path=extracted_path / 'it_dec_test_hard.tgt',
          ops_path=extracted_path / 'it_dec_test_hard.ops',
      ),
    }

  def _generate_examples(self, source_path, target_path, ops_path=None):
    return example_generator(source_path, target_path, ops_path)
