"""cfq_mcd1_it_dec dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds

from datasets.common_files.example_generator import example_generator


_DESCRIPTION = """
Iterative decoding data for the MCD1 split of the CFQ dataset.
"""

_CITATION = """
@article{keysers2019measuring,
  title={Measuring compositional generalization: A comprehensive method on 
  realistic data},
  author={Keysers, Daniel and Sch{\"a}rli, Nathanael and Scales, Nathan and 
  Buisman, Hylke and Furrer, Daniel and Kashubin, Sergii and Momchev, Nikola 
  and Sinopalnikov, Danila and Stafiniak, Lukasz and Tihon, Tibor and others},
  journal={arXiv preprint arXiv:1912.09713},
  year={2019}
}
"""


class CfqMcd1ItDec(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cfq_mcd1_it_dec dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Run datasets/cfq/data_generation.py and save the output files in 
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
        disable_shuffling=False,
        homepage='',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    archive_path = dl_manager.manual_dir / 'data'
    extracted_path = dl_manager.extract(archive_path)
    return {
      'train': self._generate_examples(
          source_path=extracted_path / 'train.src',
          target_path=extracted_path / 'train.tgt',
      ),
      # Val is the split used to check standard generalization to unseen data.
      'val': self._generate_examples(
          source_path=extracted_path / 'val.src',
          target_path=extracted_path / 'val.tgt',
      ),
      # Test is the split used to check iterative decoding generalization.
      'test': self._generate_examples(
          source_path=extracted_path / 'test.src',
          target_path=extracted_path / 'test.tgt',
      ),
      'it_dec_train': self._generate_examples(
          source_path=extracted_path / 'it_dec_train.src',
          target_path=extracted_path / 'it_dec_train.tgt',
      ),
      # Val is the split used to check standard generalization to unseen data.
      'it_dec_val': self._generate_examples(
          source_path=extracted_path / 'it_dec_val.src',
          target_path=extracted_path / 'it_dec_val.tgt',
      ),
      # Test is the split used to check iterative decoding generalization.
      'it_dec_test': self._generate_examples(
          source_path=extracted_path / 'it_dec_test.src',
          target_path=extracted_path / 'it_dec_test.tgt',
          ops_path=extracted_path / 'it_dec_test.ops',
      ),
    }

  def _generate_examples(self, source_path, target_path, ops_path=None):
    return example_generator(source_path, target_path, ops_path)
