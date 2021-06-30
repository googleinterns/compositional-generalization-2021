"""pcfg_productivity_data dataset."""

import tensorflow_datasets as tfds

_DESCRIPTION = """
Iterative decoding data for the productivity split of the PCFG dataset.
"""

_CITATION = """
@article{DBLP:journals/corr/abs-1908-08351,
  author    = {Dieuwke Hupkes and
               Verna Dankers and
               Mathijs Mul and
               Elia Bruni},
  title     = {The compositionality of neural networks: integrating symbolism and
               connectionism},
  journal   = {CoRR},
  volume    = {abs/1908.08351},
  year      = {2019},
  url       = {http://arxiv.org/abs/1908.08351},
  archivePrefix = {arXiv},
  eprint    = {1908.08351},
  timestamp = {Mon, 26 Aug 2019 13:20:40 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1908-08351.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


class PcfgProductivityData(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for pcfg_productivity_data dataset."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Run datasets/pcfg/data_generation.py on the productivity split of the PCFG data 
  (available at https://github.com/i-machine-think/am-i-compositional/tree/
  master/data/pcfgset/productivity) and save output files in `manual_dir/data`.
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
        }),
        supervised_keys=('source', 'target'), 
        homepage='https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset/productivity',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    archive_path = dl_manager.manual_dir / 'data'
    extracted_path = dl_manager.extract(archive_path)
    return {
      'train': self._generate_examples(
          source_path=extracted_path / 'it_dec_train.src',
          target_path=extracted_path / 'it_dec_train.tgt',
      ),
      'test': self._generate_examples(
          source_path=extracted_path / 'it_dec_test.src',
          target_path=extracted_path / 'it_dec_test.tgt',
      ),
    }

  def _generate_examples(self, source_path, target_path):
    """Yields examples."""
    with open(source_path) as file:
      source_lines = file.readlines()
    with open(target_path) as file:
      target_lines = file.readlines()
    count = 0
    for src, tgt in zip(source_lines,target_lines):
      line_id = count
      count+=1
      yield line_id, {
          'source': src.strip('\n'),
          'target': tgt.strip('\n'),
      }
      
