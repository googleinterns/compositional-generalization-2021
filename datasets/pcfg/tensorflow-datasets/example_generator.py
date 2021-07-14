"""example generator for annotated iterative decoding datasets."""

def example_generator(source_path, target_path, ops_path=None):
  """Yields examples."""
  with open(source_path) as file:
    source_lines = file.readlines()
  with open(target_path) as file:
    target_lines = file.readlines()
  count = 0
  if ops_path is None:
    for src, tgt in zip(source_lines, target_lines):
      line_id = count
      count += 1
      yield line_id, {
        'source': src.strip('\n'),
        'target': tgt.strip('\n'),
        'op': 0,
      }
  else:
    with open(ops_path) as file:
      ops_lines = file.readlines()
    for src, tgt, ops in zip(source_lines, target_lines, ops_lines):
      line_id = count
      count += 1
      yield line_id, {
        'source': src.strip('\n'),
        'target': tgt.strip('\n'),
        'op': int(ops.strip('\n')),
      }