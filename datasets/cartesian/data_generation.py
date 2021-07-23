""" Generating training and test files for iterative decoding on cartesian
    product problem.
"""

# Standard library imports
import numpy as np
import numpy.random as random
# Specify local path in which the data will be saved
data_path = "MyDrive/preprocess_cartesian/token/data/"
# Flags
row = False # whether to predict next row or next pair of tokens
copy = False # whether to copy previous output to next output
short_input = False # whether to copy only last predicted pair to input or
                    # entire predicition so far
# Tokens
SEP_TOKEN = "[SEP]"
IN_OUT_TOKEN = "[SEP2]"
END_TOKEN = "[END]"
START_TOKEN = "[START]"
END_ITERATION_TOKEN = "[ENDITER]"

def create_cartesian_dataset(trainsize, testsize, trainmindigits=1, 
                             trainmaxdigits=6, testmindigits_nb = 6, 
                             testmaxdigits_nb=7, testmindigits_lt=6, 
                             testmaxdigits_lt=7, repeat_digits=False, 
                             reverse=False, row=False, copy_output=False, 
                             short_input=False):
  """
  Generates the cartesian dataset.
  
  Generates a training set and an easy and a hard test sets for the cartesian 
  product problem, in original and iterative decoding form. The easy test set 
  contains samples with same length as the training set. The hard test set 
  contains only longer samples.
  
  Args:
    trainsize: a scalar int corresponding to the number of training samples
    testsize: a scalar int corresponding to the number of test samples
    trainmindigits: a scalar int corresponding to the minimum length of the 
      training samples
    trainmaxdigits: a scalar int corresponding to the maximum length of the 
      training samples (not included in the range)
    testmindigits_nb: a scalar int corresponding to the minimum length of the 
      number arguments of the test samples
    testmaxdigits_nb: a scalar int corresponding to the maximum length of the 
      number arguments of the test samples (not included in the range)
    testmindigits_lt: a scalar int corresponding to the minimum length of the 
      letter arguments of the test samples
    testmaxdigits_lt: a scalar int corresponding to the maximum length of the 
      letter arguments of the test samples (not included in the range)
    repeat_digits: a boolean flag indicating whether to repeat digits in each 
      sample
    reverse: a boolean flag indicating whether to reverse inputs and outputs
    row: a boolean flag indicating whether to include the next pair of tokens or
      the next row of pairs of tokens in the iterative decoding outputs
    copy_output: a boolean flag indicating whether to copy the previous 
      iterative decoding output in the next iterative decoding output 
    short_input: a boolean flag indicating whether to only include the previous 
      pair of tokens or the entire previous output in the next iterative decoding
      input 

  Returns:
    A tuple containing six lists corresponding to the trainining examples, the 
    iterative decoding trainining examples, the easy test examples, the iterative
    decoding easy test examples, the hard test examples, and the iterative 
    decoding hard test examples. Each list contains two sublists, one for the
    input samples and one for the output samples.
    
  """

  def create_example(minlen_nb, maxlen_nb, minlen_lt, maxlen_lt):
    symbols1 = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    symbols2 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    l1 = random.randint(minlen_nb, maxlen_nb)
    l2 = random.randint(minlen_lt, maxlen_lt)
    set1 = []
    set2 = []
    for i in range(l1):
      number = random.choice(symbols1)
      if not repeat_digits:
        symbols1.remove(number)
      set1.append(number)
    for i in range(l2):
      letter = random.choice(symbols2)
      if not repeat_digits:
        symbols2.remove(letter)
      set2.append(letter)
    example_in = set1 + [SEP_TOKEN] + set2 + [END_TOKEN]
    example_out = []
    count = 0
    for i in set1:
      for j in set2:
        example_out.append(i)
        example_out.append(j)
        if count < len(set1) * len(set2) - 1: 
          example_out.append(SEP_TOKEN)
        count += 1
    example_out.append(END_TOKEN)
    if reverse:
      return example_out, [START_TOKEN] + example_in
    else:
      return example_in, [START_TOKEN] + example_out


  def iteratively_decode(example_in, example_out, reverse=False, row=False, 
                         copy_output=False, short_input=False):
    """
    Creates one iterative decoding cartesian example.
    
    Generates an iterative decoding input-output pair from a cartesian input-
    output pair.
    
    Args:
      example_in: a list corresponding to a cartesian input
      example_out: a list corresponding to a cartesian output
      reverse: a boolean flag indicating whether to reverse inputs and outputs
      row: a boolean flag indicating whether to include the next pair of tokens or
        the next row of pairs of tokens in the iterative decoding outputs
      copy_output: a boolean flag indicating whether to copy the previous 
        iterative decoding output in the next iterative decoding output 
      short_input: a boolean flag indicating whether to only include the previous 
        pair of tokens or the entire previous output in the next iterative decoding
        input 

    Returns:
      An iterative decoding cartesian example, i.e., a list of iterative decoding
      inputs and a list of iterative decoding outputs with shape specified by the
      flags row, copy_output and short_input.
      
    """
    
    size = len(example_in)
    idx_list = [idx + 1 for idx, val in
             enumerate(example_in) if val == SEP_TOKEN]
    sets = [example_in[i:j-1] for i, j in
         zip([0] + idx_list, idx_list + 
         ([size] if idx_list[-1] != size else []))]
    inputs = [example_in]
    outputs = []
    input = example_in[:-1] + [IN_OUT_TOKEN]
    output = [START_TOKEN] 
    if not row:
      for i in range(len(sets[0])):
        for j in range(len(sets[1])):
          if i > 0 or j > 0:
            input += [SEP_TOKEN]
            output += [SEP_TOKEN]
          if not copy_output:
            output = [START_TOKEN] 
          if short_input:
            input = example_in[:-1] + [IN_OUT_TOKEN]
          input += [sets[0][i]] + [sets[1][j]] 
          output += [sets[0][i]] + [sets[1][j]] 
          if i < len(sets[0]) - 1 or j < len(sets[1]) - 1: 
            inputs.append(input + [END_TOKEN])
          else:
            output += [END_ITERATION_TOKEN]
          outputs.append(output + [END_TOKEN]) 
    else:
      for i in range(len(sets[0])):
        if not copy_output:
          output = [START_TOKEN]
        if short_input:
            input = example_in[:-1] + [IN_OUT_TOKEN]
        for j in range(len(sets[1])):
          if not short_input and (i > 0 or j > 0):
            input += [SEP_TOKEN]
          if short_input and j > 0:
            input += [SEP_TOKEN]
          if copy and (i > 0 or j > 0):
            output += [SEP_TOKEN]
          if not copy and j > 0:
            output += [SEP_TOKEN]
          input += [sets[0][i]] + [sets[1][j]] 
          output += [sets[0][i]] + [sets[1][j]] 
        if i < len(sets[0]) - 1:                  
          inputs.append(input + [END_TOKEN])
        else:
          output += [END_ITERATION_TOKEN]
        outputs.append(output + [END_TOKEN]) 
    if reverse:
      return outputs, inputs
    else:
      return inputs, outputs


  def create_examples(n, minlen_nb, maxlen_nb, minlen_lt, maxlen_lt):
    examples_in, examples_out = [], []
    it_dec_examples_in, it_dec_examples_out = [], []
    for i in range(n):
      ein, eout = create_example(minlen_nb, maxlen_nb, minlen_lt, maxlen_lt)
      it_dec_ein, it_dec_eout = iteratively_decode(ein, eout, reverse, row, 
                                                   copy_output, short_input)
      examples_in.append(ein)
      examples_out.append(eout)
      it_dec_examples_in.append(it_dec_ein)
      it_dec_examples_out.append(it_dec_eout)
    return [examples_in, examples_out], [it_dec_examples_in, it_dec_examples_out]


  train_examples, it_dec_train_examples = create_examples(trainsize, trainmindigits,
                                                          trainmaxdigits, trainmindigits,
                                                          trainmaxdigits)
  test_easy_examples, it_dec_test_easy_examples = create_examples(testsize, 
                                                                  trainmindigits, 
                                                                  trainmaxdigits,
                                                                  trainmindigits,
                                                                  trainmaxdigits)
  test_hard_examples, it_dec_test_hard_examples = create_examples(testsize, 
                                                                  testmindigits_nb, 
                                                                  testmaxdigits_nb,
                                                                  testmindigits_lt, 
                                                                  testmaxdigits_lt)
  return (train_examples, it_dec_train_examples, 
          test_easy_examples, it_dec_test_easy_examples,
          test_hard_examples, it_dec_test_hard_examples)


def main():

  def generate_original_data_files(examples_in, examples_out, filepath_src, 
                                  filepath_tgt):
    out_file_src = open(filepath_src, "w")
    out_file_tgt = open(filepath_tgt, "w")
    for in_line, out_line in zip(examples_in, examples_out):
      out_file_src.write(" ".join(in_line) + "\n")
      out_file_tgt.write(" ".join(out_line) + "\n")
    out_file_src.close()
    out_file_tgt.close()


  def generate_it_dec_data_files(examples_in, examples_out, filepath_src, 
                                  filepath_tgt):
    out_file_src = open(filepath_src, "w")
    out_file_tgt = open(filepath_tgt, "w")
    for in_line, out_line in zip(examples_in, examples_out):
      for l1, l2 in zip(in_line, out_line):
        out_file_src.write(" ".join(l1) + "\n")
        out_file_tgt.write(" ".join(l2) + "\n")
    out_file_src.close()
    out_file_tgt.close()


  examples = create_cartesian_dataset(trainsize = 200000, testsize = 1024, 
                                      trainmindigits = 1, trainmaxdigits = 6, 
                                      testmindigits_nb = 6, testmaxdigits_nb = 7,
                                      testmindigits_lt = 5, testmaxdigits_lt = 6,  
                                      repeat_digits = False, reverse = False, 
                                      row = row, copy_output = copy, 
                                      short_input = short_input)
  # Generating (normal) training data.
  generate_original_data_files(examples[0][0], examples[0][1], data_path + "train.src",
                               data_path + "train.tgt")
  # Generating (normal) test data (easy and hard).
  generate_original_data_files(examples[2][0], examples[2][1], 
                               data_path + "test_easy.src",
                               data_path + "test_easy.tgt")
  generate_original_data_files(examples[4][0], examples[4][1], 
                               data_path + "test_hard.src",
                               data_path + "test_hard.tgt")
  # Generating iterative decoding training data.
  generate_it_dec_data_files(examples[1][0], examples[1][1], 
                               data_path + "it_dec_train.src",
                               data_path + "it_dec_train.tgt")
  # Generating iterative decoding val and test data (easy).
  ratio = 0.1 # ratio of test samples used for validation, i.e., to calculate
              # step-by-step accuracy
  limit_idx = int(np.ceil(ratio * len(examples[2][0])))
  examples_in = examples[3][0][:limit_idx]
  examples_out = examples[3][1][:limit_idx]
  generate_it_dec_data_files(examples_in, examples_out, 
                               data_path + "it_dec_val_easy.src",
                               data_path + "it_dec_val_easy.tgt")
  examples_in = examples[2][0][limit_idx:]
  examples_out = examples[2][1][limit_idx:]
  it_dec_examples_in = examples[3][0][limit_idx:]
  out_file_ops_test = open(data_path + "it_dec_test_easy.ops", "w")
  for count in range(len(examples_out)):
      examples_out[count] = examples_out[count][:-1]
      examples_out[count].append(END_ITERATION_TOKEN)
      examples_out[count].append(END_TOKEN)
      out_file_ops_test.write(str(len(it_dec_examples_in[count])) + "\n")
  out_file_ops_test.close()
  generate_original_data_files(examples_in, examples_out, 
                               data_path + "it_dec_test_easy.src",
                               data_path + "it_dec_test_easy.tgt")
  # Generating iterative decoding val and test data (hard).
  ratio = 0.1 # ratio of test samples used for validation, i.e., to calculate
              # step-by-step accuracy
  limit_idx = int(np.ceil(ratio * len(examples[4][0])))
  examples_in = examples[5][0][:limit_idx]
  examples_out = examples[5][1][:limit_idx]
  generate_it_dec_data_files(examples_in, examples_out, 
                               data_path + "it_dec_val_hard.src",
                               data_path + "it_dec_val_hard.tgt")
  examples_in = examples[4][0][limit_idx:]
  examples_out = examples[4][1][limit_idx:]
  it_dec_examples_in = examples[5][0][limit_idx:]
  out_file_ops_test = open(data_path + "it_dec_test_hard.ops", "w")
  for count in range(len(examples_out)):
      examples_out[count] = examples_out[count][:-1]
      examples_out[count].append(END_ITERATION_TOKEN)
      examples_out[count].append(END_TOKEN)
      out_file_ops_test.write(str(len(it_dec_examples_in[count])) + "\n")
  out_file_ops_test.close()
  generate_original_data_files(examples_in, examples_out, 
                               data_path + "it_dec_test_hard.src",
                               data_path + "it_dec_test_hard.tgt")


if __name__ == '__main__':
  main()
