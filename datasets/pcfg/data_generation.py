""" Generating training and test files for iterative decoding on PCFG.
"""

from absl import app
import numpy as np
import re

# Specify local path to PCFG data in
# https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset
# Choose between iid (pcfgset), productivity and systematicity splits
data_path = "/content/drive/MyDrive/preprocess_pcfg/pcfgset/data/"


def generate_intermediate_tasks(source_line, target_line):
  """Generates intermediate tasks for iterative decoding.

  Generates a list of intermediate tasks (including original instruction 
  and original output) for each source line and target line pair

  Args:
    source_line: A source (instruction) string from the PCFG dataset.
    target_line: A target (output) string from the PCFG dataset.

  Returns:
    A list of strings where each string corresponds to an intermediate decoding 
    task. The last string is always the final output. For example:

    ["swap_first_last copy remove_second E18 E15 Q6 , P15 L18 X10 I15 Y14",
    "swap_first_last copy E18 E15 Q6", "swap_first_last E18 E15 Q6",
    "Q6 E15 E18"]

  Raises:
    AssertionError: Output and target strings do not match.
  """
  
  unary_keys = ['copy', 'reverse', 'shift', 'swap_first_last', 'repeat', 'echo']
  binary_keys =['append', 'prepend', 'remove_first', 'remove_second']
  key_lengths = {k: len(k) for k in unary_keys + binary_keys}

  # Listing all the operations in a sentence (ordered from right to left).
  op_idx = []
  operations = []
  for k in unary_keys + binary_keys:
    this_op_idx = [m.start() for m in re.finditer(k, source_line)]
    for i in this_op_idx:
      op_idx.append(i)
      operations.append(k)
  op_idx = np.array(op_idx)
  sort_idx = np.argsort(-op_idx)
  op_idx = op_idx[sort_idx]
  operation_list = [operations[i] for i in sort_idx.tolist()]

  # Populating list of intermediate tasks.
  intermediate_tasks = [source_line]
  aux_line = source_line
  for i, op in zip(op_idx, operation_list):
    this_int_task = aux_line[0:i]
    right_idx = i + key_lengths[op] + 1
    aux_op = aux_line[right_idx:]
    # Unary keys:
    if op in unary_keys:
      left_idx = aux_op.find(' , ')
      if left_idx != -1:
        aux_op = aux_op[:left_idx]
      # aux_list contains the tokens to be processed with the current operation.
      aux_list = aux_op.split(' ')
      # Copy or repeat:
      if op == 'copy' or op == 'repeat':
        this_int_task = this_int_task + aux_op
        if op == 'repeat':
          this_int_task = this_int_task + ' ' + aux_op
        if left_idx != -1:  
          this_int_task += ' ' + aux_line[right_idx+left_idx+1:]
      # Other unary operations:
      else:
        # To implement the following operations, we first modify aux_list to 
        # contain the tokens that will be outputted by the operation, in the
        # correct order. E.g., for 'reverse', the tokens do not change, but we 
        # need to reverse the order of aux_list.
        if op == 'reverse':
          aux_list = aux_list[::-1]
        elif op == 'shift':
          shifted_token = aux_list.pop(0)
          aux_list.append(shifted_token)
        elif op == 'swap_first_last':
          aux_list[0], aux_list[-1] = aux_list[-1], aux_list[0]
        else: # op == 'echo'
          aux_list.append(aux_list[-1])
        # Once aux_list is updated, we write it into the instruction string for 
        # the next task with spaces in between consecutive tokens ...
        for token in aux_list:
          this_int_task += token + ' '
        # ... and either remove the last space if this is the end of the string,
        # or append the remainder of the string.
        if left_idx == -1:
          this_int_task = this_int_task[0:-1]
        else:
          this_int_task += aux_line[right_idx+left_idx+1:]
    # Binary keys:
    else:
      aux_list = aux_op.split(' , ')
      if op =='append':
        this_int_task = this_int_task + aux_list[0] + ' ' + aux_list[1]        
      elif op =='prepend':
        this_int_task = this_int_task + aux_list[1] + ' ' + aux_list[0]
      elif op =='remove_first':
        this_int_task = this_int_task + aux_list[1]
      else: # op == 'remove_second'
        this_int_task = this_int_task + aux_list[0]
      for j in range(2, len(aux_list)):
          this_int_task += ' , ' + aux_list[j]
    intermediate_tasks.append(this_int_task)
    aux_line = this_int_task

  assert aux_line == target_line, 'Output and target strings do not match'
  last_task = intermediate_tasks.pop()
  intermediate_tasks.append(last_task + ' END')

  return intermediate_tasks


def main(argv: Sequence[str]):
    
    # Generating training data.
    with open(data_path + 'train.src') as file:
      train_input_lines = file.readlines()
    with open(data_path + 'train.tgt') as file:
      train_output_lines = file.readlines()
    
    out_file_src = open(data_path + 'it_dec_train.src', 'w')
    out_file_tgt = open(data_path + 'it_dec_train.tgt', 'w') 
    
    for in_line, out_line in zip(train_input_lines, train_output_lines):
      int_tasks = generate_intermediate_tasks(in_line.strip('\n'), out_line.strip('\n'))
      for i in range(len(int_tasks) - 1):
        out_file_src.write(int_tasks[i] + '\n')
        out_file_tgt.write(int_tasks[i + 1] + '\n')
    
    out_file_src.close()
    out_file_tgt.close()
    
    # Generating test data.
    with open(data_path + 'test.src') as file:
      test_input_lines = file.readlines()
    with open(data_path + 'test.tgt') as file:
      test_output_lines = file.readlines()
    
    out_file_src = open(data_path + 'it_dec_test.src', 'w')
    out_file_tgt = open(data_path + 'it_dec_test.tgt', 'w')
    
    for in_line, out_line in zip(test_input_lines, test_output_lines):
      out_file_src.write(in_line)
      out_file_tgt.write(out_line.strip('\n') + ' END\n')
    
    out_file_src.close()
    out_file_tgt.close()
    
    
if __name__ == '__main__':
  app.run(main)