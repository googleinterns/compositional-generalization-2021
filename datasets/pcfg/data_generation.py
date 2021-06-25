""" Generating training and test files for iterative decoding on PCFG.
"""

import re
import numpy as np

# Specify local path to PCFG data in
# https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset
# Choose between iid (pcfgset), productivity and systematicity splits
data_path = "/content/drive/MyDrive/preprocess_pcfg/pcfgset/data/"

# Function that generates a list of intermediate tasks (including original source 
# and original target) for each source line and target line pair
def generate_intermediate_tasks(source_line, target_line):

  unary_keys = ['copy', 'reverse', 'shift', 'swap_first_last', 'repeat', 'echo']
  binary_keys =['append', 'prepend', 'remove_first', 'remove_second']
  key_lengths = {k: len(k) for k in unary_keys + binary_keys}

  # Listing all the operations in a sentence (ordered from right to left)
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

  # Populating list of intermediate tasks
  intermediate_tasks = [source_line]
  aux_line = source_line
  for i, op in zip(op_idx, operation_list):
    this_int_task = aux_line[0:i]
    right_idx = i + key_lengths[op] + 1
    aux_op = aux_line[right_idx:]
    # Unary keys
    if op in unary_keys:
      left_idx = aux_op.find(' , ')
      if left_idx != -1:
        aux_op = aux_op[:left_idx]
      aux_list = aux_op.split(' ')
      # Copy or repeat
      if op == 'copy' or op == 'repeat':
        this_int_task = this_int_task + aux_op
        if op == 'repeat':
          this_int_task = this_int_task + ' ' + aux_op
        if left_idx != -1:  
          this_int_task += ' ' + aux_line[right_idx+left_idx+1:]
      # Other unary operations
      else:
        if op == 'reverse':
          aux_list = aux_list[::-1]
        elif op == 'shift':
          shifted_elt = aux_list.pop(0)
          aux_list.append(shifted_elt)
        elif op == 'swap_first_last':
          aux_list[0], aux_list[-1] = aux_list[-1], aux_list[0]
        else: # op == 'echo'
          aux_list.append(aux_list[-1])
        for elt in aux_list:
          this_int_task += elt + ' '
        if left_idx == -1:
          this_int_task = this_int_task[0:-1]
        else:
          this_int_task += aux_line[right_idx+left_idx+1:]
    # Binary keys
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

  assert aux_line == target_line # to make sure last string is equal to target
  last_task = intermediate_tasks.pop()
  intermediate_tasks.append(last_task + ' END')

  return intermediate_tasks

# Generating training data
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

# Generating test data
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
