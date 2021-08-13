""" Generating training and test files for iterative decoding on CFQ.
"""

import csv
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


START_TOKEN = "[START]"
SEP_TOKEN = "[SEP]"
IN_OUT_TOKEN = "[SEP2]"
END_TOKEN = "[END]"
END_ITER_TOKEN = "[ENDITER]"


def cfq_decompose_output(line):
  tokens = line.split(" ")
  prefix = ""
  postfix = ""
  triplets_text = ""
  state = 0
  for token in tokens:
    if state == 0:
      if token == "{":
        prefix += token + " "
        state = 1
      else:
        prefix += token + " "
    elif state == 1:
      if token == "}":
        postfix += token + " "
        state = 2
      else:
        triplets_text += token + " "
    else:
      postfix += token + " "
  triplets = triplets_text.strip().split(" . ")
  return prefix, triplets, postfix


def cfq_rewrite_cartesian(triplets):
  if not triplets:
    return triplets
  triplet = triplets[0]
  tokens = triplet.split(" ")
  if len(tokens) == 3 and tokens[1] != "a":
    relation = tokens[1]
    left_tokens = [tokens[0]]
    right_tokens = [tokens[2]]
    relation_pairs = [(tokens[0], tokens[2])]
    to_delete = [triplet]
    to_keep = []
    for triplet2 in triplets[1:]:
      tokens2 = triplet2.split(" ")
      if len(tokens2) == 3 and tokens2[1] == relation:
        relation_pairs.append((tokens2[0], tokens2[2]))
        if tokens2[0] not in left_tokens:
          left_tokens.append(tokens2[0])
        if tokens2[2] not in right_tokens:
          right_tokens.append(tokens2[2])
        to_delete.append(triplet2)
      else:
        to_keep.append(triplet2)
    any_missing = False
    for left_token in left_tokens:
      for right_token in right_tokens:
        if (left_token, right_token) not in relation_pairs:
          any_missing = True
          break
      if any_missing:
        break
    if any_missing:
      return ["( " + tokens[0] + " ) ( " + relation + " ) ( " + tokens[2] + " )"] + cfq_rewrite_cartesian(triplets[1:])
    else:
      new_triplet = "( " + " ".join(left_tokens) + " ) ( " + relation + " ) ( " + " ".join(right_tokens) + " )"
      return [new_triplet] + cfq_rewrite_cartesian(to_keep)

  else:
    return [triplet] + cfq_rewrite_cartesian(triplets[1:])


def cfq_merge_cartesians(triplets):
  if not triplets:
    return triplets
  triplet = triplets[0]
  if triplet[0]== "(":
    tokens = triplet.split(" ) ( ")
    if len(tokens) == 3:
      to_keep = []
      relations = [tokens[1]]
      for triplet2 in triplets[1:]:
        if triplet2[0] == "(":
          tokens2 = triplet2.split(" ) ( ")
          if len(tokens2) == 3 and tokens[0] == tokens2[0] and tokens[2] == tokens2[2]:
            relations.append(tokens2[1])
          else:
            to_keep.append(triplet2)
        else:
          to_keep.append(triplet2)
      new_triplet = tokens[0] + " ) ( " + " ".join(relations) + " ) ( " + tokens[2];
      return [new_triplet] + cfq_merge_cartesians(to_keep)
    else:  
      return [triplet] + cfq_merge_cartesians(triplets[1:])
  else:
    return [triplet] + cfq_merge_cartesians(triplets[1:])


def simplify_cfq_output(output):
  """Simplifies CFQ output.

  Simplifies cartesian product in CFQ queries by aggregating them in a single
  clause.

  Args:
      output: string corresponding to the CFQ query (output).  The clauses
        should be separated with spaces (" ").

  Returns:
      A modified SPARQL query with aggregated cartesian products.

  """
  prefix, triplets, postfix = cfq_decompose_output(output)
  triplets = cfq_rewrite_cartesian(triplets)
  triplets = cfq_merge_cartesians(triplets)
  return prefix + " . ".join(triplets) + " " + postfix


def permute_clauses(question, query, csv_map, random=False):
    """Permutes CFQ query clauses.

    Permutes clauses according to the order of the words in the question, and 
    according to the word-SPARQL command dictionary loaded in csv_map. 

    Args:
        question: A string corresponding to the CFQ question (input).
        query: A string corresponding to the CFQ query (output). The clauses
          should be separated with line breaks ("\n").
        csv_map: Optional; Dictionary connecting question words to their
          corresponding SPARQL query commands.
        random: Optional; boolean indicating whether to permute the remaining 
          clauses (which haven't been ordered according to word order and csv_map)
          at random.

    Returns:
        A modified SPARQL query ordered according to the word order in question
          and csv_map.

    """
    words = question.split(" ")
    clauses = query.split("\n")
    prefix = clauses[0]
    postfix = clauses[-1]
    clauses = clauses[1:-1]
    remaining_clauses = len(clauses)
    clause_ordering = -1 * np.ones(len(clauses))
    
    if csv_map is not None:
      count = 0
      for word in words:
        if word in csv_map.keys():
            for idx, clause in enumerate(clauses):
                if csv_map[word] in clause and clause_ordering[idx] == -1:
                   clause_ordering[idx] = count
                   count += 1
                   remaining_clauses -= 1
    
    if remaining_clauses > 0:
      perm_idx = np.arange(len(clauses) - remaining_clauses, len(clauses))
      if random:
        perm_idx = np.random.permutation(perm_idx)
      count = 0
      for idx in range(len(clauses)):
        if clause_ordering[idx] == -1:
            clause_ordering[idx] = perm_idx[count]
            count += 1
            
    clause_ordering = list(clause_ordering.astype(int))
    clauses = [clauses[i] for i in clause_ordering] 
    for idx, clause in enumerate(clauses):
      if idx == len(clauses) - 1:
        if " ." in clause:
            clauses[idx] = clause.replace(" .", "")  
      elif " ." not in clause:
        clauses[idx] += " ."
    out_list = [prefix] + clauses + [postfix]
    return "\n".join(out_list)
      

def generate_seq2seq_examples(ds, filename_in, filename_out, simplify_output=False,
                              permute=False, respect_question_order=False, random=False):
    input_file = open(filename_in, "w")
    output_file = open(filename_out, "w")
    
    csv_map = None
    if respect_question_order:
      csv_map = {}
      with open("cfq_map.csv", mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if row[0] not in csv_map.keys():
                csv_map[str(row[0])] = row[1]
    
    for example in ds: 
      query = example["query"].numpy().decode("utf-8").strip()
      query = query.replace("\n", " ")
      if simplify_output:
        query = simplify_cfq_output(query)
      question = example["question"].numpy().decode("utf-8").strip()
      if permute:
        query = query.replace("{ ", "{\n")
        query = query.replace(". ", ".\n")
        query = query.replace(" }", "\n}")
        query = permute_clauses(question, query, csv_map, random)
        query = query.replace("\n", " ")
      
      input_file.write(question.strip("\n") + "\n")
      output_file.write(query.strip("\n") + "\n")
    
    input_file.close()
    output_file.close()


def generate_it_dec_examples(ds, split, filename_in, filename_out, filename_ops=None, 
                             short_input=False, short_output=True,
                             simplify_output=False, permute=False, 
                             respect_question_order=False, random=False):
    input_file = open(filename_in, "w")
    output_file = open(filename_out, "w")
    if filename_ops is not None:
        ops_file = open(filename_ops, "w")
        
    csv_map = None
    if respect_question_order:
      csv_map ={}
      with open("cfq_map.csv", mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            if row[0] not in csv_map.keys():
                csv_map[str(row[0])] = row[1]
    
    for example in ds: 
      query = example["query"].numpy().decode("utf-8").strip()
      query = query.replace("\n", " ")
      if simplify_output:
        query = simplify_cfq_output(query)
        query = query.replace("{ ", "{\n")
        query = query.replace(". ", ".\n")
        query = query.replace(" }", "\n}")
      question = example["question"].numpy().decode("utf-8").strip()
      if permute:
        query = permute_clauses(question, query, csv_map, random)
      
      list_of_outputs = query.split("\n")
      if filename_ops is not None:
        ops_file.write(str(len(list_of_outputs)) + "\n")
      
      cur_input = question + " " + END_TOKEN
      for i, partial_output in enumerate(list_of_outputs):
        if "test" not in split or i == 0:
            input_file.write(START_TOKEN + " " + cur_input + "\n")
        
        if i == 0:
            cur_output = partial_output
            final_output = cur_output + " " + END_TOKEN
        else:
            cur_output += " " + SEP_TOKEN + " " + partial_output 
            if not short_output:
                final_output = cur_output + " " + END_TOKEN
            else:
                final_output = partial_output + " " + END_TOKEN
            if i == len(list_of_outputs) - 1:
                final_output = final_output.replace(END_TOKEN, END_ITER_TOKEN)
                final_output += " " + END_TOKEN
        
        if "test" not in split:        
            output_file.write(START_TOKEN + " " + final_output + "\n")
        
        cur_input = question + " " + IN_OUT_TOKEN 
        if short_input:
            cur_input += " " + partial_output + " " + END_TOKEN
        else:
            cur_input += " " + cur_output + " " + END_TOKEN
      
      if "test" in split:
        final_output = START_TOKEN + " " + cur_output 
        final_output += " " + END_ITER_TOKEN + " " + END_TOKEN 
        output_file.write(final_output + "\n")
    
    input_file.close()
    output_file.close()
    if filename_ops is not None:
        ops_file.close()


def main():        
    train_ds = tfds.load("cfq/mcd1", split="train")
    train_filename_in = "train.src"
    train_filename_out = "train.tgt"
    generate_seq2seq_examples(train_ds, train_filename_in, train_filename_out, 
                             simplify_output=True, permute=True, 
                             respect_question_order=True)
    it_dec_train_filename_in = "it_dec_train.src"
    it_dec_train_filename_out = "it_dec_train.tgt"
    generate_it_dec_examples(train_ds, "train", it_dec_train_filename_in, 
                             it_dec_train_filename_out, simplify_output=True, 
                             permute=True, respect_question_order=True)
    
    val_ds = tfds.load("cfq/mcd1", split="test[:10%]")
    val_filename_in = "val.src"
    val_filename_out = "val.tgt"
    generate_seq2seq_examples(val_ds, val_filename_in, val_filename_out,
                              simplify_output=True)
    it_dec_val_filename_in = "it_dec_val.src"
    it_dec_val_filename_out = "it_dec_val.tgt"
    generate_it_dec_examples(val_ds, "val", it_dec_val_filename_in, it_dec_val_filename_out,
                             simplify_output=True)
    
    test_ds = tfds.load("cfq/mcd1", split="test")
    test_filename_in = "test.src"
    test_filename_out = "test.tgt"
    generate_seq2seq_examples(test_ds, test_filename_in, test_filename_out, 
                              simplify_output=True)
    it_dec_test_filename_in = "it_dec_test.src"
    it_dec_test_filename_out = "it_dec_test.tgt"
    it_dec_test_filename_ops = "it_dec_test.ops"
    generate_it_dec_examples(test_ds, "test", it_dec_test_filename_in, it_dec_test_filename_out, 
                             it_dec_test_filename_ops, simplify_output=True)
                           
                             
if __name__ == '__main__':
    main()
        
