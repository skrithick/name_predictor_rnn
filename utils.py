import torch
import unicodedata
import string
import random

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


# Yanked from stackoverflow to convert unicode
def unicodeToAscii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in all_letters
  )

def letterToIndex(letter):
  return all_letters.find(letter)


# One hot vector for the letter
def lineToTensor(line):
  tensor = torch.zeros(len(line), 1, n_letters)
  for li, letter in enumerate(line):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    # 0 because we want a 1xN tensor, not NxN
    # example: line = 'abc' -> tensor[0][0][letterToIndex('a')] = 1, tensor[1][0][letterToIndex('b')] = 1, tensor[2][0][letterToIndex('c')] = 1
    # The final tensor would look like this:
    # [[[0, 1, 0, ..., 0],  # 'a'
    #   [0, 0, 1, ..., 0],  # 'b'
    #   [0, 0, 0, ..., 1]]] # 'c'
    tensor[li][0][letterToIndex(letter)] = 1
  return tensor

def categoryFromOutput(output, all_categories):
  _, top_value = output.topk(1)
  top_idx = top_value[0].item()
  return all_categories[top_idx], top_idx

def randomChoice(l):
  return l[random.randrange(0, len(l))]

def randomTrainingExample(category_lines, all_categories):
  category = randomChoice(all_categories)
  line = randomChoice(category_lines[category])
  category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
  line_tensor = lineToTensor(line)
  return category, line, category_tensor, line_tensor