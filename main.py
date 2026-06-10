import os
import torch
from glob import glob
from rnn import RNN
from utils import unicodeToAscii, categoryFromOutput, randomTrainingExample, lineToTensor, n_letters
from torch import nn, optim

category_lines = {}
all_categories = []

def readLines(filename):
  with open(filename, encoding='utf-8') as f:
    # remove extra spacing, and split by new line
    lines = f.read().strip().split('\n')
  return [unicodeToAscii(line) for line in lines]

for filename in glob('data/names/*.txt'):
  category = os.path.splitext(os.path.basename(filename))[0]
  all_categories.append(category)
  lines = readLines(filename)
  category_lines[category] = lines

n_categories = len(all_categories)

rnn = RNN(n_letters, 128, n_categories)

optimizer = optim.SGD(rnn.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

def train(category_tensor, line_tensor):
  hidden = rnn.initHidden()
  optimizer.zero_grad()

  for i in range(len(line_tensor)):
    output, hidden = rnn(line_tensor[i], hidden)
  
  loss = criterion(output, category_tensor)

  loss.backward()
  optimizer.step()

  return output, loss.item()

epochs = 100000
current_loss = 0
logger_epoch = 5000

for e in range(1, epochs+1):
  category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
  output, loss = train(category_tensor, line_tensor)
  output_name, _ = categoryFromOutput(output, all_categories)
  current_loss += loss

  if e % logger_epoch == 0:
    avg_loss = current_loss / logger_epoch
    print(f'Epoch: {e} | Loss: {avg_loss:.4f} | Name: {line} | Predicted: {output_name} / {category}')
    current_loss = 0

def predict(input_line, n_predictions=3):
  print(f'\n Predicting: {input_line}')

  with torch.no_grad():
    line_tensor = lineToTensor(input_line)
    hidden = rnn.initHidden()

    for i in range(len(line_tensor)):
      output, hidden = rnn(line_tensor[i], hidden)
    
    topv, topi = output.topk(n_predictions, 1, True)

    for j in range(n_predictions):
      val = topv[0][j].item()
      category = topi[0][j].item()
      print(f'{val} -> {all_categories[category]}')

predict('Antonelli')
predict('Verstappen')
predict('Sainz')


