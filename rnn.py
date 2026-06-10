import torch
from torch import nn

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()

    self.hidden_size = hidden_size
    
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.i2o = nn.Linear(input_size + hidden_size, output_size)

  def forward(self, input, hidden):
    x = torch.cat((input, hidden), 1)
    next_hidden = self.i2h(x)
    output = self.i2o(x)
    return output, next_hidden

  def initHidden(self):
    return torch.zeros(1, self.hidden_size)
