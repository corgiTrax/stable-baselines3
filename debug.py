import torch.nn as nn
import pdb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nets = [None for i in range(2)]
    

net = Net()
pdb.set_trace()