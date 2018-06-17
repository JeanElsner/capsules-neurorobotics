import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VectorCapsules(nn.Module):
    def __init__(self, routing_iterations, num_classes=5):
        super(VectorCapsules, self).__init__()

    def forward(self, input):
        return input