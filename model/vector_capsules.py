import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def squash(s):
    s_square = s.pow(2).sum(dim=2)
    return s * (s_square / (1 + s_square) / s_square.sqrt()).view(s.size(0), s.size(1), 1)

class RoutingByAgreement(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):
        super(RoutingByAgreement, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()

        c = F.softmax(self.b)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = F.softmax(b_batch.view(-1, output_caps)).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)
        return v

class Capsules(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module):
        super(Capsules, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim))
        self.routing_module = routing_module
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):
        caps_output = caps_output.unsqueeze(2)
        print(caps_output.size(), self.weights.size())
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)
        v = self.routing_module(u_predict)
        return v

class PrimaryCapsules(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsules, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()
        out = out.view(N, self.output_caps, self.output_dim, H, W)

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(out.size(0), -1, out.size(4))
        out = squash(out)
        return out

class VectorCapsules(nn.Module):
    def __init__(self, routing_iterations, num_classes=5):
        super(VectorCapsules, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2)
        self.primaryCaps = PrimaryCapsules(32, 32, 14, kernel_size=1, stride=2)  # outputs 8*8
        self.num_primaryCaps = 32 * 14 * 14
        routing_module = RoutingByAgreement(self.num_primaryCaps, num_classes, routing_iterations)
        self.digitCaps = Capsules(self.num_primaryCaps, 14, num_classes, 1, routing_module)

    def forward(self, input, r):
        x = self.conv1(input)
        x = F.relu(x)
        print(x.size())
        x = self.primaryCaps(x)
        print(x.size())
        x = self.digitCaps(x)
        return x.pow(2).sum(dim=2).sqrt()