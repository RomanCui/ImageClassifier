import torch.nn as nn
import torch
import torch.nn.functional as F


class NetBuiltin(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(NetBuiltin, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes
        self.l1 = nn.Linear(28*28, 64)
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 10)
        self.softmax = nn.Softmax()

    def params_and_grads(self):
        params = list(self.parameters())

        grads = [param.grad for param in params]

        return params, grads

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.l1(x)
        output = self.tanh(output)
        output = self.l2(output)
        output = output - 0.2 * torch.sin(output)
        output = self.l3(output)
        output = self.softmax(output)

        return output

    def get_loss(self, output, target):

        if self.loss_type == 'l2':
            loss = F.mse_loss(output, F.one_hot(
                target, num_classes=10).float()).float()
        else:
            loss = F.cross_entropy(output, target)

        return loss

    def backward_pass(self, loss, x, output, target):
        loss.backward()
