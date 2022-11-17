class NetManual:
    def __init__(self, loss_type, num_classes, torch_fns, device):
        self.tensor, self.zeros, \
            self.mm, self.mul, self.sum, self.mean, self.log, \
            self.sin, self.cos, self.transpose, \
            self.tanh, self.softmax, self.one_hot = torch_fns

        self.num_classes = num_classes
        self.loss_type = loss_type
        self.device = device

        self.W, self.b = self.init_weights()
        self.dW = [None, None, None]
        self.db = [None, None, None]

    # allow an object of this class to be called like a function to perform forward pass

    def __call__(self, x):
        return self.forward(x)

    def grads(self):
        return self.dW, self.db

    def parameters(self):
        return self.W, self.b

    def params_and_grads(self):
        W1, W2, W3 = self.W
        b1, b2, b3 = self.b

        dW1, dW2, dW3 = self.dW
        db1, db2, db3 = self.db

        params = [W1, b1, W2, b2, W3, b3]
        grads = [dW1, db1, dW2, db2, dW3, db3]

        return params, grads

    def init_weights(self):
        import math

        r = 0.5

        W1 = self.zeros(
            (28 * 28, 64), device=self.device).uniform_(-r, r) / math.sqrt(28 * 28)
        b1 = self.zeros((1, 64), device=self.device)

        W2 = self.zeros(
            (64, 32), device=self.device).uniform_(-r, r) / math.sqrt(64)
        b2 = self.zeros([1, 32], device=self.device)

        W3 = self.zeros(
            (32, 10), device=self.device).uniform_(-r, r) / math.sqrt(32)
        b3 = self.zeros([1, 10], device=self.device)

        W = [W1, W2, W3]
        b = [b1, b2, b3]

        return W, b

    def train(self):
        pass

    def eval(self):
        pass

    def get_loss(self, output, target):

        if self.loss_type == 'l2':
            loss = self.mean(
                (self.one_hot(target, num_classes=10).float() - output)**2)
        else:
            loss = -self.mean(self.one_hot(target,
                              num_classes=10).float() * self.log(output))

        return loss

    def forward(self, x):
        W1, W2, W3 = self.W
        b1, b2, b3 = self.b

        x = x.view(x.size(0), -1)
        Z1 = self.mm(x, W1) + b1
        Z2 = self.tanh(Z1)
        Z3 = self.mm(Z2, W2) + b2
        Z4 = Z3 - 0.2 * self.sin(Z3)
        Z5 = self.mm(Z4, W3) + b3
        output = self.softmax(Z5, dim=1)

        self.Z1 = Z1
        self.Z2 = Z2
        self.Z3 = Z3
        self.Z4 = Z4
        self.Z5 = Z5

        return output

    def backward_pass(self, loss, x, output, target):

        dW1 = dW2 = dW3 = None
        db1 = db2 = db3 = None

        W1, W2, W3 = self.W
        b1, b2, b3 = self.b

        x = x.view(x.size(0), -1)

        Z1 = self.Z1
        Z2 = self.Z2
        Z3 = self.Z3
        Z4 = self.Z4
        Z5 = self.Z5

        # deal with loss here

        if self.loss_type == 'l2':

            dZ5 = self.zeros(output.size(0), 10).to(self.device)
            dYp = (output - self.one_hot(target, num_classes=10).float())
            i = 0
            for item in output:
                item = item.view(1, 10)
                dZ5[i] = dYp[i].view(1, 10) * item - \
                    self.mm(dYp[i].view(1, 10), self.mm(
                        self.transpose(item, 1, 0), item))
                i = i + 1

        else:
            dZ5 = output - self.one_hot(target, num_classes=10).float()

        dZ4 = self.mm(dZ5, self.transpose(W3, 0, 1))
        dZ3 = (1 - 0.2 * self.cos(Z3)) * dZ4
        dZ2 = self.mm(dZ3, self.transpose(W2, 0, 1))
        dZ1 = (1 - self.tanh(Z1)**2) * dZ2

        dW3 = self.mm(self.transpose(self.Z4, 0, 1), dZ5) / x.size(0)
        dW2 = self.mm(self.transpose(self.Z2, 0, 1), dZ3) / x.size(0)
        dW1 = self.mm(self.transpose(x, 0, 1), dZ1) / x.size(0)

        db3 = self.sum(dZ5, 0, True) / x.size(0)
        db2 = self.sum(dZ3, 0, True) / x.size(0)
        db1 = self.sum(dZ1, 0, True) / x.size(0)

        self.dW[:] = dW1, dW2, dW3
        self.db[:] = db1, db2, db3
