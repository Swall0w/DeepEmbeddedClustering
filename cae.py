import chainer
import chainer.links as L
import chianer.functions as F
from chainer import Variable, optimizers, Chain, datasets

class CAE(Chain):
    def __init__(self, input_filter, mid_filter, size_filter):
        super(CAE, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(input_filter, mid_filter, size_filter)
            self.dconv1 = L.Deconvolution2D(mid_filter, input_filter, size_filter)
            self.bn = L.BatchNormalization(mid_filter)

    def __call__(self, x):
        h = F.relu(self.bn(self.conv1(x)))
        h = F.relu(self.dconv1(h))
        return h
