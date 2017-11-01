import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, optimizers, Chain, datasets
import argparse
from chainer.dataset import convert
from chainer import serializers

from chainer.datasets import get_cifar10


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


def arg():
    parser = argparse.ArgumentParser(description='Convolutional Auto Encoder')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Learning rate for SGD')
    return parser.parse_args()

def main():
    args = arg()
    print('# GPU: {}'.format(args.gpu))
    print('# epoch: {}'.format(args.epoch))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    print('Using CIFAR10 dataset.')
    class_labels = 10
    train, test = get_cifar10()

    if args.test:
        train = train[:200]
        test = test[:200]

    train_count = len(train)
    test_count = len(test)

    model = L.Classifier(CAE(3, 16, 3), lossfun=F.mean_squared_error)
    model.compute_accuracy = False

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    sum_loss = 0

    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        # Reduce learning rate by 0.5 every 25 epochs.
        if train_iter.epoch % 25 == 0 and train_iter.is_new_epoch:
            optimizer.lr *= 0.5
            print('Reducing learning rate to: ', optimizer.lr)

        x_array, _ = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(x_array)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * len(t.data)

        if train_iter.is_new_epoch:
            print('epoch: {}'.format(train_iter.epoch))
            print('train mean loss: {}'.format(
                  sum_loss / train_count))

            # evaluation
            sum_loss = 0
            model.predictor.train = False
            for batch in test_iter:
                x_array, _ = convert.concat_examples(batch, args.gpu)
                x = chainer.Variable(x_array)
                t = chainer.Variable(x_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)

            test_iter.reset()
            model.predictor.train = True
            print('test mean loss: {}'.format(
                  sum_loss / test_count))
            sum_loss = 0
    print('save the model')
    serializers.save_npz('cifar10.weights', model)
    print('save the optimizer')
    serializers.save_npz('cifar10.state', optimizer)

if __name__ == '__main__':
    main()
