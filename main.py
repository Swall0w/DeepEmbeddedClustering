# coding:utf-8
import argparse
import os

import chainer
from chainer.datasets import mnist
from chainer import optimizers
from chainer import serializers
from chainer import cuda
from chainer.dataset import convert
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from stacked_denoising_autoencoder import StackedDenoisingAutoEncoder
from deep_embedded_clustering import DeepEmbeddedClustering
from tdistribution_kl_divergence import tdistribution_kl_divergence


def plot_tsne(model, data, labels, seed, iter_num, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with chainer.using_config("train", False):
        z = model(data)
        z.to_cpu()
        z = z.data
        centroids = model.get_centroids()
        centroids.to_cpu()
        centroids = centroids.data
        centroids_num = centroids.shape[0]
        data = np.vstack((z, centroids))
        tsne = TSNE(n_components=2, random_state=1, perplexity=30, n_iter=1000)
        x = tsne.fit_transform(data)
        x = x[:-centroids_num]
        embed_centroids = x[-centroids_num:]

        x1 = [data[0] for data in x]
        y1 = [data[1] for data in x]

        x_max = max(x1)
        x_min = min(x1)
        y_max = max(y1)
        y_min = min(y1)

        embx = [data[0] for data in embed_centroids]
        emvy = [data[1] for data in embed_centroids]

        plt.figure(figsize=(40, 40))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.scatter(x1, y1, s=500, alpha=0.8, c=list(labels), cmap="Paired")
        plt.colorbar()
        plt.scatter(embx, emvy, s=1000, c="black", marker="^")
        filename = "{}/output_seed{}_iter{}.png".format(save_dir, seed, iter_num)
        plt.savefig(filename)
        print("save png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_seed', type=int, default=0)
    parser.add_argument('--cluster', type=int, default=10)
    parser.add_argument('--stop_iter', type=int, default=30)
    args = parser.parse_args()

    gpu_id = args.gpu
    seed = args.seed
    model_seed = args.model_seed
    np.random.seed(seed)
    train, _ = mnist.get_mnist()
    concat_train_data, concat_train_label = convert.concat_examples(train, device=gpu_id)
    perm = np.random.permutation(concat_train_data.shape[0])

    # Load Pretrain Model
    sdae = StackedDenoisingAutoEncoder(concat_train_data.shape[1])
    serializers.load_npz("StackedDenoisingAutoEncoder-seed{}.model".format(args.model_seed), sdae)
    chains = [dae for dae in sdae.children()]
    model = DeepEmbeddedClustering(chains)
    if chainer.cuda.available and args.gpu >= 0:
        model.to_gpu(gpu_id)

    # Initialize centroid
    k = args.cluster
    Z = model(concat_train_data)
    Z.to_cpu()
    Z = Z.data
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    last_labels = kmeans.labels_

    if chainer.cuda.available and args.gpu >= 0:
        centroids = cuda.to_gpu(kmeans.cluster_centers_)
    else:
        centroids = kmeans.cluster_centers_

    model.add_centroids(centroids)
    optimizer = optimizers.MomentumSGD(lr=0.01)
    optimizer.setup(model)

    i = 0
    with chainer.using_config("train", False):
        # Not use Trainer because stop condition is difficult
        print("train DEC")
        while True:
            print("Epoch {}".format(i+1))
            Z = model(concat_train_data)
            centroids = model.get_centroids()
            loss = tdistribution_kl_divergence(Z, centroids)
            print("loss {}".format(loss.data))
            model.cleargrads()
            loss.backward()
            optimizer.update()
            if i % 5 == 0:
                new_labels = model.predict_label(concat_train_data)
                new_labels = cuda.to_cpu(new_labels)
                diff = float(len(np.where(np.equal(new_labels, last_labels) == False)[0])) / Z.shape[0]
                last_labels = new_labels
                plot_tsne(model, concat_train_data[perm[:500]], concat_train_label[perm[:500]], seed, i, "modelseed{}_seed{}".format(model_seed, seed))
                print("diff {}".format(diff))

            if diff <= 0.001:
                break

            if i > args.stop_iter:
                print("Couldn't reach tol")
                break

            i += 1

        outfile = "DeepEmbeddedClustering_modelseed{}_seed{}.model".format(model_seed, seed)
        serializers.save_npz(outfile, model)

if __name__ == '__main__':
    main()
