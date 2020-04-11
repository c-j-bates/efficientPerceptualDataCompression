import os
from sklearn.decomposition import PCA
import numpy as np
from VAE_fruits import make_net
from collections import namedtuple
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CHECKPOINT_DIR = os.environ['MEMNET_CHECKPOINTS']
TRAININGSET_DIR = os.environ['MEMNET_TRAININGSETS']


def build(beta):
    vae_args = {
        "activation": "relu",
        "checkpoint_dir": CHECKPOINT_DIR,
        "trainingset_dir": TRAININGSET_DIR,
        "dataset": "fruits360",
        "regenerate_steps": 10000,
        "batch": 128,
        "buffer_size": 10,
        "reconstruction_loss_weights": [0.1, 1.],
        "rate_loss_weight": beta,
        "beta0": None,
        "beta_steps": None,
        "decision_loss_weights": [1., 1.],
        "regularizer_loss_weight": 1e-8,
        "dropout_prob": 1.0,
        "sampling_off": False,
    }
    vae_args = namedtuple("arguments", vae_args.keys())(*vae_args.values())
    net = make_net(vae_args)
    net.build()
    return net


def get_PCA_fits_and_labels(n_batches, net):
    z = []  # Memory vectors
    labels = []  # Which category each belongs to
    for i in range(n_batches):
        print("{}/{}".format(str(i + 1), n_batches))
        x, y = net.generate_batch("test", keep_session=True)
        # z.extend(net.encode(x, keep_session=True))
        z.extend(net.encode_presample(x, keep_session=True)[0])
        labels.extend(y)
    z_fit = pca.fit_transform(z)
    return z_fit, labels


def plot(Z_fit, Labels, f_ext="pdf"):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    color_dict = {0: "blue", 1: "red", 2: "orange"}
    lims = np.array(Z_fit).min(), np.array(Z_fit).max()
    titles = ["High capacity", "Medium capacity", "Low capacity"]
    for i in range(3):
        colors = [color_dict[l] for l in Labels[i]]
        ax[i].scatter(Z_fit[i][:, 0], Z_fit[i][:, 1], color=colors, s=0.5)
        if i == 0:
            ax[i].set_xlim(lims[0], lims[1])
            ax[i].set_ylim(lims[0], lims[1])
        else:
            ax[i].set_xlim(lims[0] / 2, lims[1] / 2)
            ax[i].set_ylim(lims[0] / 2, lims[1] / 2)
        ax[i].xaxis.set_major_locator(
            ticker.MultipleLocator(int((lims[1] - lims[0]) / 3)))
        ax[i].yaxis.set_major_locator(
            ticker.MultipleLocator(int((lims[1] - lims[0]) / 3)))
        if i == 1:
            ax[i].set_xlabel("1st principal component")
        if i == 0:
            ax[i].set_ylabel("2nd principal component")
        ax[i].set_title(titles[i])
        ax[i].set_aspect("equal")
        ax[i].spines["top"].set_visible(False)
        ax[i].spines["right"].set_visible(False)
    plt.tight_layout()
    save_pth = "plots/fruits_memory_pca/fruits_memory_pca.{}".format(f_ext)
    plt.savefig(save_pth)
    print("Saved to " + save_pth)


n_batches = 100
betas = [1e-7, 0.001, 0.01]

Z, L = [], []
for beta in betas:
    net = build(beta)
    pca = PCA(n_components=2)
    z_fit, labels = get_PCA_fits_and_labels(n_batches, net)
    Z.append(z_fit)
    L.append(labels)
    del net
    tf.reset_default_graph()  # Must include to delete TF graph
plot(Z, L)
