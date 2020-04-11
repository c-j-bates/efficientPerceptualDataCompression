import os
from VAE import make_net
from analyze import measure_decision_distortion, measure_rate
from path import Path
from collections import namedtuple
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHECKPOINT_DIR = os.environ['MEMNET_CHECKPOINTS']
TRAININGSET_DIR = os.environ['MEMNET_TRAININGSETS']


def build(setsize, beta, dataset_size):
    vae_args = {
        "hidden": 500,
        "latent": 500,
        "activation": "tanh",
        "checkpoint_dir": CHECKPOINT_DIR,
        "trainingset_dir": TRAININGSET_DIR,
        "dataset": "gabor_array" + str(setsize),
        "dataset_size": dataset_size,
        "regenerate_steps": 10000,
        "batch": 128,
        "image_width": 120,
        "RGB": False,
        "task_weights": [1] * setsize,
        "decision_dim": setsize,
        "decision_size": 100,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "decision_layers": 1,
        "load_decision_weights": False,
        "load_memnet_weights": False,
        "rate_loss_weight": beta,
        "reconstruction_loss_weights": [0.01, 1],
        "decision_loss_weights": [1, 1],
        "beta0": None,
        "beta_steps": None,
        "regularizer_loss_weight": 1e-8,
        "sampling_off": False,
        "encode_probe": False,
        "layer_type": "MLP",
        "decision_target": "recall",
        "loss_func_decision": "squared_error",
        "loss_func_reconstruction": "squared_error",
        "mean": None,
        "std": None,
        "dim": None,
        "probe_noise_std": None,
        "alpha": None,
        "sample_distribution": "gaussian",
        "seqlen": None,
        "dropout_prob": 1.0
    }
    vae_args = namedtuple("arguments", vae_args.keys())(*vae_args.values())
    net = make_net(vae_args)
    net.build()
    return net


def plot(rates, distortions, setsize_beta_datasetsize):
    ss = [x[0] for x in setsize_beta_datasetsize]
    setsizes = sorted(set(ss))
    fig, ax = plt.subplots()
    for setsize in setsizes:
        # Plot rate versus distortion, grouped by set-size
        R = [r for r, s in zip(rates, ss) if s == setsize]
        D = [d / setsize for d, s in zip(distortions, ss) if s == setsize]
        ax.plot(R, D, label="set-size " + str(setsize), marker=".")
    save_dir = Path("plots/RD_curves_setsize/")
    ax.set_xlabel("Rate (nats)")
    ax.set_ylabel("Mean squared error (radians)")
    ax.set_ylim(0, max(D) * 1.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()
    if not save_dir.exists():
        save_dir.makedirs()
    save_pth = save_dir.joinpath("gabor_setsize.pdf")
    print("Saving figure to " + save_pth)
    plt.savefig(save_pth)


if __name__ == '__main__':
    setsize_beta_datasetsize = [
        (1, 0.1, 2000),
        (1, 0.5, 2000),
        (1, 1.0, 2000),
        (1, 2.0, 2000),
        (2, 0.1, 2000),
        (2, 0.5, 2000),
        (2, 1.0, 2000),
        (2, 2.0, 2000),
        (3, 0.1, 4000),
        (3, 0.5, 4000),
        (3, 1.0, 4000),
        (3, 2.0, 4000),
        (4, 0.1, 4000),
        (4, 0.5, 4000),
        (4, 1.0, 4000),
        (4, 2.0, 4000),
        (5, 0.1, 4000),
        (5, 0.5, 4000),
        (5, 1.0, 4000),
        (5, 2.0, 4000),
        (6, 0.1, 4000),
        (6, 0.5, 4000),
        (6, 1.0, 4000),
        (6, 2.0, 4000),
    ]

    rates = []
    distortions = []
    for setsize, beta, dataset_size in setsize_beta_datasetsize:
        net = build(setsize, beta, dataset_size)
        rates.append(measure_rate(net))
        distortions.append(measure_decision_distortion(net))
        del net
        tf.reset_default_graph()

    print(rates)
    print(distortions)
    plot(rates, distortions, setsize_beta_datasetsize)
