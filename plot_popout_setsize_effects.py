import os
from VAE_popout import make_net
from analyze import measure_decision_distortion, measure_rate
from path import Path
from collections import namedtuple
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHECKPOINT_DIR = os.environ['MEMNET_CHECKPOINTS']
TRAININGSET_DIR = os.environ['MEMNET_TRAININGSETS']


def build(search_type, beta, dataset_size):
    vae_args = {
        "activation": "relu",
        "checkpoint_dir": CHECKPOINT_DIR,
        "trainingset_dir": TRAININGSET_DIR,
        "dataset": search_type,
        "dataset_size": dataset_size,
        "regenerate_steps": 10000,
        "batch": 128,
        "rate_loss_weight": beta,
        "reconstruction_loss_weights": [1., 1.],
        "decision_loss_weights": [0.01, 1.],
        "beta0": None,
        "beta_steps": None,
        "regularizer_loss_weight": 1e-8,
        "sampling_off": False,
        "dropout_prob": 1.0
    }
    vae_args = namedtuple("arguments", vae_args.keys())(*vae_args.values())
    net = make_net(vae_args)
    net.build()
    return net


def plot(rates, distortions, searchtype_beta_datasetsize):
    search_types = [x[0] for x in searchtype_beta_datasetsize]
    search_type_set = sorted(set(search_types))
    fig, ax = plt.subplots()
    for search_type in search_type_set:
        # Plot rate versus distortion, grouped by set-size
        R = [r for r, s in zip(rates, search_types) if s == search_type]
        D = [d for d, s in zip(distortions, search_types) if s == search_type]
        if "both2" in search_type:
            label = "Conjunction"
        elif "both" in search_type:
            label = "Conjunction (complex)"
        else:
            label = "Single feature"
        ax.plot(R, D, label=label, marker=".")
    save_dir = Path("plots/RD_curves_setsize/")
    ax.set_xlabel("Rate (nats)")
    ax.set_ylabel("Mean squared error")
    ax.set_ylim(0, max(D) * 1.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend()
    if not save_dir.exists():
        save_dir.makedirs()
    save_pth = save_dir.joinpath("popout_setsize.pdf")
    print("Saving figure to " + save_pth)
    plt.savefig(save_pth)


searchtype_beta_datasetsize = [
    ("attention_search_shape", 0.01, 2000),
    ("attention_search_shape", 0.02, 2000),
    ("attention_search_shape", 0.03, 2000),
    ("attention_search_shape", 0.04, 2000),
    ("attention_search_shape", 0.05, 2000),
    ("attention_search_both2", 0.01, 2000),
    ("attention_search_both2", 0.02, 2000),
    ("attention_search_both2", 0.03, 2000),
    ("attention_search_both2", 0.04, 2000),
    ("attention_search_both2", 0.05, 2000),
    # ("attention_search_both", 0.01, 2000),
    # ("attention_search_both", 0.015, 2000),
    # ("attention_search_both", 0.02, 2000),
    # ("attention_search_both", 0.03, 2000),
    # ("attention_search_both", 0.04, 2000),
    # ("attention_search_both", 0.05, 2000),
]

rates = []
distortions = []
for search_type, beta, dataset_size in searchtype_beta_datasetsize:
    net = build(search_type, beta, dataset_size)
    rates.append(measure_rate(net))
    distortions.append(measure_decision_distortion(net))
    del net
    tf.reset_default_graph()

print(rates)
print(distortions)
plot(rates, distortions, searchtype_beta_datasetsize)
