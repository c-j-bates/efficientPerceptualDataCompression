from VAE import add_parser_args as add_vae_parser_args
from VAE import make_net as make_vae
from VAE_fruits import add_parser_args as add_fruits_parser_args
from VAE_fruits import make_net as make_fruits
from VAE_popout import add_parser_args as add_popout_parser_args
from VAE_popout import make_net as make_popout
from data_utils import (generate_training_data,
                        make_memnet_checkpoint_dir, load_plants,
                        make_dataset_pths, load_or_make_dataset)

import numpy as np
from itertools import product
from path import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_network_samples(net,
                              outputs_per_example=5,
                              color_map="gray",
                              setsize=[1, 6],
                              f_ext="pdf"):
    """Make grid of sample images from decoder
    """
    n_examples = net.batch_size
    if net.name == "cifar":
        inputs, targets = generate_training_data(
            None,
            None,
            dataset=net.dataset,
            train_test="test",
            conv=True,
            data_dir=net.trainingset_dir)[0:2]
        inputs, targets = net.get_batch(inputs, targets, None)[0:2]
        sig = True  # For predicting decisions
    elif net.name == "fruits":
        inputs, recall_targets = net.generate_batch("test")
        targets = inputs
        sig = True
    elif net.name == "popout":
        inputs, recall_targets = net.generate_training_data(n_examples)
        targets = inputs
        sig = False
    else:
        inputs = generate_training_data(
            n_examples,
            net.image_width,
            task_weights=net.task_weights,
            dataset=net.dataset,
            conv=net.layer_type == "conv",
            data_dir=net.trainingset_dir,
            setsize=setsize,
            mean=net.input_mean,
            std=net.input_std,
            dim=net.input_distribution_dim)[0]
        targets = inputs
        sig = False
    if net.image_channels > 1:
        shp = [net.image_width, net.image_width, net.image_channels]
    else:
        shp = [net.image_width, net.image_width]
    preds0, dec0 = zip(*[
        net.predict_both(
            inputs, np.zeros_like(inputs), sigmoid=sig, keep_session=True)
        for i in range(outputs_per_example)
    ])
    preds1 = np.array(preds0)
    preds2 = preds1.swapaxes(0, 1)
    preds = preds2.reshape([n_examples, outputs_per_example] + shp)
    preds = preds.clip(0, 1)
    # dec = np.array(dec0).swapaxes(0, 1)
    # print("decisions: ", dec)
    # print(dec)
    save_dir = make_memnet_checkpoint_dir(
        Path("plots/vis_memnet_samples/"), net)
    if not save_dir.exists():
        save_dir.makedirs()
    print("Saving images to: ", save_dir)
    for i in range(n_examples):
        fig, axes = plt.subplots(
            nrows=1, ncols=outputs_per_example + 1, figsize=(45, 10))
        if net.RGB:
            axes[0].imshow(targets[i], cmap=color_map)
        else:
            axes[0].imshow(
                targets[i].reshape(net.image_width, net.image_width),
                cmap=color_map)
        [
            axes[j + 1].imshow(preds[i][j], cmap=color_map)
            for j in range(outputs_per_example)
        ]
        [ax.set_aspect('equal') for ax in axes]
        xlabs = ["Target"] + [
            "Sample {}".format(j) for j in range(1, outputs_per_example + 1)
        ]
        [ax.set_xlabel(l, size=50) for ax, l in zip(axes, xlabs)]
        for ax in axes:
            ax.set_xticks([])
            ax.set_xticklabels("")
            ax.set_yticks([])
            ax.set_yticklabels("")
        plt.tight_layout()
        plt.savefig(
            save_dir.joinpath('setsize_{}_sample{}.{}'.format(
                "_".join([str(x) for x in setsize]),
                str(i).zfill(2), f_ext)))


def visualize_network_samples_grid(net, start=5, stop=100, step=10,
                                   f_ext="pdf"):
    """
    [PLANTS STIMULI ONLY]
    Plot grid of images, where (x, y) position corresponds to the stimulus
    values (width, droop) of the input to the network. Images are the outputs
    of the network given the input.
    """
    save_dir = make_memnet_checkpoint_dir(
        Path("plots/vis_memnet_samples/grid/"), net)
    if not save_dir.exists():
        save_dir.makedirs()
    inds = list(product(range(start, stop, step), range(start, stop, step)))
    grid_width = len(range(start, stop, step))
    print("Loading plants.")
    inputs, stim_vals = load_plants(
        net.image_width,
        net.trainingset_dir,
        layer_type=net.layer_type,
        normalize=True)
    print("Finished.")
    fig, axes = plt.subplots(grid_width, grid_width, figsize=(50, 50))
    preds = []
    print("Making plots...")
    for i in range(len(stim_vals)):
        if tuple(stim_vals[i]) in inds:
            print("Making ", stim_vals[i])
            preds.append(
                net.predict(inputs[i:i + 1], keep_session=True).reshape(
                    net.image_width, net.image_width))
            grid_coord = tuple(stim_vals[i] / step)
            ax = axes[int(grid_coord[1]), int(grid_coord[0])]
            # ax = axes[tuple(stim_vals[i] / step)]
            ax.imshow(preds[-1], cmap="gray_r")
            ax.set_xticklabels("")
            ax.set_xticks([])
            ax.set_xticklabels("")
            ax.set_yticks([])
            ax.set_yticklabels("")
    plt.savefig(save_dir.joinpath('grid.' + f_ext))
    print("Saved to ", save_dir.joinpath('grid.' + f_ext))
    return


def visualize_network_samples_grid_1D(net, start=5, stop=100, step=10,
                                      f_ext="pdf"):
    """
    [PLANTS STIMULI ONLY]
    Like the 2D version above, but just a single line of images (just one
    stimulus dimension).
    """
    save_dir = make_memnet_checkpoint_dir(
        Path("plots/vis_memnet_samples/grid/"), net)
    if not save_dir.exists():
        save_dir.makedirs()
    relevant_vals = range(start, stop, step)
    grid_width = len(relevant_vals)
    print("Loading plants.")
    inputs0, stim_vals0 = load_plants(
        net.image_width,
        net.trainingset_dir,
        layer_type=net.layer_type,
        normalize=True)
    inputs0 = inputs0 / 255.  # Normalize
    print("Finished.")
    irrelevant_dim = int(net.input_distribution_dim == 0)
    if irrelevant_dim == 0:
        inds = list(product([50], relevant_vals))
    else:
        inds = list(product(relevant_vals, [50]))
    inputs_valid = inputs0[stim_vals0[:, irrelevant_dim] == 50]
    stim_vals = stim_vals0[stim_vals0[:, irrelevant_dim] == 50]
    fig, axes = plt.subplots(
        1, grid_width, figsize=(50, 50 / len(relevant_vals)))
    preds = []
    inputs = []
    print("Making plots...")
    for i in range(len(stim_vals)):
        if tuple(stim_vals[i]) in inds:
            print("Making ", stim_vals[i])
            inputs.append(inputs_valid[i])
    preds = net.predict(inputs, keep_session=True)
    for i in range(len(preds)):
        ax = axes[i]
        ax.imshow(
            preds[i:i + 1].reshape(net.image_width, net.image_width),
            cmap="gray_r")
        ax.set_xticklabels("")
        ax.set_xticks([])
        ax.set_xticklabels("")
        ax.set_yticks([])
        ax.set_yticklabels("")
    plt.savefig(save_dir.joinpath('grid.' + f_ext))
    print("Saved to ", save_dir.joinpath('grid.' + f_ext))
    return


def plants_average_reconstruction(net,
                                  plant_coords=[[0, 0], [0, 99], [99, 99],
                                                [99, 0]],
                                  n_samples=100,
                                  f_ext="pdf"):
    """For the specified plants (given by `plant_coords`), plot
    the target image alongside the mean reconstructed image, averaged
    over `n_samples` samples.
    """
    images, stim_vals = load_plants(
        net.image_width,
        net.trainingset_dir,
        layer_type=net.layer_type,
        normalize=True)
    save_dir = make_memnet_checkpoint_dir(
        Path("plots/plants_average_reconstruction/"), net)
    if not save_dir.exists():
        save_dir.mkdir()

    for coord in plant_coords:
        # Iterate over all stimulus values for target image
        i_target = np.logical_and(coord[0] == stim_vals[:, 0],
                                  coord[1] == stim_vals[:, 1])
        x = images[i_target]
        X = np.repeat(x, n_samples, axis=0)  # Take n samples
        Xhat = net.predict(X, keep_session=True)
        mean_reconstruction = Xhat.mean(axis=0)  # Take mean over samples
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5.6, 3.1))
        ax[0].imshow(
            x.reshape(net.image_width, net.image_width), cmap="gray_r")
        ax[1].imshow(
            mean_reconstruction.reshape(net.image_width, net.image_width),
            cmap="gray_r")
        ax[0].set_xlabel("Target", fontsize=14)
        ax[1].set_xlabel("Mean reconstruction", fontsize=14)
        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[0].set_xticklabels("")
        ax[1].set_xticklabels("")
        ax[0].set_yticks([])
        ax[1].set_yticks([])
        ax[0].set_yticklabels("")
        ax[1].set_yticklabels("")
        fig.suptitle("Leaf width {}, leaf angle {}".format(coord[0], coord[1]))
        plt.tight_layout()
        # plt.subplots_adjust(top=0.9)
        save_pth = save_dir.joinpath("width{}_angle{}.{}".format(
            coord[0], coord[1], f_ext))
        plt.savefig(save_pth, dpi=300)
    print("Saved to " + save_dir)
    return


def correlation(net, start=0, stop=100, step=10, n_samples=2, f_ext="pdf"):
    """An analysis for networks trained with non-uniform input distributions.
    (Designed for 'plants' experiment)

    Correlate reconstructed images with all original stimulus images.
    Since one plant dimension will be uniform and one will be non-uniform,
    marginalize over the non-uniform dimension

    n_samples: Number of network samples to average over
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import auc
    # Save directory
    save_dir = make_memnet_checkpoint_dir(Path("plots/correlation/"), net)
    if not save_dir.exists():
        save_dir.mkdir()
    dim = net.input_distribution_dim
    if dim == -1:
        raise Exception
    other_dim = 1 if dim == 0 else 0
    images, stim_vals = load_plants(
        net.image_width,
        net.trainingset_dir,
        layer_type=net.layer_type,
        normalize=True)
    target_selection = range(start, stop, step)  # Select only subset of values
    probe_selection = range(0, 100, 1)
    unique_stim_vals = 100
    corrs = np.zeros((len(target_selection), len(probe_selection)))
    xhats = np.zeros((len(target_selection), len(probe_selection),
                      unique_stim_vals, images.shape[1]))
    A_left = []
    A_right = []
    for i, dim_t in enumerate(target_selection):
        # Iterate over all stimulus values for target image
        inds_x = [
            np.logical_and(dim_t == stim_vals[:, dim],
                           d == stim_vals[:, other_dim])
            for d in range(unique_stim_vals)
        ]
        x = np.vstack([images[ix] for ix in inds_x])
        for j, dim_p in enumerate(probe_selection):
            # Iterate over all stimulus values for probe image
            inds_y = [
                np.logical_and(dim_p == stim_vals[:, dim],
                               d == stim_vals[:, other_dim])
                for d in range(unique_stim_vals)
            ]
            y = np.vstack([images[iy] for iy in inds_y])
            xhats[i, j] = np.mean(
                [net.predict(x, keep_session=True) for _ in range(n_samples)],
                axis=0)
            corrs[i, j] = np.mean([
                pearsonr(xhats[i, j][k], y[k])[0]
                for k in range(unique_stim_vals)
            ])
            print(i, j)
        i_true_target_val = probe_selection.index(
            dim_t
            # Note, probe_selection must include all the target values for this
            # to work
        )
        rAleft = corrs[i, max(0, i_true_target_val - 10):i_true_target_val + 1]
        rAright = corrs[i, i_true_target_val:i_true_target_val + 11]
        if len(rAleft) > 1:
            A_left.append(auc(range(len(rAleft)), rAleft))
        else:
            A_left.append(0)
        if len(rAright) > 1:
            A_right.append(auc(range(len(rAright)), rAright))
        else:
            A_right.append(0)
    cutoff = 100
    fig, ax = plt.subplots()
    for i in range(len(target_selection)):
        ind_t = probe_selection.index(target_selection[i])
        line = ax.plot(probe_selection[max(0, ind_t - cutoff):ind_t + cutoff],
                       corrs[i][max(0, ind_t - cutoff):ind_t + cutoff])
        color = line[0].get_color()
        ax.axvline(x=target_selection[i], color=color, linestyle="--")
    ax.set_ylim(0.6, 1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("stimulus value", fontsize=16)
    ax.set_ylabel("correlation coefficient", fontsize=16)
    save_pth = save_dir.joinpath("correlation_dim{}_mean{}_std{}.{}".format(
        net.input_distribution_dim, net.input_mean, net.input_std, f_ext))
    plt.savefig(save_pth)
    print("Saved to ", save_pth)
    return


def measure_rate(net, n_samples=10):
    """Use monte carlo approximation of network's rate, as given by equation
    2 in https://openreview.net/pdf?id=H1rRWl-Cb
    """
    from scipy.stats import multivariate_normal as mn

    dataset_dir, dataset_pth = make_dataset_pths(net)
    print("dataset path: ", dataset_pth)
    X, Probes, Change_prob, Perceptual_dist = load_or_make_dataset(
        net, dataset_pth, dataset_dir, net.dataset_size)
    Rx = []
    mu, logsigma = [], []
    i = 0
    while True:
        if i * net.batch_size > len(X) - 1:
            break
        x = X[i * net.batch_size:(i + 1) * net.batch_size]
        m, ls = net.encode_presample(x, keep_session=True)
        mu.append(m)
        logsigma.append(ls)
        i += 1
    mu = np.vstack(mu)
    logsigma = np.vstack(logsigma)
    print("Dataset encoded.")
    z = [
        mn.rvs(mean=mu[j], cov=np.diag(np.exp(logsigma[j])), size=n_samples)
        for j in range(len(mu))
    ]
    p_z_given_x = [
        mn.logpdf(z[j], mean=mu[j], cov=np.diag(np.exp(logsigma[j])))
        for j in range(len(mu))
    ]
    m_z = [
        mn.logpdf(
            z[j], mean=np.zeros(net.latent_size), cov=np.eye(net.latent_size))
        for j in range(len(mu))
    ]
    Rx = [(p_z_given_x[j] - m_z[j]).sum() / n_samples for j in range(len(mu))]
    R = sum(Rx) / len(X)
    print(R)
    return R


def measure_pixel_distortion(net, n_samples=10):
    """Use monte carlo approximation to estimate pixel-wise distortion
    over dataset.
    """
    dataset_dir, dataset_pth = make_dataset_pths(net)
    print("dataset path: ", dataset_pth)
    X0, Probes, Change_prob, Perceptual_dist = load_or_make_dataset(
        net, dataset_pth, dataset_dir, net.dataset_size)
    X = np.repeat(X0, n_samples, axis=0)
    Y = net.predict(X)
    D = np.sum((Y - X) ** 2 / len(X))
    print(D)
    return D


def measure_decision_distortion(net, n_samples=10):
    """Use monte carlo approximation to estimate decision distortion
    over dataset.
    """
    dataset_dir, dataset_pth = make_dataset_pths(net)
    print("dataset path: ", dataset_pth)
    X0, Probes, Change_prob, Recall_targets0 = load_or_make_dataset(
        net, dataset_pth, dataset_dir, net.dataset_size)
    X = np.repeat(X0, n_samples, axis=0)
    Recall_targets = np.repeat(Recall_targets0, n_samples, axis=0)
    Y = []
    i = 0
    while True:
        if i * net.batch_size > len(X) - 1:
            break
        x = X[i * net.batch_size:(i + 1) * net.batch_size]
        Y.append(net.predict_response(x, np.zeros_like(x), sigmoid=False,
                                      keep_session=True))
        i += 1
    Y = np.vstack(Y)
    D = np.sum((Y - Recall_targets)**2) / len(X)
    print(D)
    return D


def add_analysis_args(parser):
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Make images of output samples given targets (mem"
        "net).")
    parser.add_argument(
        "--visualize_grid",
        action="store_true",
        help="Visualize network reconstructions at evenly-"
        "spaced stimulus values along a grid.")
    parser.add_argument(
        "--grid_start",
        type=int,
        default=5,
        help="[FOR VISUALIZE_GRID, CORRELATION] Lowest "
        "stimulus value")
    parser.add_argument(
        "--grid_stop",
        type=int,
        default=100,
        help="[FOR VISUALIZE_GRID, CORRELATION] Highest "
        "stimulus value")
    parser.add_argument(
        "--grid_step",
        type=int,
        default=10,
        help="[FOR VISUALIZE_GRID, CORRELATION] Increment "
        "between images")
    parser.add_argument(
        "--correlation",
        action="store_true",
        help="Get mean correlation values between "
        "reconstructed images for given targets and all "
        "original stimulus images.")
    parser.add_argument(
        "--plants_average_reconstruction",
        action="store_true",
        help="Create plots with pairs of images, where each "
        "pair consists of a true target image and the average"
        "reconstructed image conditioned on that target.")
    parser.add_argument(
        "--plant_coords",
        type=str,
        default="[[0, 0], [0, 99], [99, 99], [99, 0]]",
        help="Which coordinates to use for "
        "plants_average_reconstruction")
    parser.add_argument(
        "--vis_setsize",
        type=int,
        nargs="*",
        default=[1, 6],
        help="When visualizing network samples for"
        "object arrays, range of set sizes to sample from")
    parser.add_argument(
        "--colormap",
        type=str,
        default=None,
        help="Color map for visualizing network samples")
    parser.add_argument(
        "--rate",
        action="store_true",
        help="Estimate the mutual information between input "
        "and latent activations over dataset")
    parser.add_argument(
        "--pixel_distortion",
        action="store_true",
        help="Estimate pixel-wise distortion over dataset.")
    parser.add_argument(
        "--decision_distortion",
        action="store_true",
        help="Estimate response (decision) distortion over "
        "dataset.")
    parser.add_argument("--f_ext", type=str, default="pdf")


def make_parser(name, parser_func):
    parser = subparsers.add_parser(name)
    parser = parser_func(parser)
    parser = add_analysis_args(parser)
    return parser


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")
    parser_vae = make_parser("vae", add_vae_parser_args)
    parser_fruits = make_parser("fruits", add_fruits_parser_args)
    parser_popout = make_parser("popout", add_popout_parser_args)
    args = parser.parse_args()
    if args.mode == "vae":
        net = make_vae(args)
        net.build()
    elif args.mode == "fruits":
        net = make_fruits(args)
        net.build()
    elif args.mode == "popout":
        net = make_popout(args)
        net.build()
    else:
        raise NotImplementedError
    print("Checkpoint directory: " + net.checkpoint_dir)
    if args.visualize:
        visualize_network_samples(
            net,
            setsize=args.vis_setsize,
            color_map=args.colormap,
            f_ext=args.f_ext)
    if args.visualize_grid:
        if "1D" in net.dataset:
            visualize_network_samples_grid_1D(
                net,
                args.grid_start,
                args.grid_stop,
                args.grid_step,
                f_ext=args.f_ext)
        else:
            visualize_network_samples_grid(
                net,
                args.grid_start,
                args.grid_stop,
                args.grid_step,
                f_ext=args.f_ext)
    if args.correlation:
        correlation(net, args.grid_start, args.grid_stop, args.grid_step,
                    f_ext=args.f_ext)
    if args.plants_average_reconstruction:
        plants_average_reconstruction(
            net, f_ext=args.f_ext, plant_coords=eval(args.plant_coords))
    if args.rate:
        measure_rate(net)
    if args.pixel_distortion:
        measure_pixel_distortion(net)
    if args.decision_distortion:
        measure_decision_distortion(net)
