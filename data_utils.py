import copy
from itertools import product
import numpy as np
from path import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import truncnorm, norm
from scipy import fftpack


# Dataset categories
FILTERS = ["pink", "bandpass", "pink_addednoiseprobe",
           "bandpass_addednoiseprobe"]
TASK_WEIGHTED = [
    # "rectangle",
    # "letter",
    # "episodic_shape_race",
    # "episodic_setsize",
    "plants",
]
IMAGES = FILTERS + [
    "plants",
    "plants_setsize",
    "plants_setsize1",
    "plants_setsize2",
    "plants_setsize3",
    "plants_setsize4",
    "plants_setsize5",
    "plants_setsize6",
    "plants_categorical",
    "plants_modal_prior",
    "plants_modal_prior_1D",
    # "rectangle",
    # "letter",
    # "object_array",
    "gabor_array",
    "gabor_array1",
    "gabor_array2",
    "gabor_array3",
    "gabor_array4",
    "gabor_array5",
    "gabor_array6",
    # "episodic_shape_race",
    # "episodic_setsize",
    # "attention_MOT",
    # "cifar10",
    # "places365",
    "fruits360"
]
NONIMAGE = ["2Dfloat", "NDim_float"]
# Cannot auto-generate (near-)infinite number of stimuli:
FINITE_SET = [
    "plants",
    "plants_setsize",
    "plants_setsize1",
    "plants_setsize2",
    "plants_setsize3",
    "plants_setsize4",
    "plants_setsize5",
    "plants_setsize6",
    "plants_categorical",
    "plants_modal_prior",
    "plants_modal_prior_1D",
    # "rectangle",
    # "cifar10",
]
NATURAL_IMAGES = [
    # "cifar10",
    # "places365",
    "fruits360"
]
FIXED_ARCHITECTURE = NATURAL_IMAGES + [
    # For these tasks, I've decided to fix the network's design
    # because finding an architecture that works at all is difficult
    "attention_search",
    "attention_search_shape",
    "attention_search_color",
    "attention_search_both",
    "attention_search_both2"
]
XENTROPY = [
    # "rectangle"
]
BINARY = [
    # "rectangle"
]
RECURRENT = [
    # "episodic_shape_race",
    # "episodic_setsize"
]
ALL = list(
    set(
        FILTERS +
        TASK_WEIGHTED +
        IMAGES +
        NONIMAGE +
        FINITE_SET +
        XENTROPY +
        BINARY +
        RECURRENT +
        NATURAL_IMAGES +
        FIXED_ARCHITECTURE
    )
)
TASKS = {
    "all": ALL,
    "filters": FILTERS,
    "has_task_weights": TASK_WEIGHTED,
    "images": IMAGES,
    "nonimage": NONIMAGE,
    "finite_set": FINITE_SET,
    "xentropy": XENTROPY,
    "binary": BINARY,
    "recurrent": RECURRENT,
    "natural_images": NATURAL_IMAGES,
}


def open_rgb(img_file, pad=0):
        img = Image.open(img_file)
        if len(np.array(img).shape) == 2:
            # Convert grayscale images to RGB
            w, h = img.size
            rgbimg = Image.new("RGB", (w + pad, h + pad))
            rgbimg.paste(img)
            return rgbimg
        elif len(np.array(img).shape) == 3:
            if pad > 0:
                w, h = img.size
                rgbimg = Image.new("RGB", (w + pad, h + pad))
                rgbimg.paste(img)
                return rgbimg
            else:
                return img
        else:
            raise Exception("Not a proper image.")


def make_memnet_checkpoint_dir(base_dir, net):
    if net.__dict__.get("input_distribution_dim") is not None:
        if net.input_distribution_dim != -1:
            inputdist_str = "_inputdist{}_{}_{}".format(
                net.input_distribution_dim, net.input_mean, net.input_std
            )
        else:
            inputdist_str = ""
    if net.dataset == "plants_categorical" \
            or net.dataset == "plants_modal_prior_1D" \
            or net.dataset == "plants_modal_prior" \
            or net.dataset == "plants_setsize":
        # This option differs from IMAGES dataset category below only in the
        # last parameter, net.input_distribution_dim, which specifies for
        # plants dataset which dimension to operate over (leaf width or angle).
        # The other dimension is fixed at some value in the training set.
        pth = Path(base_dir).joinpath(
            "{}_width{}_{}_lossweights{}_{}_{}_{}_{}_hidden{}_latent{}"
            "_decision{}_layers{}_{}_{}_reconloss_{}_dim{}".format(
                net.dataset,
                net.image_width,
                net.layer_type,
                net.w_rate,
                net.w_reconstruction_enc,
                net.w_reconstruction_dec,
                net.w_decision_enc,
                net.w_decision_dec,
                net.hidden_size,
                net.latent_size,
                net.decision_size,
                net.encoder_layers,
                net.decoder_layers,
                net.decision_layers,
                net.loss_func_recon,
                net.input_distribution_dim
            )
        )
    elif net.dataset in FIXED_ARCHITECTURE:
        pth = Path(base_dir).joinpath(
            "{}_lossweights{}_{}_{}_{}_{}".format(
                net.dataset,
                net.w_rate,
                net.w_reconstruction_enc,
                net.w_reconstruction_dec,
                net.w_decision_enc,
                net.w_decision_dec,
            )
        )
    elif net.dataset in TASK_WEIGHTED:
        pth = Path(base_dir).joinpath(
            "{}_width{}_{}_lossweights{}_{}_{}_{}_{}_hidden{}_latent{}_decision"
            "{}_layers{}_{}_{}_taskweights{}{}_losses_{}_{}_dectarget_{}".format(
                net.dataset,
                net.image_width,
                net.layer_type,
                net.w_rate,
                net.w_reconstruction_enc,
                net.w_reconstruction_dec,
                net.w_decision_enc,
                net.w_decision_dec,
                net.hidden_size,
                net.latent_size,
                net.decision_size,
                net.encoder_layers,
                net.decoder_layers,
                net.decision_layers,
                "_".join([str(d) for d in net.task_weights]),
                inputdist_str,
                net.loss_func_recon,
                net.loss_func_dec,
                net.decision_target,
            )
        )
    elif net.dataset in FILTERS:
        pth = Path(base_dir).joinpath(
            "{}_width{}_{}_lossweights{}_{}_{}_{}_{}_hidden{}_latent{}_decision"
            "{}_layers{}_{}_{}{}".format(
                net.dataset + str(net.probe_noise_std),
                net.image_width,
                net.layer_type,
                net.w_rate,
                net.w_reconstruction_enc,
                net.w_reconstruction_dec,
                net.w_decision_enc,
                net.w_decision_dec,
                net.hidden_size,
                net.latent_size,
                net.decision_size,
                net.encoder_layers,
                net.decoder_layers,
                net.decision_layers,
                inputdist_str,
            )
        )
    elif net.dataset in IMAGES:
        pth = Path(base_dir).joinpath(
            "{}_width{}_{}_lossweights{}_{}_{}_{}_{}_hidden{}_latent{}_decision"
            "{}_layers{}_{}_{}_reconloss_{}".format(
                net.dataset,
                net.image_width,
                net.layer_type,
                net.w_rate,
                net.w_reconstruction_enc,
                net.w_reconstruction_dec,
                net.w_decision_enc,
                net.w_decision_dec,
                net.hidden_size,
                net.latent_size,
                net.decision_size,
                net.encoder_layers,
                net.decoder_layers,
                net.decision_layers,
                net.loss_func_recon,
            )
        )
    elif net.dataset in NONIMAGE:
        pth = Path(base_dir).joinpath(
            "{}_inpdim{}_lossweights{}_{}_{}_{}_{}_hidden{}_latent{}_decision{}_"
            "layers{}_{}_{}_taskweights{}_reconloss_{}".format(
                net.dataset,
                net.image_width,
                net.w_rate,
                net.w_reconstruction_enc,
                net.w_reconstruction_dec,
                net.w_decision_enc,
                net.w_decision_dec,
                net.hidden_size,
                net.latent_size,
                net.decision_size,
                net.encoder_layers,
                net.decoder_layers,
                net.decision_layers,
                "_".join([str(d) for d in net.task_weights]),
                net.loss_func_recon,
            )
        )
    else:
        raise NotImplementedError
    return pth


# DEPRECATED
# def make_remap_checkpoint_dir(base_dir, dataset, w_rate, w_reconstruction,
#                               w_decision, hidden, memnet_hidden, memnet_latent,
#                               memnet_decision, task_weights):
#     return Path(base_dir).joinpath(
#         "{}_memnetlossweights{}_{}_{}_hidden{}_memnethidden{}_memnetlatent{}"
#         "_memnetdecision{}_taskweights{}".format(
#             dataset, w_rate, w_reconstruction, w_decision, hidden,
#             memnet_hidden, memnet_latent, memnet_decision,
#             "_".join([str(d) for d in task_weights])))


def make_dataset_pths(net):
    if net.dataset == "2Dfloat":
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_taskweights{}_size{}/".format(
                net.dataset,
                "_".join([str(t) for t in net.task_weights]),
                net.dataset_size,
            )
        )
    elif net.dataset == "plants_categorical" \
            or net.dataset == "plants_modal_prior_1D" \
            or net.dataset == "plants_modal_prior":
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_width{}_size{}_dim{}/".format(
                net.dataset + str(net.probe_noise_std),
                net.image_width,
                net.dataset_size,
                net.input_distribution_dim
            )
        )
    elif net.dataset in FIXED_ARCHITECTURE:
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_size{}".format(
                net.dataset,
                net.dataset_size))
    elif net.dataset == "cifar10":
        dataset_dir = net.trainingset_dir.joinpath("cifar-10-batches-py/")
    elif net.dataset in FILTERS:
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_width{}_size{}/".format(
                net.dataset + str(net.probe_noise_std),
                net.image_width,
                net.dataset_size,
            )
        )
    elif net.dataset in IMAGES and net.dataset not in TASK_WEIGHTED:
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_width{}_size{}/".format(
                net.dataset + str(net.probe_noise_std),
                net.image_width,
                net.dataset_size,
            )
        )
    elif net.dataset in IMAGES and net.dataset in TASK_WEIGHTED:
        if net.__dict__.get("input_distribution_dim") not in [-1, None]:
            input_dist_str = "_mean{}_std{}_dim{}".format(
                net.input_mean, net.input_std, net.input_distribution_dim
            )
        else:
            input_dist_str = ""
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_width{}_taskweights{}{}_size{}_{}/".format(
                net.dataset + str(net.probe_noise_std),
                net.image_width,
                "_".join([str(t) for t in net.task_weights]),
                input_dist_str,
                net.dataset_size,
                "logistic" if net.logistic_decision else "tpdist",
            )
        )
    # Override above if recurrent
    if net.dataset in RECURRENT:
        input_dist_str = ""
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_width{}_seqlen{}_taskweights{}{}_size{}_{}/".format(
                net.dataset + str(net.probe_noise_std),
                net.image_width,
                net.seqlen,
                "_".join([str(t) for t in net.task_weights]),
                input_dist_str,
                net.dataset_size,
                "logistic" if net.logistic_decision else "tpdist",
            )
        )
    dataset_pth = dataset_dir.joinpath("data.pkl")
    return dataset_dir, dataset_pth


def load_or_make_dataset(net, dataset_pth, dataset_dir, dataset_size):
    import pickle
    if not hasattr(net, "task_weights"):
        net.task_weights = None
    if not hasattr(net, "layer_type"):
        net.layer_type = None
    if not hasattr(net, "input_mean"):
        net.input_mean = None
    if not hasattr(net, "input_std"):
        net.input_std = None
    if not hasattr(net, "input_distribution_dim"):
        net.input_distribution_dim = None
    if not hasattr(net, "probe_noise_std"):
        net.probe_noise_std = None
    if not hasattr(net, "seqlen"):
        net.seqlen = None

    if dataset_pth.exists():
        print("Loading dataset...")
        with dataset_pth.open("rb") as fid:
            Data = pickle.load(fid)
        print("Dataset loaded.")
        if len(Data) == 4:
            X, Probes, Change_prob, Recall_targets = Data
        elif len(Data) == 2:
            X, Recall_targets = Data
            Probes = None
            Change_prob = None
        else:
            raise Exception("Loaded data was unexpected length.")
        # Reshape data if necessary
        if net.dataset in RECURRENT:
            raveled_dims = 3
        else:
            raveled_dims = 2
        if len(X.shape) == raveled_dims and net.layer_type == "conv":
            if net.dataset in RECURRENT:
                # For recurrent datasets, include temporal dimension
                X = X.reshape(
                    -1, net.seqlen, net.image_width, net.image_width, net.image_channels
                )
                if Probes is not None:
                    Probes = Probes.reshape(
                        -1, net.seqlen, net.image_width, net.image_width, net.image_channels
                    )
            else:
                # For non-recurrent, exclude temporal dimension
                X = X.reshape(-1, net.image_width, net.image_width, net.image_channels)
                if Probes is not None:
                    Probes = Probes.reshape(
                        -1, net.image_width, net.image_width, net.image_channels
                    )
        if len(X.shape) > raveled_dims and net.layer_type == "MLP":
            X = X.reshape(-1, net.image_width * net.image_width)
            if Probes is not None:
                Probes = Probes.reshape(-1, net.image_width * net.image_width)
    else:
        print("Generating dataset...")
        Data = generate_training_data(
            dataset_size,
            net.image_width,
            dataset=net.dataset,
            conv=net.layer_type == "conv",
            task_weights=net.task_weights,
            data_dir=net.trainingset_dir,
            mean=net.input_mean,
            std=net.input_std,
            dim=net.input_distribution_dim,
            probe_noise_std=net.probe_noise_std,
            seqlen=net.seqlen,
        )
        if not dataset_dir.exists():
            dataset_dir.makedirs()
        with dataset_pth.open("wb") as fid:
            pickle.dump(Data, fid)
        if len(Data) == 2:
            X, Recall_targets = Data
            Probes = None
            Change_prob = None
        elif len(Data) == 4:
            X, Probes, Change_prob, Recall_targets = Data
        else:
            raise Exception("generate_training_data produced the wrong length")
    return X, Probes, Change_prob, Recall_targets


def data_recur2conv(X):
    """Reshape recurrent data for feedforward convolutional layers
    """
    if X.ndim != 5:
        raise Exception(
            "In reshaping recurrent data, expected data of length"
            "5, but got length " + str(X.ndim)
        )
    seqlen = X.shape[1]
    X = np.split(X, seqlen, axis=1)
    X = [x.squeeze(axis=1) for x in X]
    return np.concatenate(X, axis=-1)


def data_conv2recur(X, channels=3):
    """Inverse of data_recur2conv
    """
    if X.ndim != 4:
        raise Exception(
            "In reshaping conv feedforward data, expected data of"
            "length 4, but got length " + str(X.ndim)
        )
    if X.shape[-1] % channels:
        raise Exception(
            "In reshaping conv feedforward data, incompatible"
            "number of channels. Last dim of X must be multiple of"
            "of channels."
        )
    seqlen = X.shape[-1] / channels
    X = np.stack(np.split(X, seqlen, axis=-1), axis=1)
    return X


def calc_decision_params(max_dist, min_prob_change=0.0001, max_prob_change=1 - 1e-8):
    """Solve for logistic sigmoid parameters (bias and scale) with two anchor
    points: left-most is x=0 and right-most is user-specified (max_dist).
    min_prob_change and max_prob_change are the values of the function at
    each anchor point, respectively.

    sigm(x) = 1 / (1 + e^-(x * scale + bias))

    b = -log(1/p - 1) - x * scale
    scale = (-log(1/p -1) - b) / x

    where x=0 when p=min_prob_change, and
          x=max_dist when p=max_prob_change
    """
    bias = -np.log(1.0 / min_prob_change - 1.0)
    scale = (-np.log(1.0 / max_prob_change - 1.0) - bias) / max_dist
    return bias, scale


# def square_or_triangle(N, size):
#     """Make training examples that are images composed of polygons.
#     The polygon category, size, and location are drawn randomly for each
#     example.
#     Categories:
#     0: square
#     1: triangle
#     """
#     max_side_len = size
#     shapes = np.random.choice(range(2), replace=True, size=N)
#     side_lens = np.random.choice(range(2, max_side_len - 1), replace=True, size=N)
#     X0 = []
#     Y0 = []
#     X0_rml = []
#     Y0_rml = []
#     images = []
#     images_rml = []
#     for shp, side_len in zip(shapes, side_lens):
#         x0, y0 = np.random.choice(range(size - side_len + 1), replace=True, size=2)
#         # TODO: Generalize (currently just picks new location, uniform random)
#         x0_rml, y0_rml = np.random.choice(
#             range(size - side_len + 1), replace=True, size=2
#         )
#         X0_rml.append(x0_rml)
#         Y0_rml.append(y0_rml)
#         X0.append(x0)
#         Y0.append(y0)
#         img = np.zeros((size, size))
#         img_rml = np.zeros((size, size))
#         if shp == 0:
#             img[x0 : x0 + side_len, y0 : y0 + side_len] = 1
#             img_rml[x0_rml : x0_rml + side_len, y0_rml : y0_rml + side_len] = 1
#             # if img.sum() != side_len ** 2:
#             #     print "out of bounds"
#         elif shp == 1:
#             # Triangle
#             for i in range(side_len):
#                 l = side_len - 2 * i
#                 y0i = y0 + i
#                 x0i = x0 + i
#                 y0i_rml = y0_rml + i
#                 x0i_rml = x0_rml + i
#                 img[x0i, y0i : y0i + l] = 1
#                 img_rml[x0i_rml, y0i_rml : y0i_rml + l] = 1
#         images.append(img)
#         images_rml.append(img_rml)
#     X0 = np.array(X0)
#     Y0 = np.array(Y0)
#     X0_rml = np.array(X0_rml)
#     Y0_rml = np.array(Y0_rml)
#     images = np.array(images)
#     images_rml = np.array(images_rml)
#     return shapes, X0, Y0, side_lens, images, images_rml


# def rectangles_grid(size=30):
#     """Make all possible stimuli along both stimulus dimensions.
#     """
#     min_l = 1
#     max_l = size / 3
#     L = range(min_l, max_l + 1)
#     pos = [[2, 2], [size / 2, 2], [2, size / 2], [size / 2, size / 2]]
#     Images = []
#     for l0 in L:
#         images = []
#         for l1 in L:
#             img = np.zeros((size, size))
#             for px, py in pos:
#                 img[px : px + l0, py : py + l1] = 1
#             images.append(img)
#         Images.append(images)
#     return np.array(Images)


# def rectangles(N, size=30, task_weights=[1 / 3.0, 1 / 3.0, 1 / 3.0]):
#     # np.set_printoptions(linewidth=160)
#     """Each scene consists of a single rectangle. Each side length determines
#     the value along the two stimulus dimensions,
#     respectively. In other words, the generative process is: First choose
#     side-length 1, then side-length 2, then randomly place the "leaves".
#     task_weights: probability that probe varies along each dimension on a give
#     change trial (3 values, see below).
#     """
#     task_weights = np.array(task_weights) / float(sum(task_weights))
#     # Min and max side lengths for leaves
#     min_l = 1
#     max_l = size / 3
#     L = range(min_l, max_l + 1)
#     l0 = np.random.choice(L, size=N)
#     l1 = np.random.choice(L, size=N)
#     change_trial = np.random.choice(2, size=N)
#     # pos = np.random.choice(range(0, size - 3), size=(N, n_leaves, 2))
#     pos = [[2, 2], [size / 2, 2], [2, size / 2], [size / 2, size / 2]]
#     # deltas = [_ for _ in range(-4, 5) if _ != 0]
#     # probs = norm.pdf(deltas, loc=0, scale=4)
#     # probs /= probs.sum()
#     target_images = []
#     probe_images = []
#     for i in range(N):
#         img = np.zeros((size, size))
#         img_probe = np.zeros((size, size))
#         if change_trial[i]:
#             # Change trial (either l0 changes, l1 changes, or both)
#             change_type = np.random.choice(3, p=task_weights)
#             if change_type == 0:
#                 change_l0 = 1
#                 change_l1 = 0
#             if change_type == 1:
#                 change_l0 = 0
#                 change_l1 = 1
#             if change_type == 2:
#                 change_l0 = 1
#                 change_l1 = 1
#             if change_l0:
#                 valid_probe_vals_l0 = [l for l in L if l != l0[i]]
#                 l0_probe = np.random.choice(valid_probe_vals_l0)
#             else:
#                 l0_probe = l0[i]
#             if change_l1:
#                 valid_probe_vals_l1 = [l for l in L if l != l1[i]]
#                 l1_probe = np.random.choice(valid_probe_vals_l1)
#             else:
#                 l1_probe = l1[i]

#         else:
#             # Same trial
#             l0_probe = l0[i]
#             l1_probe = l1[i]
#         for px, py in pos:
#             px = px + np.random.choice([-2, -1, 0, 1, 2])
#             py = py + np.random.choice([-2, -1, 0, 1, 2])
#             img[px : px + l0[i], py : py + l1[i]] = 1
#             img_probe[px : px + l0_probe, py : py + l1_probe] = 1
#         target_images.append(img)
#         probe_images.append(img_probe)
#     return np.array(target_images), np.array(probe_images), change_trial


# def rectangle_grid(size=30, layer_type="MLP"):
#     """Make all possible stimuli along both stimulus dimensions.
#     """
#     L = range(1, size + 1)
#     Images = []
#     for l0 in L:
#         images = []
#         for l1 in L:
#             img = np.zeros((size, size))
#             img[0:l0, 0:l1] = 1
#             images.append(img)
#         Images.append(images)
#     if layer_type == "MLP":
#         Images = np.reshape(images, (L, L, -1))  # Flatten along pixels
#     return Images


# def rectangle(N, size=30, task_weights=[1 / 3.0, 1 / 3.0, 1 / 3.0]):
#     """Each image contains a single rectangle whose upper-left-most point
#     is anchored at the image corner.
#     """
#     max_dist = 2 * size
#     probe_sample_var = size / 10.0
#     decision_bias, decision_scale = calc_decision_params(
#         max_dist / (probe_sample_var * 3.0)
#     )
#     task_weights = np.array(task_weights) / float(max(task_weights))
#     perceptual_weights = task_weights[:2] / task_weights[:2].sum()
#     L = range(1, size + 1)
#     d0 = np.random.choice(L, size=N)  # Dimension 0
#     d1 = np.random.choice(L, size=N)  # Dimension 1
#     d0_probes = []
#     d1_probes = []
#     target_images = []
#     probe_images = []
#     prob_change = []
#     for i in range(N):
#         img = np.zeros((size, size))
#         img_probe = np.zeros((size, size))
#         tn0 = truncnorm(a=-d0[i], b=size - d0[i], loc=d0[i], scale=probe_sample_var)
#         tn1 = truncnorm(a=-d1[i], b=size - d1[i], loc=d1[i], scale=probe_sample_var)
#         pdf0 = tn0.pdf(range(size))
#         pdf1 = tn1.pdf(range(size))
#         d0_probe = np.random.choice(range(size), p=pdf0 / pdf0.sum())
#         d1_probe = np.random.choice(range(size), p=pdf1 / pdf1.sum())
#         d0_probes.append(d0_probe)
#         d1_probes.append(d1_probe)
#         target_probe_dist = (
#             abs(d0_probe - d0[i]) * perceptual_weights[0]
#             + abs(d1_probe - d1[i]) * perceptual_weights[1]
#         )
#         prob_change.append(
#             1.0 / (1.0 + np.exp(-(decision_scale * target_probe_dist + decision_bias)))
#         )
#         img[0 : d0[i], 0 : d1[i]] = 1
#         img_probe[0:d0_probe, 0:d1_probe] = 1
#         target_images.append(img)
#         probe_images.append(img_probe)
#     # print np.vstack([d0_probes, d1_probes]).T - np.vstack([d0, d1]).T
#     # print prob_change
#     return (
#         np.array(target_images),
#         np.array(probe_images),
#         np.array(prob_change)[:, None],
#         np.vstack([d0, d1]).T,
#     )


# def NDim_float(N, dims, task_weights=None, max_val=100):
#     """Stimuli are vectors of length dims, taking on uniform random float
#     values on a specified range."""

#     if task_weights is None:
#         task_weights = [1.0] * dims
#     task_weights = np.array(task_weights) / float(max(task_weights))
#     max_dist = max_val
#     probe_sample_var = max_val / 10.0
#     decision_bias, decision_scale = calc_decision_params(
#         max_dist * dims / (probe_sample_var * 3.0)
#     )
#     targets = np.random.uniform(0, max_val, size=(N, dims))
#     probes = []
#     prob_change = []
#     target_probe_dist = []
#     for i, t in enumerate(targets):
#         probe = t
#         probe[np.random.choice(dims)] = np.random.rand() * max_val
#         tp_dist = sum([abs(probe[i] - t[i]) * task_weights[i] for i in range(len(t))])
#         target_probe_dist.append(tp_dist)
#         probes.append(probe)
#         prob_change.append(
#             1.0 / (1.0 + np.exp(-(decision_scale * tp_dist + decision_bias)))
#         )
#     probes = np.array(probes)
#     prob_change = np.array(prob_change)[:, None]
#     target_probe_dist = np.array(target_probe_dist)[:, None]
#     return targets, probes, prob_change, target_probe_dist


# def simple_2Dfloat(N, task_weights=[1.0, 1.0], max_val=100):
#     """Stimuli are vectors of length 2, taking on uniform random float values
#     on a specified range.
#     """
#     task_weights = np.array(task_weights) / float(max(task_weights))
#     max_dist = max_val
#     probe_sample_var = max_val / 10.0
#     decision_bias, decision_scale = calc_decision_params(
#         max_dist / (probe_sample_var * 3.0)
#     )
#     targets = np.random.uniform(0, max_val, size=(N, 2))
#     probes = []
#     prob_change = []
#     for i, t in enumerate(targets):
#         tn0 = truncnorm(a=-t[0], b=max_val - t[0], loc=t[0], scale=probe_sample_var)
#         tn1 = truncnorm(a=-t[1], b=max_val - t[1], loc=t[1], scale=probe_sample_var)
#         probe = [tn0.rvs(), tn1.rvs()]
#         target_probe_dist = (
#             abs(probe[0] - t[0]) * task_weights[0]
#             + abs(probe[1] - t[1]) * task_weights[1]
#         )
#         probes.append(probe)
#         prob_change.append(
#             1.0 / (1.0 + np.exp(-(decision_scale * target_probe_dist + decision_bias)))
#         )
#     probes = np.array(probes)
#     # print np.sort(prob_change)  # DEBUG
#     # print targets - probes
#     # from ipdb import set_trace as BP; BP()
#     return targets, probes, np.array(prob_change)[:, None]


# def letter(N, size, task_weights=[1.0, 1.0, 1.0, 1.0], min_font=4, max_font=20):
#     """Stimuli are letters/a letter of the alphabet, varying in font size and
#     position on the canvas.

#     task_weights: specify perceptual weights for all four stimulus dimensions:
#         -(x,y) position
#         -size
#         -font case
#     """
#     task_weights = np.array(task_weights) / float(max(task_weights))
#     min_pos = -int(size * 0.0)
#     max_pos = int(size * 0.7)
#     decision_bias, decision_scale = calc_decision_params(size / 5.0)
#     probe_pos_sample_var = (max_pos - min_pos) / 7.0  # Tuned by inspection
#     probe_size_sample_var = (max_font - min_font) / 7.0  # Tuned by inspection
#     target_sizes = np.random.choice(range(min_font, max_font + 1), N)
#     target_positions = np.random.choice(range(min_pos, max_pos + 1), size=(N, 2))
#     uppercase_target = np.random.choice([True, False], size=N)
#     uppercase_probe = np.random.choice([True, False], size=N)
#     probe_sizes = []
#     probe_positions = []
#     target_images = []
#     probe_images = []
#     prob_change = []
#     for i in range(N):
#         # Make target image
#         lttr_target = "A" if uppercase_target[i] else "a"
#         img_t = Image.new("L", (size, size), color=0)
#         draw_t = ImageDraw.Draw(img_t)
#         font_t = ImageFont.truetype("fonts/Arial.ttf", target_sizes[i])
#         draw_t.text(target_positions[i], lttr_target, font=font_t, fill=255)
#         target_images.append(np.array(img_t))
#         # Randomly draw probe dimensions
#         lttr_probe = "A" if uppercase_probe[i] else "a"
#         tn_posx = truncnorm(
#             a=min_pos - target_positions[i][0],
#             b=max_pos - target_positions[i][0],
#             loc=target_positions[i][0],
#             scale=probe_pos_sample_var,
#         )
#         tn_posy = truncnorm(
#             a=min_pos - target_positions[i][1],
#             b=max_pos - target_positions[i][1],
#             loc=target_positions[i][1],
#             scale=probe_pos_sample_var,
#         )
#         tn_size = truncnorm(
#             a=min_font - target_sizes[i],
#             b=max_font - target_sizes[i],
#             loc=target_sizes[i],
#             scale=probe_size_sample_var,
#         )
#         pdf_x = tn_posx.pdf(range(size))  # TODO: CHECK THIS
#         pdf_y = tn_posy.pdf(range(size))
#         pdf_size = tn_size.pdf(range(size))
#         posx_probe = np.random.choice(range(size), p=pdf_x / pdf_x.sum())
#         posy_probe = np.random.choice(range(size), p=pdf_y / pdf_y.sum())
#         size_probe = np.random.choice(range(size), p=pdf_size / pdf_size.sum())
#         probe_positions.append([posx_probe, posy_probe])
#         probe_sizes.append(size_probe)
#         target_probe_dist = (
#             (lttr_target != lttr_probe) * task_weights[0]
#             + abs(posx_probe - target_positions[i][0]) * task_weights[1]
#             + abs(posy_probe - target_positions[i][1]) * task_weights[2]
#             + abs(size_probe - target_sizes[i]) * task_weights[3]
#         )
#         prob_change.append(
#             1.0 / (1.0 + np.exp(-(decision_scale * target_probe_dist + decision_bias)))
#         )
#         # Make probe image
#         img_p = Image.new("L", (size, size), color=0)
#         draw_p = ImageDraw.Draw(img_p)
#         font_p = ImageFont.truetype("fonts/Arial.ttf", probe_sizes[i])
#         draw_p.text(probe_positions[i], lttr_probe, font=font_p, fill=255)
#         probe_images.append(np.array(img_p))
#         # img_t.show()
#         # img_p.show()
#     target_images = np.array(target_images) / 255.0
#     probe_images = np.array(probe_images) / 255.0
#     prob_change = np.array(prob_change)[:, None]
#     # print prob_change
#     # from ipdb import set_trace as BP; BP()
#     return target_images, probe_images, prob_change


# def object_array(N, size, setsize_rng=[1, 6]):
#     """Array of gray-scale squares of random shades. Set size varies according
#     to setsize_rng.
#     """
#     decision_bias, decision_scale = calc_decision_params(1.0)
#     targets = []
#     probes = []
#     prob_change = []
#     target_probe_dist = []
#     for i in range(N):
#         t = np.zeros((size, size))
#         p = np.zeros((size, size))
#         setsize = np.random.choice(np.arange(min(setsize_rng), max(setsize_rng) + 1))
#         xpos = []
#         ypos = []
#         while len(xpos) < setsize:
#             x, y = np.random.choice(range(size), size=2)
#             if x not in xpos and y not in ypos:
#                 t[x, y] = np.random.rand()
#                 p[x, y] = t[x, y]
#                 xpos.append(x)
#                 ypos.append(y)
#         if np.random.choice(2):
#             # 'Change' trial
#             i_change = np.random.choice(setsize)  # Which item changes
#             p[xpos[i_change], ypos[i_change]] = np.random.rand()
#         tp_dist = np.sum(np.abs(t - p))
#         target_probe_dist.append(tp_dist)
#         prob_change.append(
#             1.0 / (1.0 + np.exp(-(decision_scale * tp_dist + decision_bias)))
#         )
#         probes.append(p)
#         targets.append(t)
#     prob_change = np.array(prob_change)[:, None]
#     target_probe_dist = np.array(target_probe_dist)[:, None]
#     targets = np.array(targets)
#     probes = np.array(probes)
#     return targets, probes, prob_change, target_probe_dist


def gabor(size, omega, theta, func=np.cos, K=np.pi):
    """Borrowed from the internet:
    http://vision.psych.umn.edu/users/kersten/kersten-lab/courses/
        Psy5036W2017/Lectures/17_PythonForVision/Demos/html/2b.Gabor.html
    """
    radius = (int(size[0] / 2.0), int(size[1] / 2.0))
    [x, y] = np.meshgrid(
        range(-radius[0], radius[0] + 1), range(-radius[1], radius[1] + 1)
    )

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)

    gauss = (
        omega ** 2
        / (4 * np.pi * K ** 2)
        * np.exp(-omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    )
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    gab = gauss * sinusoid
    return gab


def gabor_array(N, size, setsize_rng=[1, 6], output_dim=None):
    from PIL import Image
    from itertools import product

    cycles_per_image = 120.0
    gabor_coords = [(size / 3 * i, size / 3 * j) for j in range(3) for i in range(2)]
    if output_dim is None:
        output_dim = max(setsize_rng)
    targets = []
    probes = []  # TODO
    prob_change = []  # TODO
    orientations = []
    for i in range(N):
        target = Image.new("L", (size, size))
        pixmap = target.load()
        setsize = np.random.choice(range(setsize_rng[0], setsize_rng[1] + 1))
        thetas = (
            np.zeros(output_dim) - 1
        )  # If setsize smaller than max, set orientation value to -1
        for j in range(setsize):
            theta = np.random.rand() * np.pi
            thetas[j] = theta
            gab = gabor([int(size / 3) - 1] * 2, cycles_per_image / size, theta)
            gab = (255 * gab).astype(int)
            for k, m in product(range(len(gab)), range(len(gab))):
                pixmap[gabor_coords[j][0] + k, gabor_coords[j][1] + m] = int(gab[k, m])
        targets.append(np.array(target))
        orientations.append(np.array(thetas))
    targets = np.array(targets) / 255.
    orientations = np.array(orientations)
    probes = np.zeros_like(targets)
    prob_change = np.zeros((N, 1))
    return targets, probes, prob_change, orientations


def gabor_array1(N, size):
    """Gabor array with set-size 1
    """
    return gabor_array(N, size, setsize_rng=[1, 1])


def gabor_array2(N, size):
    """Gabor array with set-size 2
    """
    return gabor_array(N, size, setsize_rng=[2, 2])


def gabor_array3(N, size):
    """Gabor array with set-size 3
    """
    return gabor_array(N, size, setsize_rng=[3, 3])


def gabor_array4(N, size):
    """Gabor array with set-size 4
    """
    return gabor_array(N, size, setsize_rng=[4, 4])


def gabor_array5(N, size):
    """Gabor array with set-size 5
    """
    return gabor_array(N, size, setsize_rng=[5, 5])


def gabor_array6(N, size):
    """Gabor array with set-size 6
    """
    return gabor_array(N, size, setsize_rng=[6, 6])


def load_plants(size, data_dir=".", layer_type="conv", normalize=True):
    """Load plant images and convert to arrays.
    """
    # leaf width, leaf droop values corresponding to loaded images
    stim_vals = list(product(range(100), range(100)))
    # Load all images
    images = [
        np.array(
            Image.open(
                Path(data_dir).joinpath(
                    "plant_stimuli",
                    "2D_{}".format(size),
                    "{}_{}.png".format(str(i).zfill(2), str(j).zfill(2)),
                )
            )
        )
        for i, j in stim_vals
    ]
    if layer_type == "conv":
        images = np.array(images)[..., None]
        stim_vals = np.array(stim_vals)[..., None]
    elif layer_type == "MLP":
        images = np.array(images).reshape(len(images), -1)
        stim_vals = np.array(stim_vals).reshape(len(images), -1)
    else:
        raise Exception
    if normalize:
        images = images / 255.
    return images, stim_vals


def plant_batch(
    N,
    images,
    stim_vals,
    task_weights=[1.0, 1.0],
    targets_only=False,
    dim=-1,
    mean=50,
    std=5,
    sample_var_factors=[1.0, 1.0],
    logistic_decision=True,
):
    """Make plants training batch of specified size by combining targets with
    random probes. 'images' is full dataset, preloaded. 'mean', and 'std'
    specify a normal distribution to draw target stimuli from, and 'dim'
    specifies along which dimension to draw from. The other dimension remains
    uniform random. If dim=-1, both dimensions are drawn uniformly.
    """
    # from PIL import Image
    images = np.array(images)
    stim_vals = np.array(stim_vals)
    task_weights = np.array(task_weights) / float(max(task_weights))
    width_min = stim_vals[:, 0].min()
    width_max = stim_vals[:, 0].max()
    droop_min = stim_vals[:, 1].min()
    droop_max = stim_vals[:, 1].max()
    probe_width_sample_var = (width_max - width_min) * sample_var_factors[0]
    probe_droop_sample_var = (droop_max - droop_min) * sample_var_factors[1]
    decision_bias, decision_scale = calc_decision_params((width_max - width_min) / 2.0)
    if dim == -1:
        inds = np.random.choice(range(len(images)), size=N)
    elif dim == 0:
        pdf = norm.pdf(range(100), loc=mean, scale=std)
        pdf /= pdf.sum()
        wvals = np.random.choice(100, size=N, p=pdf)
        dvals = np.random.choice(100, size=N)
        inds = np.array(
            [
                np.where(
                    np.logical_and(i_w == stim_vals[:, 0], i_d == stim_vals[:, 1])
                )[0]
                for i_w, i_d in zip(wvals, dvals)
            ]
        )
        inds = inds.reshape(-1)
    elif dim == 1:
        pdf = norm.pdf(range(100), loc=mean, scale=std)
        pdf /= pdf.sum()
        wvals = np.random.choice(100, size=N)
        dvals = np.random.choice(100, size=N, p=pdf)
        inds = np.array(
            [
                np.where(
                    np.logical_and(i_w == stim_vals[:, 0], i_d == stim_vals[:, 1])
                )[0]
                for i_w, i_d in zip(wvals, dvals)
            ]
        )
        inds = inds.reshape(-1)
    targets = images[inds]
    recall_target = stim_vals[inds]
    if targets_only:
        return targets
    t_vals = stim_vals[inds]
    probes = []
    prob_change = []
    target_probe_dist = []
    # Pick probes based on targets we already picked
    for t, tval in zip(targets, t_vals):
        tn_width = truncnorm(
            a=stim_vals.min() - tval[0],
            b=stim_vals.max() - tval[0],
            loc=tval[0],
            scale=probe_width_sample_var,
        )
        tn_droop = truncnorm(
            a=stim_vals.min() - tval[1],
            b=stim_vals.max() - tval[1],
            loc=tval[1],
            scale=probe_droop_sample_var,
        )
        pdf_w = tn_width.pdf(range(width_min, width_max + 1))
        pdf_d = tn_droop.pdf(range(droop_min, droop_max + 1))
        width_probe = np.random.choice(
            range(width_min, width_max + 1), p=pdf_w / pdf_w.sum()
        )
        droop_probe = np.random.choice(
            range(droop_min, droop_max + 1), p=pdf_d / pdf_d.sum()
        )
        probes.append(
            images[
                np.logical_and(
                    stim_vals[:, 0] == width_probe, stim_vals[:, 1] == droop_probe
                ).squeeze()
            ][0]
        )
        tp_dist = np.sum(np.abs((width_probe, droop_probe) - tval) * task_weights)
        target_probe_dist.append(tp_dist)
        prob_change.append(
            1.0 / (1.0 + np.exp(-(decision_scale * tp_dist + decision_bias)))
        )
    # Normalize images to between 0 and 1
    targets = np.array(targets)
    probes = np.array(probes)
    prob_change = np.array(prob_change)[:, None]
    target_probe_dist = np.array(target_probe_dist)[:, None]
    if logistic_decision:
        # Get change probability, as parameterized by a logistic sigmoid
        change_var = prob_change
    else:
        # Use target-probe distance in stimulus space, instead
        change_var = target_probe_dist
    # print prob_change
    return targets, probes, change_var, recall_target


def plant_batch_categorical(N, images, stim_vals, categ_dim=0):
    """Make plants training batch of specified size by combining targets with
    random probes. 'images' is full dataset, preloaded. Training targets
    for decision module are categorical---i.e., it is trying to predict
    whether the target and probe come from the same or different categories
    (a binary response, 1=different or 0=same).

    'categ_dim' specifies along which of the two plant dimensions (leaf-width,
    leaf-angle) to put the category boundary.
    """
    # from PIL import Image
    if categ_dim not in [0, 1]:
        raise Exception(
            "Category dimension must be either 0 or 1 for "
            "plant_batch_categorical")
    images = np.array(images)
    stim_vals = np.array(stim_vals)
    inds_t = np.random.choice(range(len(images)), size=N)  # Targets
    inds_p = np.random.choice(range(len(images)), size=N)  # Probes
    targets = images[inds_t]
    probes = images[inds_p]
    prob_change = []
    # Pick probes randomly (about half the time, target and probe will fall
    # into different categories)
    for t, p in zip(inds_t, inds_p):
        categ_t = int(stim_vals[t][categ_dim] <= 50)
        categ_p = int(stim_vals[p][categ_dim] <= 50)
        prob_change.append(1 if categ_t != categ_p else 0)
    targets = np.array(targets)
    probes = np.array(probes)
    prob_change = np.array(prob_change)[:, None]
    return targets, probes, prob_change, prob_change


def plant_batch_modal_prior(N, images, stim_vals, dim=0, means=[25, 75], std=5):
    """Make plants training batch of specified size by combining targets with
    random probes. 'images' is full dataset, preloaded. 'stim_vals' are tuples
    corresponding the width and angle of each stimulus. 'dim' specifies along
    which of the two plant dimensions (leaf-width, leaf-angle) to put the
    prior (other dim will be uniform).

    Prior along chosen dimension is bimodal gaussian.
    """
    # from PIL import Image
    images = np.array(images)
    stim_vals = np.array(stim_vals)
    pdf = norm.pdf(range(100), loc=means[0], scale=std)
    pdf += norm.pdf(range(100), loc=means[1], scale=std)
    pdf /= pdf.sum()
    vals_gauss = np.random.choice(100, size=N, p=pdf)
    vals_unif = np.random.choice(100, size=N, p=pdf)
    # vals_unif = np.random.choice(100, size=N)
    if dim == 0:
        wvals = vals_gauss  # leaf-width
        avals = vals_unif  # leaf-angle
    elif dim == 1:
        wvals = vals_unif
        avals = vals_gauss
    else:
        raise Exception("Dimension for modal plants prior must be 0 or 1")
    inds_t = np.array(
        [
            np.where(np.logical_and(i_w == stim_vals[:, 0], i_a == stim_vals[:, 1]))[0]
            for i_w, i_a in zip(wvals, avals)
        ]
    )
    targets = images[inds_t]
    targets = np.array(targets)
    probes = np.zeros_like(targets)
    prob_change = np.zeros((N, 1))
    return targets, probes, prob_change, prob_change


def plant_batch_modal_prior_1D(N, images, stim_vals, dim=0, means=[0, 100],
                               std=15):
    """Make plants training batch of specified size by combining targets with
    random probes. 'images' is full dataset, preloaded. 'stim_vals' are tuples
    corresponding the width and angle of each stimulus. 'dim' specifies along
    which of the two plant dimensions (leaf-width, leaf-angle) to put the
    prior (other dim will be uniform).

    Prior along chosen dimension is bimodal gaussian.
    """
    # from PIL import Image
    images = np.array(images)
    stim_vals = np.array(stim_vals)
    pdf = norm.pdf(range(100), loc=means[0], scale=std)
    pdf += norm.pdf(range(100), loc=means[1], scale=std)
    pdf /= pdf.sum()
    vals_gauss = np.random.choice(100, size=N, p=pdf)
    vals_unif = np.ones(N) * 50  # Irrelevant dimension always same value
    if dim == 0:
        wvals = vals_gauss  # leaf-width
        avals = vals_unif  # leaf-angle
    elif dim == 1:
        wvals = vals_unif
        avals = vals_gauss
    else:
        raise Exception("Dimension for modal plants prior must be 0 or 1")
    inds_t = np.array(
        [
            np.where(np.logical_and(i_w == stim_vals[:, 0], i_a == stim_vals[:, 1]))[0]
            for i_w, i_a in zip(wvals, avals)
        ]
    )
    targets = images[inds_t]
    # t_vals = stim_vals[inds_t]
    prob_change = []
    targets = np.array(targets)
    probes = np.zeros_like(targets)
    prob_change = np.zeros((N, 1))
    return targets, probes, prob_change, prob_change


def plant_array(N, size, images, stim_vals, setsize_rng=[1, 6],
                output_dim=None, stim_dim=0):
    """Form images that are composed of multiple plants for set-size task.
    `stim_dim` is which stimulus dimension to vary (leaf width or leaf angle)
    """
    from PIL import Image
    from itertools import product

    stim_vals = stim_vals.reshape(-1, 2)

    coords = [
        (size * i, size * j) for j in range(3) for i in range(2)]
    if output_dim is None:
        output_dim = max(setsize_rng)
    targets = []
    decision_targets = []
    for i in range(N):
        target = Image.new("L", (size * 3, size * 3))
        pixmap = target.load()
        setsize = np.random.choice(range(setsize_rng[0], setsize_rng[1] + 1))
        stimvals = (
            np.zeros(output_dim) - 1
        )  # If setsize smaller than max, set orientation value to -1
        for j in range(setsize):
            stimval = np.random.choice(100)  # 100 possible stim vals each dim
            stimvals[j] = stimval
            ind_bool = np.logical_and(
                stim_vals[:, stim_dim] == stimval,
                stim_vals[:, int(not stim_dim)] == 50).reshape(-1)
            plant = images[ind_bool].squeeze()
            plant = (255 * plant).astype(int)
            for k, m in product(range(len(plant)), range(len(plant))):
                pixmap[coords[j][0] + k, coords[j][1] + m] = int(plant[m, k])
        targets.append(np.array(target))
        decision_targets.append(np.array(stimvals))
    targets = np.array(targets) / 255.
    decision_targets = np.array(decision_targets)
    probes = np.zeros_like(targets)
    prob_change = np.zeros((N, 1))
    return targets, probes, prob_change, decision_targets


def load_places365_batch(N, train_test, dataset_dir, models):
    from keras.applications.vgg19 import preprocess_input
    if train_test == "train":
        category_dirs = Path(dataset_dir).joinpath("places365_standard", "train").dirs()
    elif train_test == "test":
        category_dirs = Path(dataset_dir).joinpath("places365_standard", "val").dirs()
    else:
        raise Exception("Invalid value '{}' for train_test".format(train_test))
    # Data labels
    labels_dict = {
        name.splitall()[-1]: i for i, name in enumerate(category_dirs)
    }  # Same categories for both train and test
    images = []
    features = []
    labels = []
    cat_inds = np.random.choice(range(365), size=N)
    categories_used = list(set(cat_inds.tolist()))
    for k, cat_ind in enumerate(categories_used):
        print("Extracting features for category {}/{}...".format(
              k + 1, len(categories_used)))
        img_files = category_dirs[cat_ind].files("*.jpg")
        n = (cat_inds == cat_ind).sum()  # How many times this category was drawn
        img_inds = np.random.choice(range(len(img_files)), size=n)
        # Choose each training example from a random category
        imgs = [open_rgb(img_files[i]) for i in img_inds]
        x = np.array([np.array(f) for f in imgs])
        y = np.hstack(
            [
                model.predict(preprocess_input(x), batch_size=16).reshape(
                    len(x), -1
                )
                for model in models
            ]
        )
        lab = np.array([labels_dict[img_files[i].splitall()[-2]] for i in img_inds])
        # Downsample images
        x1 = [np.array(img.resize((128, 128), Image.ANTIALIAS)) for img in imgs]
        images.append(x1)
        features.append(y)
        labels.append(lab)
    images = np.concatenate(images)
    features = np.concatenate(features)
    features = (features - features.mean()) / features.std()
    labels = np.concatenate(labels)
    return features, images / 255.0, labels


def load_fruits360(N, train_test, dataset_dir):
    """Load fruits-360 dataset into numpy arrays, and create corresponding
    labels
    """
    classes = ["Apple", "Banana", "Tomato"]
    labels_dict = dict(zip(classes, range(len(classes))))
    if train_test == "train":
        category_dirs = Path(dataset_dir).joinpath("fruits-360", "Training").dirs()
    elif train_test == "test":
        category_dirs = Path(dataset_dir).joinpath("fruits-360", "Test").dirs()
    else:
        raise Exception("Invalid value '{}' for train_test".format(train_test))
    category_dirs = [  # Filter for just the classes we've specified above
        d for d in category_dirs if any([c in d for c in classes])
    ]
    labels = []
    images = []
    for d in category_dirs:
        # print("Loading " + d)
        imgs = [np.array(open_rgb(f, pad=12)) for f in d.files("*.jpg")]
        images.extend(imgs)
        for c in classes:
            if c in d:
                labels.extend([labels_dict[c]] * len(imgs))
    images = np.array(images) / 255.
    labels = np.array(labels)
    if N > 0:
        # Select subset
        inds_sampled = np.random.choice(len(images), size=N)
        return images[inds_sampled], labels[inds_sampled]
    else:
        # Select all
        return images, labels


def get_fruit360_filenames_and_labels(train_test, dataset_dir):
    """Get list of file names and corresponding class labels for fruits-360
    dataset. Returned as tensorflow constants.
    """
    # import tensorflow as tf
    classes = ["Apple", "Banana", "Tomato"]
    labels_dict = dict(zip(classes, range(len(classes))))
    if train_test == "train":
        category_dirs = Path(dataset_dir).joinpath("fruits-360", "Training").dirs()
    elif train_test == "test":
        category_dirs = Path(dataset_dir).joinpath("fruits-360", "Test").dirs()
    else:
        raise Exception("Invalid value '{}' for train_test".format(train_test))
    category_dirs = [  # Filter for just the classes we've specified above
        d for d in category_dirs if any([c in d for c in classes])
    ]
    labels = []
    image_pths = []
    for d in category_dirs:
        image_pths.extend(d.files("*.jpg"))
        for c in classes:
            if c in d:
                labels.extend([labels_dict[c]] * len(d.files("*.jpg")))
    # return tf.constant(image_pths), tf.constant(labels)
    return image_pths, labels


# def fruits360_to_tfrecords(dataset_dir, train_test):
#     images, labels = load_fruits360(-1, train_test, dataset_dir)
#     print("Dataset loaded. Making records...")
#     make_tfrecords(images, labels, "fruits360_" + train_test,
#                    "/storage/cjbates/tensorflow/tfrecords/")
#     return


# def make_tfrecords(images, labels, name, save_dir):
#     """Save images and labels as TFRecords format.

#     Borrowed from: https://github.com/tensorflow/tensorflow/blob/master/\
#         tensorflow/examples/how_tos/reading_data/convert_to_records.py
#     """
#     import tensorflow as tf

#     def _int64_feature(value):
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#     def _bytes_feature(value):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#     save_pth = Path(save_dir).joinpath(name + ".tfrecords")
#     rows = images.shape[1]
#     cols = images.shape[2]
#     depth = images.shape[3]
#     print('Writing', save_pth)
#     with tf.python_io.TFRecordWriter(save_pth) as writer:
#         for index in range(len(images)):
#             image_raw = images[index].tostring()
#             example = tf.train.Example(
#                 features=tf.train.Features(
#                     feature={
#                         'height': _int64_feature(rows),
#                         'width': _int64_feature(cols),
#                         'depth': _int64_feature(depth),
#                         'label': _int64_feature(int(labels[index])),
#                         'image_raw': _bytes_feature(image_raw)
#                     }))
#             writer.write(example.SerializeToString())
#     return


# def load_cifar10(train_test, dataset_dir):
#     img_pth = Path(dataset_dir).joinpath("data_{}.npy".format(train_test))
#     print("Using {} as training data.".format(img_pth))
#     labels_pth = Path(dataset_dir).joinpath("labels_{}.npy".format(train_test))
#     features_pth = Path(dataset_dir).joinpath("features_{}.npy".format(train_test))
#     with img_pth.open("rb") as fid:
#         images = np.load(fid)
#     with labels_pth.open("rb") as fid:
#         labels = np.load(fid)
#     with features_pth.open("rb") as fid:
#         features = np.load(fid)
#     # Normalize features
#     features = (features - features.mean()) / features.std()
#     return features, images / 255.0, labels


# def episodic_shape_race(N, size, n_frames=6, RGB=True, debug=False):
#     """Dataset of videos that contain two shapes, a cross and a square,
#     which 'race' to pass through a rectangular 'goal' patch. Cross and square
#     each follow linear trajectories, from random starting positions around the
#     perimeter.
    
#     Training targets for decision module are:
#     1) Starting coordinates of each shape
#     2) Angle of trajectory of each shape
#     3) How many frames prior to the cross that the square passes through the
#     patch

#     Each of these targets can be weighted as desired in the training loss.

#     RGB option specifies to put each of the three objects in the scene
#     in separate color channels of the image. Otherwise, if just using a single
#     channel, 

#     TODO...
    
#     """

#     def make_square(upper_left, size, canvas_size):
#         """Given upper-left corner and size, return array indices.
#         (y axis is negative)
#         """
#         X = (
#             np.array(
#                 np.meshgrid(
#                     np.arange(upper_left[0], upper_left[0] + size + 1),
#                     np.arange(upper_left[1], upper_left[1] + size + 1),
#                 )
#             )
#             .reshape(2, -1)
#             .T
#         )
#         X = np.array([x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
#         return X

#     def make_cross(upper_left, size, canvas_size):
#         """Given upper-left corner and size, return array indices.
#         (y axis is negative)
#         """
#         ix = upper_left[0] + size / 2
#         iy = upper_left[1] + size / 2
#         vertical_bar = zip(range(upper_left[0], upper_left[0] + size + 1), [iy] * size)
#         horizontal_bar = zip(
#             [ix] * size, range(upper_left[1], upper_left[1] + size + 1)
#         )
#         X = np.array(zip(*(vertical_bar + horizontal_bar))).T
#         X = np.array([x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
#         return X

#     def test_collision(inds1, inds2):
#         """Test whether any (x, y) coordinate in inds1 occurs in inds2
#         """
#         inds1 = set(zip(*inds1))
#         inds2 = set(zip(*inds2))
#         if len(list(inds1 & inds2)) > 0:
#             return True
#         else:
#             return False

#     # Get indices of goal patch. Put a square in the center of the canvas
#     # with a quarter of the area of the entire canvas.
#     goal_size = int(size * 0.25)
#     goal_top_left = [int(size * 0.25), int(size * 0.25)]
#     goal_bottom_right = [int(size * 0.75), int(size * 0.75)]
#     i_goal = make_square(goal_top_left, goal_size, size)
#     shp_size = int(size * 0.1)
#     # One of 16 possible trajectories is chosen for each shape in each
#     # scene. Below, an index is drawn to indicate which of these has been
#     # chosen. Here, we create the array that gives how many pixels in x and y
#     # directions to step on each frame (e.g. 45 degress would be one pixel in x
#     # and one pixel in y).
#     velocities0 = [
#         (1, 0),
#         (3, 1),
#         (1, 1),
#         (1, 3),
#         (0, 1),
#         (-1, 3),
#         (-1, 1),
#         (-3, 1),
#         (-1, 0),
#         (-3, -1),
#         (-1, -1),
#         (-1, -3),
#         (0, -1),
#         (1, -3),
#         (1, -1),
#         (3, -1),
#     ]
#     f = size / 40.0
#     velocities = [(v[0] * f, v[1] * f) for v in velocities0]
#     # Randomly generate stimuli
#     targets = []
#     decision_targets = []
#     counter = 0
#     while counter < N:
#         target = []
#         shp1_top_left = [
#             np.random.choice(range(0, size - shp_size)),
#             np.random.choice(range(0, size - shp_size)),
#         ]
#         shp2_top_left = [
#             np.random.choice(range(0, size - shp_size)),
#             np.random.choice(range(0, size - shp_size)),
#         ]
#         # Pick trajectory angles
#         traj_shp1 = np.random.choice(16)
#         traj_shp2 = np.random.choice(16)
#         # Keep track of whether shapes have overlapped at some point with
#         # the goal patch. If they never do, reject this random sample
#         shp1_has_touched_goal = False
#         shp2_has_touched_goal = False
#         shp1_1st_frame_touch = None
#         shp2_1st_frame_touch = None
#         # Step through frames, moving shapes according to their designated
#         # velocities
#         for step in range(n_frames):
#             # Calculate indices that each shape occupies for this time step
#             i_shp1 = make_square(shp1_top_left, shp_size, size)
#             i_shp2 = make_square(shp2_top_left, shp_size, size)
#             target.append(np.zeros((size, size, 3)))  # Three color channels
#             if len(i_goal) > 0:
#                 target[-1][i_goal[0], i_goal[1], 0] = 1.0
#             if len(i_shp1) > 0:
#                 target[-1][i_shp1[0], i_shp1[1], 1] = 1.0
#             if len(i_shp2) > 0:
#                 target[-1][i_shp2[0], i_shp2[1], 2] = 1.0
#             # Move shapes for next frame
#             shp1_top_left[0] = int(round(shp1_top_left[0] + velocities[traj_shp1][0]))
#             shp1_top_left[1] = int(round(shp1_top_left[1] + velocities[traj_shp1][1]))
#             shp2_top_left[0] = int(round(shp2_top_left[0] + velocities[traj_shp2][0]))
#             shp2_top_left[1] = int(round(shp2_top_left[1] + velocities[traj_shp2][1]))
#             # Test for collisions
#             if not shp1_has_touched_goal:
#                 shp1_has_touched_goal = test_collision(i_shp1, i_goal)
#                 if shp1_has_touched_goal:
#                     # Record first frame that shape 1 touched the goal
#                     shp1_1st_frame_touch = step
#             if not shp2_has_touched_goal:
#                 shp2_has_touched_goal = test_collision(i_shp2, i_goal)
#                 if shp2_has_touched_goal:
#                     # Record first frame that shape 2 touched the goal
#                     shp2_1st_frame_touch = step
#         decision_target = np.array(
#             shp1_top_left
#             + shp2_top_left
#             + list(velocities[traj_shp1])
#             + list(velocities[traj_shp2])
#             + [shp1_1st_frame_touch]
#             + [shp2_1st_frame_touch]
#         )
#         if shp1_has_touched_goal and shp2_has_touched_goal:
#             targets.append(target)
#             decision_targets.append(decision_target)
#             counter += 1
#             if debug:
#                 import matplotlib.pyplot as plt

#                 # from PIL import Image
#                 for t in target:
#                     plt.imshow(t)
#                     plt.show()
#                     # im = Image.fromarray((t * 255).astype(np.uint8))
#                     # im.show()
#                 # from ipdb import set_trace as BP; BP()
#     targets = np.array(targets)
#     decision_targets = np.array(decision_targets)
#     probes = np.zeros_like(targets)  # Filler
#     prob_change = np.zeros((N, 1))  # Filler
#     return targets, probes, prob_change, decision_targets


# def make_square(upper_left, size, canvas_size):
#     """Given upper-left corner and size, return array indices.
#     (y axis is negative)
#     """
#     X = (
#         np.array(
#             np.meshgrid(
#                 np.arange(upper_left[0], upper_left[0] + size + 1),
#                 np.arange(upper_left[1], upper_left[1] + size + 1),
#             )
#         )
#         .reshape(2, -1)
#         .T
#     )
#     X = np.array([x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
#     return X


# def generate_ep_ss(args):
#     """Must be pickle-able to parallelize, so this function lives outside of
#     'episodic_setsize'
#     """

#     [
#         N,
#         size,
#         shp_size,
#         max_num_shapes,
#         setsize_rng,
#         velocities,
#         n_frames,
#         debug,
#         parallel,
#         make_square,
#     ] = args
#     np.random.seed()
#     targets = []
#     decision_targets = []
#     for n in range(N):
#         target = []
#         n_shapes = np.random.choice(range(setsize_rng[0], setsize_rng[1] + 1))
#         shape_coords0 = [
#             [
#                 np.random.choice(range(0, size - shp_size)),
#                 np.random.choice(range(0, size - shp_size)),
#             ]
#             for i in range(n_shapes)
#         ]  # Save original
#         shape_coords = copy.deepcopy(shape_coords0)
#         # Pick trajectory angles
#         traj = [np.random.choice(16) for i in range(n_shapes)]
#         # Step through frames, moving shapes according to their designated
#         # velocities
#         for step in range(n_frames):
#             # Calculate indices that each shape occupies for this time step
#             i_shapes = [make_square(c, shp_size, size) for c in shape_coords]
#             target.append(np.zeros((size, size, 3)))  # Three color channels
#             for i, i_shp in enumerate(i_shapes):
#                 if len(i_shp) > 0:
#                     # Only update if the object is still visible on the canvas
#                     target[-1][i_shp[0], i_shp[1], i] = 1.0
#             # Move shapes for next frame
#             for i in range(n_shapes):
#                 shape_coords[i][0] = int(
#                     round(shape_coords[i][0] + velocities[traj[i]][0])
#                 )
#                 shape_coords[i][1] = int(
#                     round(shape_coords[i][1] + velocities[traj[i]][1])
#                 )
#         vel_decision_target = []
#         shape_coord_decision_target = []
#         for i in range(max_num_shapes):
#             if i < n_shapes:
#                 vel_decision_target.extend(velocities[traj[i]])
#                 shape_coord_decision_target.extend(shape_coords0[i])
#             else:
#                 vel_decision_target.extend([0, 0])
#                 shape_coord_decision_target.extend([-1, -1])
#         decision_target = np.array(shape_coord_decision_target + vel_decision_target)
#         targets.append(target)
#         decision_targets.append(decision_target)
#         if debug and not parallel:
#             import matplotlib.pyplot as plt

#             for t in target:
#                 plt.imshow(t)
#                 plt.show()
#     targets = np.array(targets)
#     decision_targets = np.array(decision_targets)
#     probes = np.zeros_like(targets)  # Filler
#     prob_change = np.zeros((N, 1))  # Filler
#     return targets, probes, prob_change, decision_targets


# def episodic_setsize(
#     N, size, n_frames=6, parallel=False, setsize_rng=[1, 3], debug=False
# ):
#     """Dataset of videos that contain between 1 and 3 squares
#     (one for each color channel), which move across the domain with constant
#     linear velocity. Initial positions and directions are randomly chosen.
    
#     Training targets for decision module are:
#     1) Starting coordinates of each shape
#     2) Angle of trajectory of each shape

#     Each of these targets can be weighted as desired in the training loss.
#     """

#     # def make_square(upper_left, size, canvas_size):
#     #     """Given upper-left corner and size, return array indices.
#     #     (y axis is negative)
#     #     """
#     #     X = np.array(
#     #         np.meshgrid(np.arange(upper_left[0],
#     #                               upper_left[0] + size + 1),
#     #                     np.arange(upper_left[1],
#     #                               upper_left[1] + size + 1))
#     #                     ).reshape(2, -1).T
#     #     X = np.array([x for x in X if np.all(x >= 0) and
#     #                   np.all(x < canvas_size)]).T
#     #     return X

#     shp_size = int(size * 0.2)
#     # One of 16 possible trajectories is chosen for each shape in each
#     # scene. Below, an index is drawn to indicate which of these has been
#     # chosen. Here, we create the array that gives how many pixels in x and y
#     # directions to step on each frame (e.g. 45 degress would be one pixel in x
#     # and one pixel in y).
#     velocities0 = [
#         (1, 0),
#         (3, 1),
#         (1, 1),
#         (1, 3),
#         (0, 1),
#         (-1, 3),
#         (-1, 1),
#         (-3, 1),
#         (-1, 0),
#         (-3, -1),
#         (-1, -1),
#         (-1, -3),
#         (0, -1),
#         (1, -3),
#         (1, -1),
#         (3, -1),
#     ]
#     f = size / 40.0
#     max_num_shapes = 3
#     if setsize_rng[1] > 3:
#         raise Exception("Setsize range may not exceed 3.")
#     velocities = [(v[0] * f, v[1] * f) for v in velocities0]
#     args0 = [
#         size,
#         shp_size,
#         max_num_shapes,
#         setsize_rng,
#         velocities,
#         n_frames,
#         debug,
#         parallel,
#         make_square,
#     ]
#     # Randomly generate stimuli
#     if parallel:
#         from multiprocessing import Pool
#         import multiprocessing as mp

#         n_threads = min(mp.cpu_count(), N)
#         remainder = N % n_threads
#         N_batch = [int(np.floor(N / n_threads))] * n_threads
#         if remainder:
#             N_batch.append(remainder)
#         args = [[n] + args0 for n in N_batch]
#         with Pool(n_threads) as p:
#             X = p.map(generate_ep_ss, args)
#         # Concatenate worker outputs
#         targets = np.concatenate([x[0] for x in X], axis=0)
#         probes = np.concatenate([x[1] for x in X], axis=0)
#         prob_change = np.concatenate([x[2] for x in X], axis=0)
#         decision_targets = np.concatenate([x[3] for x in X], axis=0)
#     else:
#         args = [N] + args0
#         targets, probes, prob_change, decision_targets = generate_ep_ss(args)
#     return targets, probes, prob_change, decision_targets


# def attention_MOT(N, size, n_frames=6, debug=False):
#     """Multiple-object tracking task in which the background texture may change
#     at some point during the sequence. The decision-module targets are whether
#     the background changed or not, plus the start positions and velocities of
#     the moving objects. If errors in identifying changes to the background are
#     not penalized heavily in the decision loss, then at low rates, the network
#     should be agnostic to background. Alternatively, we can penalize errors
#     when changes to the background constitute scene-category changes, but not
#     otherwise (see reference). In this case, the network should be sensitive
#     accordingly.

#     Adapted from "Spatial ensemble statistics are efficient codes that
#     can be represented with reduced attention"
#     (http://www.pnas.org/content/pnas/early/2009/04/17/0808981106.full.pdf)
#     """
#     import copy

#     def make_square(upper_left, size, canvas_size):
#         """Given upper-left corner and size, return array indices.
#         (y axis is negative)
#         """
#         X = (
#             np.array(
#                 np.meshgrid(
#                     np.arange(upper_left[0], upper_left[0] + size + 1),
#                     np.arange(upper_left[1], upper_left[1] + size + 1),
#                 )
#             )
#             .reshape(2, -1)
#             .T
#         )
#         X = np.array([x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
#         return X

#     def make_background(X, bg_id, texton_size, size):
#         # Divide into top half and bottom half, which have distinct textures.
#         # Textons are oriented gabor patches.
#         alpha = 22.5  # degrees
#         cycles_per_texton = 4
#         cycles_per_pixel = cycles_per_texton / float(texton_size)

#         # # Maintain omega^2/8K^2 ~= 0.01 for consistent peak amplitude across
#         # # image sizes (see gabor function ref)
#         # exp_decay_rate = np.sqrt(cycles_per_pixel ** 2. / 0.01 / 8.)

#         # textons per dim:
#         n_textons = np.ceil(size / float(texton_size)).astype(int)
#         if bg_id == 0:
#             print("bg_id == 0")
#             # Top texture vertical, bottom texture horizontal
#             angles_top = [90 - alpha, 90 + alpha]
#             angles_bottom = [-alpha, alpha]
#         elif bg_id == 1:
#             print("bg_id == 1")
#             # Top texture vertical, bottom texture horizontal
#             angles_top = [90 + alpha, 90 - alpha]
#             angles_bottom = [alpha, -alpha]
#         elif bg_id == 2:
#             print("bg_id == 2")
#             # Top texture horizontal, bottom texture vertical
#             angles_top = [alpha, -alpha]
#             angles_bottom = [90 + alpha, 90 - alpha]
#         for i in range(n_textons):
#             for j in range(n_textons):
#                 if i < n_textons / 2:
#                     if j % 2:
#                         theta = angles_top[0]
#                     else:
#                         theta = angles_top[1]
#                 else:
#                     if j % 2:
#                         theta = angles_bottom[0]
#                     else:
#                         theta = angles_bottom[1]
#                 theta = theta * np.pi / 180.0
#                 top_left = (i * texton_size, j * texton_size)
#                 gab = gabor(
#                     (texton_size, texton_size), cycles_per_pixel, theta, K=np.pi * 0.15
#                 )
#                 # Need to clip pixels that spill over edge of domain
#                 texton_size_clipped_x = min(texton_size, max(0, size - top_left[0]))
#                 texton_size_clipped_y = min(texton_size, max(0, size - top_left[1]))
#                 ind_x, ind_y = np.meshgrid(
#                     range(top_left[0], top_left[0] + texton_size_clipped_x),
#                     range(top_left[1], top_left[1] + texton_size_clipped_y),
#                 )
#                 # Need to truncate if texton size is even, because gabor
#                 # function always returns odd size
#                 gab = gab[:texton_size_clipped_x, :texton_size_clipped_y]
#                 gab /= np.abs(gab).max()
#                 X[ind_x.T, ind_y.T, -1] = gab
#         return X

#     shp_size = int(size * 0.1)
#     texton_size = int(round(size / 6.0))
#     # One of 16 possible trajectories is chosen for each shape in each
#     # scene. Below, an index is drawn to indicate which of these has been
#     # chosen. Here, we create the array that gives how many pixels in x and y
#     # directions to step on each frame (e.g. 45 degress would be one pixel in x
#     # and one pixel in y).
#     velocities0 = [
#         (1, 0),
#         (3, 1),
#         (1, 1),
#         (1, 3),
#         (0, 1),
#         (-1, 3),
#         (-1, 1),
#         (-3, 1),
#         (-1, 0),
#         (-3, -1),
#         (-1, -1),
#         (-1, -3),
#         (0, -1),
#         (1, -3),
#         (1, -1),
#         (3, -1),
#     ]
#     f = size / 40.0
#     max_num_shapes = 3
#     velocities = [(v[0] * f, v[1] * f) for v in velocities0]
#     # Randomly generate stimuli
#     targets = []
#     decision_targets = []
#     bg_id0 = 0  # Always start with same background, then possibly change it
#     # part-way through, to one of two other possible backgrounds. (See figure
#     # 1 from reference above)
#     for n in range(N):
#         # Handle background changes for this example
#         bg_id = bg_id0  # This variable keeps track of changes, but always
#         #                 starts with default configuration
#         background_change_step = np.random.choice(2 * n_frames - 1) + 1
#         print("background change step: " + str(background_change_step))
#         bg_id1 = np.random.choice([1, 2])  # Choose "Ensemble same" or
#         #                                    "Ensemble different"

#         target = []
#         n_shapes = 2
#         shape_coords0 = [
#             [np.random.choice(size - shp_size), np.random.choice(size - shp_size)]
#             for i in range(n_shapes)
#         ]
#         shape_coords = copy.deepcopy(shape_coords0)  # Save original
#         # Pick trajectory angles
#         traj = [np.random.choice(16) for i in range(n_shapes)]
#         # Step through frames, moving shapes according to their designated
#         # velocities
#         for step in range(n_frames):
#             # Change background
#             if step == background_change_step:
#                 bg_id = bg_id1
#             # Calculate indices that each shape occupies for this time step
#             i_shapes = [make_square(c, shp_size, size) for c in shape_coords]
#             target.append(np.zeros((size, size, 3)))  # Three color channels
#             target[-1] = make_background(target[-1], bg_id, texton_size, size)
#             for i, i_shp in enumerate(i_shapes):
#                 if len(i_shp) > 0:
#                     # Only update if the object is still visible on the canvas
#                     target[-1][i_shp[0], i_shp[1], i] = 1.0
#             # Move shapes for next frame
#             for i in range(n_shapes):
#                 shape_coords[i][0] = int(
#                     round(shape_coords[i][0] + velocities[traj[i]][0])
#                 )
#                 shape_coords[i][1] = int(
#                     round(shape_coords[i][1] + velocities[traj[i]][1])
#                 )
#         vel_decision_target = []
#         shape_coord_decision_target = []
#         for i in range(max_num_shapes):
#             if i < n_shapes:
#                 vel_decision_target.extend(velocities[traj[i]])
#                 shape_coord_decision_target.extend(shape_coords0[i])
#             else:
#                 vel_decision_target.extend([0, 0])
#                 shape_coord_decision_target.extend([-1, -1])
#         if background_change_step > n_frames - 1:
#             bg_change_decision_target = [-1]
#         else:
#             bg_change_decision_target = [background_change_step]
#         decision_target = np.array(
#             shape_coord_decision_target
#             + vel_decision_target
#             + bg_change_decision_target
#         )
#         targets.append(target)
#         decision_targets.append(decision_target)
#         if debug:
#             import matplotlib.pyplot as plt

#             # from PIL import Image
#             for t in target:
#                 plt.imshow(t)
#                 plt.show()
#     targets = np.array(targets)
#     decision_targets = np.array(decision_targets)
#     probes = np.zeros_like(targets)  # Filler
#     prob_change = np.zeros((N, 1))  # Filler
#     return targets, probes, prob_change, decision_targets


def attention_search_oddball(N, size, grid_count=4, display_type=None, debug=False):
    """Input is array of objects, similar to classic search
    time experiments (e.g. Treisman). Decision target is location of target
    object. Idea is to show that popout happens when display is easily
    compressed.

    'Oddball' means that the subject does not know ahead of time which feature they
    are looking for, just that it should be unique from distractors.

    Images are in color, and are composed of squares and crosses.
    Target on each trial can either be a solitary cross or solitary square.

    'display_type' is option to specify that 'targ_distr_differ_along' is
    always just one value, rather than chosen randomly among options.
    """

    def make_square(upper_left, size, canvas_size):
        """Given upper-left corner and size, return array indices.
        (y axis is negative)
        """
        X = (
            np.array(
                np.meshgrid(
                    np.arange(upper_left[0], upper_left[0] + size + 1),
                    np.arange(upper_left[1], upper_left[1] + size + 1),
                )
            )
            .reshape(2, -1)
            .T
        )
        X = np.array([x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
        return X

    def make_cross(upper_left, size, canvas_size):
        """Given upper-left corner and size, return array indices.
        (y axis is negative)
        """
        ix = upper_left[0] + int(round(size / 2))
        iy = upper_left[1] + int(round(size / 2))
        vertical_bar = list(
            zip(range(upper_left[0], upper_left[0] + size + 1), [iy] * size)
        )
        horizontal_bar = list(
            zip([ix] * size, range(upper_left[1], upper_left[1] + size + 1))
        )
        X = np.array(list(zip(*(vertical_bar + horizontal_bar)))).T
        X = np.array([x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
        return X

    def make_shape(x, coord, color, shape, grid_size):
        shape_size = int(round(grid_size * 0.5))
        if shape == 0:
            inds = make_square(coord, shape_size, size)
        else:
            inds = make_cross(coord, shape_size, size)
        x[inds[0], inds[1], color] = 1.0
        return x

    def make_display(colors, shapes, target_coord, grid_count):
        grid_size = int(round(size / grid_count))
        x = np.zeros((size, size, 3))
        for i in range(grid_count):
            for j in range(grid_count):
                x = make_shape(
                    x,
                    (i * grid_size, j * grid_size),
                    colors[i, j],
                    shapes[i, j],
                    grid_size,
                )
        return x

    X = []
    decision_targets = []
    combos = list(product(range(2), range(3)))  # All possible objects
    #                                             (num colors X num shapes)
    for n in range(N):
        # Choose target location
        target_coord = (
            np.random.choice(range(grid_count)),
            np.random.choice(range(grid_count)),
        )
        # Choose dimensions along which target and distractor differ
        if display_type is not None:
            targ_distr_differ_along = display_type
        else:
            targ_distr_differ_along = np.random.choice(
                ["both_two_colors"]
                # ["shape", "color", "both_two_colors"]
                # ["shape", "color", "both", "both_two_colors"]
            )
        colors = np.zeros((grid_count, grid_count)).astype(int)
        shapes = np.zeros((grid_count, grid_count))
        if targ_distr_differ_along == "shape":
            target_shape, distractor_shape = [[0, 1], [1, 0]][np.random.choice(2)]
            # Choose single color for all objects (either R, G, or B)
            colors += np.random.choice(3)
            shapes += distractor_shape
            shapes[target_coord] = target_shape
        elif targ_distr_differ_along == "color":
            # Choose one color for target and different one for distractors
            target_color = np.random.choice(3)
            shapes += np.random.choice(2)
            distractor_color = np.random.choice(
                np.arange(3)[np.arange(3) != target_color]
            )
            colors[target_coord] = target_color
            for i in range(grid_count):
                for j in range(grid_count):
                    if (i, j) == target_coord:
                        colors[i, j] = target_color
                    else:
                        colors[i, j] = distractor_color
        elif targ_distr_differ_along == "both":
            # Choose color and shape for target, then let distractors be any
            # combination other than that, but also make sure there are no
            # other singletons among distractors which could be considered
            # targets, too.
            target_color = np.random.choice(3)
            target_shape = np.random.choice(2)
            # Make new list with the target object values removed
            # as an option
            combos_allowed = [
                c for c in combos if c[0] != target_shape or c[1] != target_color
            ]
            while True:
                conjunction_counts = {(i, j): 0 for i, j in combos}
                for i in range(grid_count):
                    for j in range(grid_count):
                        if (i, j) == target_coord:
                            colors[i, j] = target_color
                            shapes[i, j] = target_shape
                        else:
                            # Pick uniformly from list of allowed distractors.
                            # This ensures that the distractor object is never the
                            # same as the target
                            shapes[i, j], colors[i, j] = combos_allowed[
                                np.random.choice(len(combos_allowed))
                            ]
                            conjunction_counts[(shapes[i, j], colors[i, j])] += 1
                if all([v != 1 for v in conjunction_counts.values()]):
                    # Illegal to have any distractors be singletons. There must
                    # be at least one other like it. Passed this check.
                    break
        elif targ_distr_differ_along == "both_two_colors":
            # Same as 'both', but with only two colors, not three
            omitted_color = np.random.choice(3)
            colors_reduced = np.arange(3)[np.arange(3) != omitted_color]
            target_color = np.random.choice(colors_reduced)
            target_shape = np.random.choice(2)
            # Make new list with the target object values removed
            # as an option
            combos_allowed = [
                c
                for c in combos
                if (c[0] != target_shape or c[1] != target_color)
                and c[1] != omitted_color
            ]
            while True:
                conjunction_counts = {(i, j): 0 for i, j in combos}
                for i in range(grid_count):
                    for j in range(grid_count):
                        if (i, j) == target_coord:
                            colors[i, j] = target_color
                            shapes[i, j] = target_shape
                        else:
                            # Pick uniformly from list of allowed distractors.
                            # This ensures that the distractor object is never the
                            # same as the target
                            shapes[i, j], colors[i, j] = combos_allowed[
                                np.random.choice(len(combos_allowed))
                            ]
                            conjunction_counts[(shapes[i, j], colors[i, j])] += 1
                if all([v != 1 for v in conjunction_counts.values()]):
                    # Illegal to have any distractors be singletons. There must
                    # be at least one other like it. Passed this check.
                    break
        x = make_display(colors, shapes, target_coord, grid_count)
        X.append(x)
        decision_targets.append(target_coord)
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(x)
            plt.show()
    X = np.array(X)
    decision_targets = np.array(decision_targets)
    probes = np.zeros_like(X)  # Filler
    prob_change = np.zeros((N, 1))  # Filler
    return X, probes, prob_change, decision_targets


def attention_search(N, size, grid_count=4, display_type=None, debug=False):
    """Input is array of objects, similar to classic search
    time experiments (e.g. Treisman). Decision target is location of target
    object. Idea is to show that popout happens when display is easily
    compressed.

    Images are in color, and are composed of squares and crosses.
    Target on each trial is a red square.

    'display_type' is option to specify that 'targ_distr_differ_along' is
    always just one value, rather than chosen randomly among options.
    """

    def make_square(upper_left, size, canvas_size):
        """Given upper-left corner and size, return array indices.
        (y axis is negative)
        """
        X = (
            np.array(
                np.meshgrid(
                    np.arange(upper_left[0], upper_left[0] + size + 1),
                    np.arange(upper_left[1], upper_left[1] + size + 1),
                )
            )
            .reshape(2, -1)
            .T
        )
        X = np.array([x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
        return X

    def make_cross(upper_left, size, canvas_size):
        """Given upper-left corner and size, return array indices.
        (y axis is negative)
        """
        ix = upper_left[0] + int(round(size / 2))
        iy = upper_left[1] + int(round(size / 2))
        vertical_bar = list(
            zip(range(upper_left[0], upper_left[0] + size + 1), [iy] * size)
        )
        horizontal_bar = list(
            zip([ix] * size, range(upper_left[1], upper_left[1] + size + 1))
        )
        X = np.array(list(zip(*(vertical_bar + horizontal_bar)))).T
        X = np.array([x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
        return X

    def make_shape(x, coord, color, shape, grid_size):
        shape_size = int(round(grid_size * 0.5))
        if shape == 0:
            inds = make_square(coord, shape_size, size)
        else:
            inds = make_cross(coord, shape_size, size)
        x[inds[0], inds[1], color] = 1.0
        return x

    def make_display(colors, shapes, target_coord, grid_count):
        grid_size = int(round(size / grid_count))
        x = np.zeros((size, size, 3))
        for i in range(grid_count):
            for j in range(grid_count):
                x = make_shape(
                    x,
                    (i * grid_size, j * grid_size),
                    colors[i, j],
                    shapes[i, j],
                    grid_size,
                )
        return x

    X = []
    decision_targets = []
    combos = list(product(range(2), range(3)))  # All possible objects
    #                                             (num colors X num shapes)
    if display_type is not None:
        print("Using " + display_type + " for targ_distr_differ_along")
    for n in range(N):
        # Choose target location
        target_coord = (
            np.random.choice(range(grid_count)),
            np.random.choice(range(grid_count)),
        )
        # Choose dimensions along which target and distractor differ
        if display_type is not None:
            targ_distr_differ_along = display_type
        else:
            targ_distr_differ_along = np.random.choice(
                ["shape", "color", "both", "both_two_colors"]
            )
        colors = np.zeros((grid_count, grid_count)).astype(int)
        shapes = np.zeros((grid_count, grid_count))
        if targ_distr_differ_along == "shape":
            target_shape, distractor_shape = [0, 1]
            # Choose single color for all objects (either R, G, or B)
            colors += 0
            shapes += distractor_shape
            shapes[target_coord] = target_shape
        elif targ_distr_differ_along == "color":
            # Choose one color for target and different one for distractors
            target_color = 0
            target_shape = 0
            distractor_shape = 0
            distractor_color = np.random.choice(
                np.arange(3)[np.arange(3) != target_color]
            )
            shapes += distractor_shape
            shapes[target_coord] = target_shape
            colors += distractor_color
            colors[target_coord] = target_color
        elif targ_distr_differ_along == "both":
            # Choose color and shape for target, then let distractors be any
            # combination other than that, but also make sure there are no
            # other singletons among distractors which could be considered
            # targets, too.
            target_color = 0  # Red
            target_shape = 0  # Square
            # Make new list with the target object values removed
            # as an option
            combos_allowed = [
                c for c in combos if c[0] != target_shape or c[1] != target_color
            ]
            while True:
                conjunction_counts = {(i, j): 0 for i, j in combos}
                for i in range(grid_count):
                    for j in range(grid_count):
                        if (i, j) == target_coord:
                            colors[i, j] = target_color
                            shapes[i, j] = target_shape
                        else:
                            # Pick uniformly from list of allowed distractors.
                            # This ensures that the distractor object is never the
                            # same as the target
                            shapes[i, j], colors[i, j] = combos_allowed[
                                np.random.choice(len(combos_allowed))
                            ]
                            conjunction_counts[(shapes[i, j], colors[i, j])] += 1
                if all([v != 1 for v in conjunction_counts.values()]):
                    # Illegal to have any distractors be singletons. There must
                    # be at least one other like it. Passed this check.
                    break
        elif targ_distr_differ_along == "both_two_colors":
            # Same as 'both', but with only two colors, not three
            omitted_color = np.random.choice([1, 2])
            target_color = 0  # Red
            target_shape = 0  # Square
            # Make new list with the target object values removed
            # as an option
            combos_allowed = [
                c
                for c in combos
                if (c[0] != target_shape or c[1] != target_color)
                and c[1] != omitted_color
            ]
            while True:
                conjunction_counts = {(i, j): 0 for i, j in combos}
                for i in range(grid_count):
                    for j in range(grid_count):
                        if (i, j) == target_coord:
                            colors[i, j] = target_color
                            shapes[i, j] = target_shape
                        else:
                            # Pick uniformly from list of allowed distractors.
                            # This ensures that the distractor object is never the
                            # same as the target
                            shapes[i, j], colors[i, j] = combos_allowed[
                                np.random.choice(len(combos_allowed))
                            ]
                            conjunction_counts[(shapes[i, j], colors[i, j])] += 1
                if all([v != 1 for v in conjunction_counts.values()]):
                    # Illegal to have any distractors be singletons. There must
                    # be at least one other like it. Passed this check.
                    break
        x = make_display(colors, shapes, target_coord, grid_count)
        X.append(x)
        decision_targets.append(target_coord)
        if debug:
            import matplotlib.pyplot as plt

            plt.imshow(x)
            plt.show()
    X = np.array(X)
    decision_targets = np.array(decision_targets)
    return X, decision_targets


def attention_search_shape(N, size, grid_count=4, debug=False):
    return attention_search(
        N, size, grid_count=grid_count, display_type="shape", debug=debug
    )


def attention_search_color(N, size, grid_count=4, debug=False):
    return attention_search(
        N, size, grid_count=grid_count, display_type="color", debug=debug
    )


def attention_search_both(N, size, grid_count=4, debug=False):
    return attention_search(
        N, size, grid_count=grid_count, display_type="both", debug=debug
    )


def attention_search_both2(N, size, grid_count=4, debug=False):
    return attention_search(
        N, size, grid_count=grid_count, display_type="both_two_colors", debug=debug
    )


# def arbitrary_filter(filter_func, im0=None, size=(256, 256), filter_args=[]):
#     """
#     Filter white noise with user-specified filter function.

#     filter_func is function that specifies the filter value as a function of x-frequency
#     and y-frequency. That is, the Fourier component with x and y frequency of (fx, fy) will
#     be multiplied by filter_func(fx, fy)
#     """
#     if im0 is None:
#         im0 = np.random.randn(*size)
#     size = im0.shape
#     noiseF = np.fft.fftshift(fftpack.fft2(im0))
#     rng = np.fft.fftshift(np.fft.fftfreq(min(size), 1.0)) * min(
#         size
#     )  # Convert from /cycles/pixles to /cycles
#     fx, fy = np.meshgrid(rng, rng)
#     imageF = filter_func(fx, fy, filter_args) * noiseF
#     image = np.real(fftpack.ifft2(np.fft.ifftshift(imageF)))
#     return im0, image


# def pink_filter(im0=None, alpha=1.0, size=(100, 100)):
#     """Filter an image in frequency domain so that it has amplitude spectrum
#     proportional to f^-alpha.

#     im0: original image to be filtered
#     image: filtered version of original
#     """
#     if im0 is None:
#         im0 = np.random.randn(*size)
#     size = im0.shape
#     noiseF = np.fft.fftshift(fftpack.fft2(im0))
#     rng = np.fft.fftshift(np.fft.fftfreq(min(size), 1.0)) * min(
#         size
#     )  # Convert from /cycles/pixles to /cycles
#     fx, fy = np.meshgrid(rng, rng)
#     rho = np.sqrt(fx ** 2.0 + fy ** 2.0)
#     rho[np.where(rho == 0.0)] = np.inf
#     one_over_f = 1.0 / rho ** alpha
#     imageF = noiseF * one_over_f
#     image = np.real(fftpack.ifft2(np.fft.ifftshift(imageF)))
#     return im0, image


# def bandpass_filter(im0, low=0, high=100):
#     """Apply bandpass filter to an image with specified low and high frequency
#     cutoff points.
#     """
#     size = im0.shape
#     noiseF = np.fft.fftshift(fftpack.fft2(im0))
#     rng = np.fft.fftshift(np.fft.fftfreq(min(size), 1.0))
#     rng *= min(size)  # Convert from /cycles/pixles to /cycles
#     fx, fy = np.meshgrid(rng, rng)
#     rho = np.sqrt(fx ** 2.0 + fy ** 2.0)
#     freqs_keep = np.logical_and(rho >= low, rho <= high)
#     noiseF[np.logical_not(freqs_keep)] = 0.0  # Set frequencies outside of band to zero
#     image = np.real(fftpack.ifft2(np.fft.ifftshift(noiseF)))
#     return im0, image


# def image_filter(
#     N,
#     size,
#     filter_type="pink",
#     alpha=1.0,
#     low=0,
#     high=20,
#     probe_noise_std=0.1,
#     normalize=True,
#     no_probes=False,
#     pchange=0.5,
# ):
#     def norm_and_center(x):
#         """x shape: (batch, rows, columns)
#         """
#         x -= np.mean(x, (1, 2))[:, None, None]
#         return x / np.abs(x).max((1, 2))[:, None, None]

#     if "pink" in filter_type:
#         white_and_filtered_imgs = [
#             pink_filter(np.random.randn(size, size), alpha=alpha) for i in range(N)
#         ]
#         white_noise_imgs, targets = zip(*white_and_filtered_imgs)
#     elif "bandpass" in filter_type:
#         white_and_filtered_imgs = [
#             bandpass_filter(np.random.randn(size, size), low=low, high=high)
#             for i in range(N)
#         ]
#         white_noise_imgs, targets = zip(*white_and_filtered_imgs)
#     elif "white" in filter_type:
#         # Just white noise, no filter
#         white_noise_imgs = [np.random.randn(size, size) for i in range(N)]
#         targets = white_noise_imgs
#     else:
#         raise NotImplementedError
#     if normalize:
#         targets = norm_and_center(targets)
#     if no_probes:
#         return np.array(targets)
#     prob_change = np.random.choice(
#         2, size=N, p=(1 - pchange, pchange)
#     )  # Choose 'change' trials
#     probes = []
#     if filter_type in ["pink", "bandpass", "white"]:
#         # Pick probe randomly from target set (easy task)
#         for i in range(len(targets)):
#             if prob_change[i]:
#                 # 'Change' trial, so choose among all other images besides target
#                 inds = [ind for ind in range(N) if ind != i]
#                 i_probe = np.random.choice(inds)
#                 probes.append(targets[i_probe])
#             else:
#                 # 'Same' trial
#                 probes.append(targets[i])
#     elif filter_type == "pink_addednoiseprobe":
#         # Make probe using additive noise (harder task)
#         for i in range(len(targets)):
#             if prob_change[i]:
#                 # 'Change' trial, so add noise to original white-noise image
#                 # that produced target and then run it through filter
#                 # to get a probe that is similar but different to the target.
#                 # Control similarity with 'probe_noise_std'.
#                 im0 = white_noise_imgs[i] + np.random.normal(
#                     size=white_noise_imgs[i].shape, scale=probe_noise_std
#                 )
#                 probes.append(pink_filter(im0, alpha=alpha)[1])
#             else:
#                 # 'Same' trial
#                 probes.append(targets[i])
#     elif filter_type == "bandpass_addednoiseprobe":
#         # Make probe using additive noise (harder task)
#         for i in range(len(targets)):
#             if prob_change[i]:
#                 # 'Change' trial, so add noise to original white-noise image
#                 # that produced target and then run it through filter
#                 # to get a probe that is similar but different to the target.
#                 # Control similarity with 'probe_noise_std'.
#                 im0 = white_noise_imgs[i] + np.random.normal(
#                     size=white_noise_imgs[i].shape, scale=probe_noise_std
#                 )
#                 probes.append(bandpass_filter(im0, low=low, high=high)[1])
#             else:
#                 # 'Same' trial
#                 probes.append(targets[i])
#     elif filter_type == "white_addednoiseprobe":
#         for i in range(len(targets)):
#             if prob_change[i]:
#                 # 'Change' trial, so add noise
#                 # Control similarity with 'probe_noise_std'.
#                 im0 = white_noise_imgs[i] + np.random.normal(
#                     size=white_noise_imgs[i].shape, scale=probe_noise_std
#                 )
#                 probes.append(im0)
#             else:
#                 # 'Same' trial
#                 probes.append(targets[i])
#     else:
#         raise NotImplementedError
#     # Normalize
#     if normalize:
#         probes = norm_and_center(probes)
#     return np.array(targets), np.array(probes), np.array(prob_change)[:, None]


def generate_training_data(
    N,
    size,
    task_weights=[1.0, 1.0, 1.0],
    dataset="rectangle",
    conv=False,
    data_dir=".",
    dim=-1,
    mean=50,
    std=10,
    low=0,
    high=20,
    alpha=1.0,
    probe_noise_std=0.1,
    normalize=True,
    setsize=[1, 6],
    logistic_decision=True,
    seqlen=10,
    attention_search_type=None,
    parallel=True,
    train_test="train",
    pretrained_model=None
):
    recall_targets = None  # Some tasks don't use this
    RGB = False  # Whether image has 3 color channels or just one
    # if dataset == "rectangle":
    #     targets, probes, prob_change, lw = rectangle(
    #         N, size=size, task_weights=task_weights
    #     )
    # elif dataset == "2Dfloat":
    #     targets, probes, prob_change = simple_2Dfloat(N, task_weights=task_weights)
    # elif dataset == "NDim_float":
    #     targets, probes, prob_change, recall_targets = NDim_float(
    #         N, size, task_weights=task_weights
    #     )
    # elif dataset == "letter":
    #     targets, probes, prob_change = letter(N, size, task_weights)
    # elif dataset == "object_array":
    #     targets, probes, prob_change, recall_targets = object_array(
    #         N, size, setsize_rng=setsize
    #     )
    elif dataset == "gabor_array":
        targets, probes, prob_change, recall_targets = gabor_array(
            N, size, setsize_rng=setsize
        )
    elif dataset == "gabor_array1":
        targets, probes, prob_change, recall_targets = gabor_array1(
            N, size
        )
    elif dataset == "gabor_array2":
        targets, probes, prob_change, recall_targets = gabor_array2(
            N, size
        )
    elif dataset == "gabor_array3":
        targets, probes, prob_change, recall_targets = gabor_array3(
            N, size
        )
    elif dataset == "gabor_array4":
        targets, probes, prob_change, recall_targets = gabor_array4(
            N, size
        )
    elif dataset == "gabor_array5":
        targets, probes, prob_change, recall_targets = gabor_array5(
            N, size
        )
    elif dataset == "gabor_array6":
        targets, probes, prob_change, recall_targets = gabor_array6(
            N, size
        )
    elif dataset == "episodic_shape_race":
        RGB = True
        targets, probes, prob_change, recall_targets = episodic_shape_race(
            N, size, n_frames=seqlen
        )
    elif dataset == "episodic_setsize":
        RGB = True
        targets, probes, prob_change, recall_targets = episodic_setsize(
            N, size, n_frames=seqlen, parallel=parallel
        )
    elif dataset == "attention_search":
        RGB = True
        inputs, recall_targets = attention_search(
            N, size, display_type=attention_search_type
        )
        return inputs, recall_targets
    elif dataset == "attention_search_shape":
        RGB = True
        inputs, recall_targets = attention_search_shape(N, size)
        return inputs, recall_targets
    elif dataset == "attention_search_color":
        RGB = True
        inputs, recall_targets = attention_search_color(N, size)
        return inputs, recall_targets
    elif dataset == "attention_search_both":
        RGB = True
        inputs, recall_targets = attention_search_both(N, size)
        return inputs, recall_targets
    elif dataset == "attention_search_both2":
        RGB = True
        inputs, recall_targets = attention_search_both2(N, size)
        return inputs, recall_targets
    elif dataset == "attention_MOT":
        RGB = True
        targets, probes, recall_targets = attention_MOT(N, size)
    elif dataset == "cifar10":
        RGB = True
        inputs, targets, recall_targets = load_cifar10(train_test, data_dir)
        return inputs, targets, recall_targets
    elif dataset == "places365":
        inputs, targets, recall_targets = load_places365_batch(
            N, train_test, data_dir, pretrained_model)
        return inputs, targets, recall_targets
    elif dataset == "fruits360":
        inputs, recall_targets = load_fruits360(
            N, train_test, data_dir)
        return inputs, inputs, recall_targets
    elif dataset == "plants":
        # Load images into arrays
        images, stim_vals = load_plants(
            size, data_dir, layer_type="conv" if conv else "MLP"
        )
        # Make training set by pairing targets with probes and same/different
        # probabilities
        targets, probes, prob_change, recall_targets = plant_batch(
            N,
            images,
            stim_vals,
            task_weights=task_weights,
            dim=dim,
            mean=mean,
            std=std,
            logistic_decision=logistic_decision,
        )
    elif dataset == "plants_setsize":
        if dim not in [0, 1]:
            raise Exception(
                "Illegal value for `dim` for dataset 'plants_setsize'")
        images, stim_vals = load_plants(size // 3, data_dir, "conv")
        targets, probes, prob_change, recall_targets = plant_array(
            N, size // 3, images, stim_vals, setsize_rng=setsize, stim_dim=dim)
    elif dataset == "plants_setsize1":
        if dim not in [0, 1]:
            raise Exception(
                "Illegal value for `dim` for dataset 'plants_setsize'")
        images, stim_vals = load_plants(size // 3, data_dir, "conv")
        targets, probes, prob_change, recall_targets = plant_array(
            N, size // 3, images, stim_vals, setsize_rng=[1, 1], stim_dim=dim)
    elif dataset == "plants_setsize2":
        if dim not in [0, 1]:
            raise Exception(
                "Illegal value for `dim` for dataset 'plants_setsize'")
        images, stim_vals = load_plants(size // 3, data_dir, "conv")
        targets, probes, prob_change, recall_targets = plant_array(
            N, size // 3, images, stim_vals, setsize_rng=[2, 2], stim_dim=dim)
    elif dataset == "plants_setsize3":
        if dim not in [0, 1]:
            raise Exception(
                "Illegal value for `dim` for dataset 'plants_setsize'")
        images, stim_vals = load_plants(size // 3, data_dir, "conv")
        targets, probes, prob_change, recall_targets = plant_array(
            N, size // 3, images, stim_vals, setsize_rng=[3, 3], stim_dim=dim)
    elif dataset == "plants_setsize4":
        if dim not in [0, 1]:
            raise Exception(
                "Illegal value for `dim` for dataset 'plants_setsize'")
        images, stim_vals = load_plants(size // 3, data_dir, "conv")
        targets, probes, prob_change, recall_targets = plant_array(
            N, size // 3, images, stim_vals, setsize_rng=[4, 4], stim_dim=dim)
    elif dataset == "plants_setsize5":
        if dim not in [0, 1]:
            raise Exception(
                "Illegal value for `dim` for dataset 'plants_setsize'")
        images, stim_vals = load_plants(size // 3, data_dir, "conv")
        targets, probes, prob_change, recall_targets = plant_array(
            N, size // 3, images, stim_vals, setsize_rng=[5, 5], stim_dim=dim)
    elif dataset == "plants_setsize6":
        if dim not in [0, 1]:
            raise Exception(
                "Illegal value for `dim` for dataset 'plants_setsize'")
        images, stim_vals = load_plants(size // 3, data_dir, "conv")
        targets, probes, prob_change, recall_targets = plant_array(
            N, size // 3, images, stim_vals, setsize_rng=[6, 6], stim_dim=dim)
    elif dataset == "plants_categorical":
        # Load images into arrays
        images, stim_vals = load_plants(
            size, data_dir, layer_type="conv" if conv else "MLP"
        )
        # Make training set by pairing targets with probes and same/different
        # probabilities
        targets, probes, prob_change, recall_targets = plant_batch_categorical(
            N, images, stim_vals, categ_dim=dim
        )
    elif dataset == "plants_modal_prior":
        # Load images into arrays
        images, stim_vals = load_plants(
            size, data_dir, layer_type="conv" if conv else "MLP"
        )
        # Make training set by pairing targets with probes and same/different
        # probabilities (TODO: targets for decision module and probes)
        targets, probes, prob_change, recall_targets = plant_batch_modal_prior(
            N, images, stim_vals, dim=dim
        )
    elif dataset == "plants_modal_prior_1D":
        # Load images into arrays
        images, stim_vals = load_plants(
            size, data_dir, layer_type="conv" if conv else "MLP"
        )
        # Make training set by pairing targets with probes and same/different
        # probabilities (TODO: targets for decision module and probes)
        targets, probes, prob_change, recall_targets = plant_batch_modal_prior_1D(
            N, images, stim_vals, dim=dim
        )
    # elif dataset in FILTERS:
    #     targets, probes, prob_change = image_filter(
    #         N,
    #         size,
    #         low=low,
    #         high=high,
    #         filter_type=dataset,
    #         alpha=alpha,
    #         probe_noise_std=probe_noise_std,
    #         normalize=normalize,
    #     )
    else:
        raise Exception
    if conv:
        if not RGB:
            targets = targets[..., None]
            probes = probes[..., None]
    elif dataset in RECURRENT:
        shp = targets.shape
        targets = targets.reshape(N, shp[1], -1)
        probes = probes.reshape(N, shp[1], -1)
    else:
        targets = targets.reshape(N, -1)
        probes = probes.reshape(N, -1)
    return targets, probes, prob_change, recall_targets


# def image_entropy(dataset, N, imsize, alpha=1., low=0, high=20., bins=50,
#                   normalize=True):
#     """Calculate entropy of set of images. Gets histogram of pixel values and
#     calculates sum(-p(x) * log(p(x))). Increase bin size and number of images
#     to get more accurate estimate.
#     """
#     x = image_filter(N, imsize, low=low, high=high, filter_type=dataset,
#                      alpha=alpha, normalize=normalize, no_probes=True)
#     # import matplotlib.pyplot as plt
#     # fig, ax = plt.subplots()
#     # # ax.plot(hist)
#     # ax.imshow(x[0])
#     # plt.show()
#     return entropy(x, bins=bins)


# def entropy(x, bins=50):
#     H = np.histogram(x, bins=bins)
#     hist = H[0].astype(float) / H[0].sum()
#     return np.nansum(-hist * np.log2(hist))


# def save_image(x, pth):
#     """Normalize values to range from 0 to 255, assuming they currently range
#     from -1 to 1. Then save to pth.
#     """
#     x = (x + 1) * 255.
#     Image.fromarray(x).convert('RGB').save(pth)


# def show_image(x):
#     """Normalize values to range from 0 to 255, assuming they currently range
#     from -1 to 1. Then save to pth.
#     """
#     x /= np.abs(x).max()
#     x = (x + 1) * 255.
#     Image.fromarray(x).convert('RGB').show()


# def select_stimuli(N, se_range=[0, 200], nlp_range=[0.6, 0.65],
#                    entropy_range=[4.55, 4.65], imsize=100, alpha=1., low=0.,
#                    high=17.5, probe_noise_std=(0.5, 0.5), nlp=0.65, se=0.013,
#                    H_target=4.75):
#     """Select stimuli for psychophysics (change-detection) experiment,
#     controlling for square-error, luminance change, and Normalized Laplacian
#     Pyramid (NLP) perceptual loss. Specifically, keep values within
#     specified range for all trials across all noise types. (Note: Luminance
#     should already be effectively zero during stimulus generation, because the
#     mean is set to zero for each image before normalizing.)

#     Reference
#     Laparra, V., Balle, J., Berardino, A., & Simoncelli, E. P. (2016).
#         Perceptual image quality assessment using a normalized Laplacian
#         pyramid. Electronic Imaging, 2016(16), 1-6.
#     """
#     def squared_error(x, y):
#         """Per pixel squared error
#         """
#         se = ((x - y) ** 2).sum(axis=(1, 2)) / imsize ** 2.
#         return se

#     # def filter_stimuli(se, nlp):
#     #     return np.logical_and.reduce([se >= se_range[0],
#     #                                   se <= se_range[1],
#     #                                   nlp >= nlp_range[0],
#     #                                   nlp <= nlp_range[1]])

#     def generate(N, filter_type, alpha=1, low=0, high=17.5,
#                  probe_noise_std=0.5, normalize=True, pchange=1,
#                  above_nlp=True, above_se=True, above_H=True):
#         targets = []
#         probes = []
#         nlp_dists = []
#         sqerrs = []
#         entropies = []
#         n_trials = 0
#         while n_trials < N:
#             t, p, _ = image_filter(1, imsize, filter_type=filter_type,
#                                    alpha=alpha, low=low, high=high,
#                                    probe_noise_std=probe_noise_std,
#                                    normalize=normalize, pchange=pchange)
#             sqerr = squared_error(t, p)[0]
#             # if not se_range[0] <= sqerr <= se_range[1]:
#             if not ((above_se and sqerr >= se) or (not above_se and sqerr <= se)):
#                 print "Squared-error out of range ({})".format(sqerr)
#                 continue
#             H = entropy(t, bins=50)
#             if not ((above_H and H >= H_target) or (not above_H and H <= H_target)):
#                 print "Entropy out of range ({})".format(H)
#                 continue
#             # if not (entropy_range[0] <= H <= entropy_range[1]):
#             #     print "Entropy out of range"
#             #     continue
#             nlp_dist = eng.NLP_dist(matlab.double(t[0].tolist()),
#                                     matlab.double(p[0].tolist()))
#             # if nlp_range[0] <= nlp_dist <= nlp_range[1]:
#             if (above_nlp and nlp_dist >= nlp) or (not above_nlp and nlp_dist <= nlp):
#                 targets.append(t[0])
#                 probes.append(p[0])
#                 n_trials += 1
#                 nlp_dists.append(nlp_dist)
#                 sqerrs.append(sqerr)
#                 entropies.append(H)
#                 print n_trials
#             else:
#                 print "NLP error out of range ({})".format(nlp_dist)
#         return np.array(targets), np.array(probes), sqerrs, nlp_dists, entropies


#     # MATLAB engine to interface with NLP function
#     import matlab
#     import matlab.engine as mateng
#     eng = mateng.start_matlab()
#     targets_pink, probes_pink, sqerrs_pink, nlp_dists_pink, H_pink = generate(
#         N, 'pink_addednoiseprobe', alpha=alpha, above_se=False, above_H=True,
#         above_nlp=False, probe_noise_std=probe_noise_std[0])
#     targets_bp, probes_bp, sqerrs_bp, nlp_dists_bp, H_bp = generate(
#         N, 'bandpass_addednoiseprobe', low=low, high=high, above_H=False,
#         probe_noise_std=probe_noise_std[1])
#     # Save to file
#     import os
#     root = os.environ["SIMPLEOBJ"]
#     # exp_dir = Path(root).joinpath(
#     #     ("filters/stimuli/alpha{}_high{}_noiseSTD{}_seRange{}_nlpRange{}"
#     #      "_size{}").format(alpha, high, probe_noise_std, se_range, nlp_range,
#     #                        imsize))
#     exp_dir = Path(root).joinpath(
#         ("filters/stimuli/alpha{}_high{}_noiseSTD{}_se{}_nlp{}_H{}"
#          "_size{}").format(alpha, high, probe_noise_std, se, nlp, H_target,
#                            imsize))
#     img_dir = exp_dir.joinpath('images')
#     img_dir_pink = img_dir.joinpath("pink")
#     img_dir_bandpass = img_dir.joinpath("bandpass")
#     if not img_dir.exists():
#         img_dir.makedirs()
#     if not img_dir_pink.exists():
#         img_dir_pink.mkdir()
#     if not img_dir_bandpass.exists():
#         img_dir_bandpass.mkdir()
#     np.save(exp_dir.joinpath("targets_pink.npy"), targets_pink)
#     np.save(exp_dir.joinpath("probes_pink.npy"), probes_pink)
#     np.save(exp_dir.joinpath("targets_bandpass.npy"), targets_bp)
#     np.save(exp_dir.joinpath("probes_bandpass.npy"), probes_bp)
#     for i in range(N):
#         save_image(targets_pink[i], img_dir_pink.joinpath(
#             'target{}.png'.format(str(i).zfill(4))))
#         save_image(probes_pink[i], img_dir_pink.joinpath(
#             'probe{}.png'.format(str(i).zfill(4))))
#         save_image(targets_bp[i], img_dir_bandpass.joinpath(
#             'target{}.png'.format(str(i).zfill(4))))
#         save_image(probes_bp[i], img_dir_bandpass.joinpath(
#             'probe{}.png'.format(str(i).zfill(4))))
#         # tp = targets_pink[i]
#         # tp = (tp + 1) * 255.
#         # pp = probes_pink[i]
#         # pp = (pp + 1) * 255.
#         # tbp = targets_bp[i]
#         # tbp = (tbp + 1) * 255.
#         # pbp = probes_bp[i]
#         # pbp = (pbp + 1) * 255.
#         # Image.fromarray(tp).convert('RGB').save(img_dir_pink.joinpath(
#         #     'target{}.png'.format(str(i).zfill(4))))
#         # Image.fromarray(pp).convert('RGB').save(img_dir_pink.joinpath(
#         #     'probe{}.png'.format(str(i).zfill(4))))
#         # Image.fromarray(tbp).convert('RGB').save(img_dir_bandpass.joinpath(
#         #     'target{}.png'.format(str(i).zfill(4))))
#         # Image.fromarray(pbp).convert('RGB').save(img_dir_bandpass.joinpath(
#         #     'probe{}.png'.format(str(i).zfill(4))))
#     print np.mean(sqerrs_pink), np.mean(sqerrs_bp)
#     print np.mean(nlp_dists_pink), np.mean(nlp_dists_bp)
#     from ipdb import set_trace as BP; BP()
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(3)
#     axes[0].hist(sqerrs_pink, alpha=0.5)
#     axes[0].hist(sqerrs_bp, alpha=0.5)
#     axes[1].hist(nlp_dists_pink, alpha=0.5)
#     axes[1].hist(nlp_dists_bp, alpha=0.5)
#     axes[2].hist(H_pink, alpha=0.5)
#     axes[2].hist(H_bp, alpha=0.5)
#     plt.show()
#     ## Preloaded version
#     # targets_pink = []
#     # probes_pink = []
#     # targets_bp = []
#     # probes_bp = []
#     # for i in range(N):
#     #     targets_pink.append(
#     #         np.loadtxt(Path(data_dir).joinpath(
#     #             "pink", "target{}.txt".format(str(i).zfill(5)))))
#     #     probes_pink.append(
#     #         np.loadtxt(Path(data_dir).joinpath(
#     #             "pink", "probe{}.txt".format(str(i).zfill(5)))))
#     #     targets_bp.append(
#     #         np.loadtxt(Path(data_dir).joinpath(
#     #             "bandpass", "target{}.txt".format(str(i).zfill(5)))))
#     #     probes_bp.append(
#     #         np.loadtxt(Path(data_dir).joinpath(
#     #             "bandpass", "probe{}.txt".format(str(i).zfill(5)))))
#     # targets_pink = np.array(targets_pink)
#     # probes_pink = np.array(probes_pink)
#     # targets_bp = np.array(targets_bp)
#     # probes_bp = np.array(probes_bp)
#     # nlp_pink = np.loadtxt(Path(data_dir).joinpath('nlp_pink.txt'))[:N]
#     # nlp_bp = np.loadtxt(Path(data_dir).joinpath('nlp_bandpass.txt'))[:N]
#     # se_pink = squared_error(targets_pink, probes_pink)
#     # se_bp = squared_error(targets_bp, probes_bp)
#     # i_filtered_pink = filter_stimuli(se_pink, nlp_pink)
#     # i_filtered_bp = filter_stimuli(se_bp, nlp_bp)
#     # X_pink_filtered = targets_pink[i_filtered_pink]
#     # X_bp_filtered = targets_bp[i_filtered_bp]
#     # print len(X_pink_filtered), len(X_bp_filtered)


# def explore_stimuli(N, imsize, alpha=1., low=0, high=17.5, bins=50,
#                     normalize=True, probe_noise_std=(0.5, 0.5, 0.525),
#                     compute_entropies=False):
#     """
#     Compare information-theoretic and statistical properties of white noise
#     with different filters (pink, bandpass, or none). [TODO: how does
#     normalizing pixel values to (-1, 1) affect results?]

#     Compare:
#     1. Average image entropy (i.e. entropy of discretized distribution of pixel
#        values in a single image, averaged over many images with same filter)
#     2. Average squared error between target and probe (for given probe noise
#        magnitude)
#     3. Average change in luminance between target and probe

#     Results (100x100 px):
#     1. Using 50 bins, entropies are:
#         white: ~4.7 bits
#         pink: ~4.75 bits
#         bandpass: ~4.74 bits
#     2. Pink is approximately matched to bandpass=(0, 17.5)
#     3. Luminance change can be set to zero by subtracting the mean pixel value
#        from each image.

#     256 x 256 px (probe noise 0.5 for both pink and bandpass):
#     Entropy approximately matched at:
#         pink: alpha=1 (H=4.602)
#         bandpass: high=38 (H=4.603)
#     MSE (per pixel):
#         pink: 0.01192
#         bandpass: 0.01193
#     (Close enough? If not, can play with probe noise)
#     """
#     def squared_error(X):
#         se = ((X[0] - X[1]) ** 2).sum(axis=(1, 2)) / imsize ** 2.
#         mse = se.mean()
#         return se, mse

#     def delta_luminance(X):
#         return np.abs(X[0].sum(axis=(1, 2)) - X[1].sum(axis=(1, 2)))

#     def hist(X, xmin, xmax, nbins=21):
#         H, x = np.histogram(X, bins=np.linspace(xmin, xmax, nbins))
#         H = H / (H.sum()).astype(float)
#         return H, x

#     def plot(x, y1, y2, y3, ymin, ymax, labels=("pink", "bandpass", "white"),
#              title="", xlabel="", ylabel="frequency"):
#         fig, ax = plt.subplots()
#         ax.plot(x[:-1], y1, label=labels[0])
#         ax.plot(x[:-1], y2, label=labels[1])
#         ax.plot(x[:-1], y3, label=labels[2])
#         ax.set_ylim(ymin, ymax)
#         ax.set_title(title)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         plt.legend()
#         return ax

#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     if compute_entropies:
#         print "Computing entropies..."
#         H_white = image_entropy('white', N, imsize, normalize=True, bins=bins,
#                                 low=low, high=high, alpha=alpha)
#         H_pink = image_entropy('pink', N, imsize, normalize=True, bins=bins,
#                                low=low, high=high, alpha=alpha)
#         H_bp = image_entropy('bandpass', N, imsize, normalize=True, bins=bins,
#                              low=low, high=high, alpha=alpha)
#         print "Image entropies (pink, bandpass, white): ", H_white, H_pink, H_bp
#     # Make datasets (target-probe pairs) for 1/f and bandpass, respectively
#     xpink = image_filter(N, imsize, low=low, high=high,
#                          filter_type='pink_addednoiseprobe', alpha=alpha,
#                          normalize=normalize,
#                          probe_noise_std=probe_noise_std[0],
#                          pchange=1)
#     xbandpass = image_filter(N, imsize, low=low, high=high,
#                              filter_type='bandpass_addednoiseprobe',
#                              alpha=alpha,
#                              normalize=normalize,
#                              probe_noise_std=probe_noise_std[1],
#                              pchange=1)
#     xwhite = image_filter(N, imsize, filter_type="white_addednoiseprobe",
#                           normalize=normalize,
#                           probe_noise_std=probe_noise_std[2],
#                           pchange=1)
#     # Compare squared-errors between targets and probes
#     se_pink, mse_pink = squared_error(xpink)
#     se_bandpass, mse_bandpass = squared_error(xbandpass)
#     se_white, mse_white = squared_error(xwhite)
#     print "MSE (pink, bandpass, white): ", mse_pink, mse_bandpass, mse_white
#     xmin = min(se_pink.min(), se_bandpass.min(), se_white.min())
#     xmax = max(se_pink.max(), se_bandpass.max(), se_white.max())
#     hist_bp, x = hist(se_bandpass, xmin, xmax)
#     hist_pink = hist(se_pink, xmin, xmax)[0]
#     hist_white = hist(se_white, xmin, xmax)[0]
#     ymin = 0
#     ymax = max(hist_pink.max(), hist_bp.max(), hist_white.max())
#     plot(x, hist_pink, hist_bp, hist_white, ymin, ymax,
#          title="MSE: alpha={}, high={}, probe_std={}".format(
#              alpha, high, probe_noise_std), xlabel="MSE")
#     # Compare luminance changes between targets and probes
#     lum_pink = delta_luminance(xpink)
#     lum_bandpass = delta_luminance(xbandpass)
#     lum_white = delta_luminance(xwhite)
#     print "Luminance change (pink, bandpass, white): ", (
#         lum_pink.mean(), lum_bandpass.mean(), lum_white.mean())
#     xmin = min(lum_pink.min(), lum_bandpass.min(), lum_white.min())
#     xmax = max(lum_pink.max(), lum_bandpass.max(), lum_white.max())
#     hist_bp, x = hist(lum_bandpass, xmin, xmax)
#     hist_pink = hist(lum_pink, xmin, xmax)[0]
#     hist_white = hist(lum_white, xmin, xmax)[0]
#     ymin = 0
#     ymax = max(hist_pink.max(), hist_bp.max())
#     plot(x, hist_pink, hist_bp, hist_white, ymin, ymax,
#          title="Luminance: alpha={}, high={}, probe_std={}".format(
#              alpha, high, probe_noise_std),
#          xlabel="mean luminance delta")
#     plt.show()
#     from ipdb import set_trace as BP; BP()
#     # x, y, _ = image_filter(5000, 100, filter_type='white_addednoiseprobe', high=17.5, probe_noise_std=0.5, pchange=1)
#     # [np.savetxt('filters/laplacian_pyramid_data/white/probe{}.txt'.format(str(i).zfill(5)), y[i]) for i in range(len(y))]


# def rectangle(N, size=30, task_weights=[1/3., 1/3., 1/3.]):
#     """Each image contains a single rectangle whose upper-left-most point
#     is anchored at the image corner.
#     """
#     # Probability of change that training targets assign for no-change
#     # trials (must be non-zero, because we use sigmoid)
#     min_prob_change = 0.0001
#     # Probability of change that training targets assign for change trials
#     # at maximum distance allowed by stimulus set
#     max_prob_change = 0.99999999999
#     decision_bias = -np.log(1. / min_prob_change - 1.)
#     max_dist = 2. * size ** 2.  # Assumes squared error loss
#     decision_scale = (-np.log(1. / max_prob_change - 1.) - decision_bias
#                       ) / (max_dist / 16.)

#     task_weights = np.array(task_weights) / float(sum(task_weights))
#     L = range(1, size + 1)
#     l0 = np.random.choice(L, size=N)
#     l1 = np.random.choice(L, size=N)
#     change_trial = np.random.choice(2, size=N, p=[0.1, 0.9])
#     target_images = []
#     probe_images = []
#     prob_change = []
#     for i in range(N):
#         img = np.zeros((size, size))
#         img_probe = np.zeros((size, size))
#         if change_trial[i]:
#             # Change trial (either l0 changes, l1 changes, or both)
#             change_type = np.random.choice(3, p=task_weights)
#             if change_type == 0:
#                 change_l0 = 1
#                 change_l1 = 0
#             if change_type == 1:
#                 change_l0 = 0
#                 change_l1 = 1
#             if change_type == 2:
#                 change_l0 = 1
#                 change_l1 = 1
#             if change_l0:
#                 valid_probe_vals_l0 = [l for l in L if l != l0[i]]
#                 l0_probe = np.random.choice(valid_probe_vals_l0)
#             else:
#                 l0_probe = l0[i]
#             if change_l1:
#                 valid_probe_vals_l1 = [l for l in L if l != l1[i]]
#                 l1_probe = np.random.choice(valid_probe_vals_l1)
#             else:
#                 l1_probe = l1[i]
#         else:
#             # Same trial
#             l0_probe = l0[i]
#             l1_probe = l1[i]
#         target_probe_dist = (l0_probe - l0[i]) ** 2. + (l1_probe - l1[i]) ** 2.
#         prob_change.append(
#             1. / (1. + np.exp(-(decision_scale * target_probe_dist + decision_bias))))
#         img[0:l0[i], 0:l1[i]] = 1
#         img_probe[0:l0_probe, 0:l1_probe] = 1
#         target_images.append(img)
#         probe_images.append(img_probe)
#     return (np.array(target_images), np.array(probe_images), change_trial,
#             np.array(prob_change)[:, None], np.vstack([l0, l1]).T)
