from itertools import product
import numpy as np
from path import Path
from PIL import Image
from scipy.stats import truncnorm, norm


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
    """Names the checkpoint directory such that the name includes information
    about the choice of training conditions and network parameters. The info
    included in the name depends on dataset and task.

    The first part of the name is the name of the dataset being trained on.
    The next part includes information about the image size, relative weights
    of the different components of the loss function (rate-loss, reconstruction
    loss, decision loss), number of units in different layers (hidden layers of
    encoder and decoder, latent layer, decision layer), number of layers in
    each module (encoder, decoder, decision module), the kind of reconstruction
    and decision-module loss function used (e.g. square-error), which dimension
    the input distribution is applied to (e.g. leaf width or leaf angle; given
    by an int) and the mean and std of that distribution, relative weights
    applied to each output dimension of the decision module in computing the
    loss, and which kind of decision target is being used (e.g.
    "same_different", a binary comparison between target and probe, or
    "recall", where the network must produce the exact value of a stimulus
    dimension).
    """
    if net.__dict__.get("input_distribution_dim") is not None:
        # If input_distribution_dim = -1, this specifies a sort of default
        # option: In the case of the "plants" dataset, this signifies that
        # both leaf width and leaf angle are to be drawn from the uniform
        # distribution. In the case of all other datasets, a value of -1 will
        # either raise an exception or be ignored. For example, in the
        # "plants_categorical" dataset, one or the other dimension must be
        # specified, since the task is designed to have the category boundary
        # be orthogonal to one of the dimensions. (Code could be modified
        # though, to include more general category boundaries.)
        if net.input_distribution_dim != -1:
            inputdist_str = "_inputdist{}_{}_{}".format(
                net.input_distribution_dim, net.input_mean, net.input_std
            )
        else:
            inputdist_str = ""
    if net.dataset in ["gabor_array", "gabor_array1", "gabor_array2",
                       "gabor_array2", "gabor_array3", "gabor_array4",
                       "gabor_array5", "gabor_array6", "plants_categorical",
                       "plants_modal_prior_1D", "plants_modal_prior",
                       "plants_setsize", "plants_setsize1", "plants_setsize2",
                       "plants_setsize3", "plants_setsize4", "plants_setsize5",
                       "plants_setsize6"]:
        # For these datasets, we encode in the file name certain aspects of the
        # architecture, such as number of layers and hidden units for each
        # module, in addition to the weights on the terms of the loss function.
        # We do not specify, however, a dataset sampling distribution or task
        # weights, as with the "plants" dataset, which we use to explore how
        # response biases change as the entropy of the prior is modulated and
        # weights on the decision variables are changed in the loss function.
        # In all datasets besides "plants", we assume uniform weighting on the
        # loss function terms.
        pth = Path(base_dir).joinpath(
            "{}_width{}_{}_lossweights{}_{}_{}_{}_{}_hidden{}_latent{}_decisi"
            "on{}_layers{}_{}_{}_reconloss_{}".format(
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
        if "plant" in net.dataset:
            # For plants datasets, need to specify which dimension (leaf width
            # or leaf angle) is the dimension of interest (while the other is
            # drawn uniform random).
            pth += "_dim{}".format(net.input_distribution_dim)
    elif net.dataset == "plants":
        # Here, we specify the dataset sampling distribution, which we use to
        # explore how response biases change as the entropy of the prior is
        # modulated, as well as the loss weights, which we use to explore how
        # response biases change as each dimension is weighted more or less
        # relative to the other.
        pth = Path(base_dir).joinpath(
            "{}_width{}_{}_lossweights{}_{}_{}_{}_{}_hidden{}_latent{}_decisi"
            "on{}_layers{}_{}_{}_taskweights{}{}_losses_{}_{}_dectarget"
            "_{}".format(
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
    elif net.dataset in [
            "fruits360", "attention_search", "attention_search_shape",
            "attention_search_color", "attention_search_both",
            "attention_search_both2"]:
        # For datasets trained on fixed architectures, we have a more minimal
        # set of specifications
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
    else:
        raise NotImplementedError
    return pth


def make_dataset_pths(net):
    if net.dataset in ["plants_categorical", "plants_modal_prior_1D",
                       "plants_modal_prior", "plants_setsize",
                       "plants_setsize1", "plants_setsize2",
                       "plants_setsize3", "plants_setsize4",
                       "plants_setsize5", "plants_setsize6"]:
        # For these datasets, just need to specify image dimension, number of
        # samples to generate, and the stimulus dimension to apply distribution
        # to. (Note for these, the distribution is considered fixed, so the
        # shape of the distribution is not specified here.)
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_width{}_size{}_dim{}/".format(
                net.dataset,
                net.image_width,
                net.dataset_size,
                net.input_distribution_dim
            )
        )
    elif net.dataset in ["attention_search", "attention_search_shape",
                         "attention_search_color", "attention_search_both",
                         "attention_search_both2", "fruits360"]:
        # Image dimensions are fixed, so do not include in file name
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_size{}".format(
                net.dataset,
                net.dataset_size))
    elif net.dataset in ["gabor_array", "gabor_array1", "gabor_array2",
                         "gabor_array3", "gabor_array4", "gabor_array5",
                         "gabor_array6"]:
        # In these datasets, we do not need to choose a dimension, like with
        # the plants datasets. There is only one possible stimulus dimension
        # which is angle of orientation. Thus, dim is excluded from file name.
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_width{}_size{}/".format(
                net.dataset,
                net.image_width,
                net.dataset_size,
            )
        )
    elif net.dataset == "plants":
        # Dataset is N iid Gaussian samples from full set of images, so we need
        # to specify parameters of that distribution. (Note:
        # 'plants_modal_prior' and 'plants_modal_prior_1D' could also
        # fall under this category, but I kept the distribution fixed so I
        # excluded mean and std from the file name)
        if net.__dict__.get("input_distribution_dim") not in [-1, None]:
            input_dist_str = "_mean{}_std{}_dim{}".format(
                net.input_mean, net.input_std, net.input_distribution_dim
            )
        else:
            input_dist_str = ""
        dataset_dir = net.trainingset_dir.joinpath(
            "{}_width{}_taskweights{}{}_size{}_{}/".format(
                net.dataset,
                net.image_width,
                "_".join([str(t) for t in net.task_weights]),
                input_dist_str,
                net.dataset_size,
                # Logistic function for binary same/different judgment or
                # target-probe distance (in units of stimulus space) for
                # recall decision (i.e. recalling the precise difference in
                # stimulus values between target and probe)
                "logistic" if net.logistic_decision else "tpdist"
            )
        )
    else:
        raise NotImplementedError()
    dataset_pth = dataset_dir.joinpath("data.pkl")
    return dataset_dir, dataset_pth


def load_or_make_dataset(net, dataset_pth, dataset_dir, dataset_size):
    import pickle

    # Fill in some possibly missing attributes and set to None
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
        # Reshape data if necessary, depending on conv or MLP architecture
        if len(X.shape) == 2 and net.layer_type == "conv":
            # Reshape data so it works for convolutional architecture
            X = X.reshape(-1, net.image_width, net.image_width,
                          net.image_channels)
            if Probes is not None:
                Probes = Probes.reshape(-1, net.image_width, net.image_width,
                                        net.image_channels)
        elif len(X.shape) > 2 and net.layer_type == "MLP":
            # Reshape data so it works for MLP architecture
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


def calc_decision_params(max_dist, min_prob_change=0.0001,
                         max_prob_change=1 - 1e-8):
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
        omega ** 2 / (4 * np.pi * K ** 2) * np.exp(
            -omega ** 2 / (8 * K ** 2) * (4 * x1 ** 2 + y1 ** 2))
    )
    sinusoid = func(omega * x1) * np.exp(K ** 2 / 2)
    gab = gauss * sinusoid
    return gab


def gabor_array(N, size, setsize_rng=[1, 6], output_dim=None):
    from PIL import Image
    from itertools import product

    cycles_per_image = 120.0
    gabor_coords = [
        (size / 3 * i, size / 3 * j) for j in range(3) for i in range(2)]
    if output_dim is None:
        output_dim = max(setsize_rng)
    targets = []
    probes = []
    prob_change = []
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
            gab = gabor(
                [int(size / 3) - 1] * 2, cycles_per_image / size, theta)
            gab = (255 * gab).astype(int)
            for k, m in product(range(len(gab)), range(len(gab))):
                pixmap[gabor_coords[j][0] + k, gabor_coords[j][1] + m] = int(
                    gab[k, m])
        targets.append(np.array(target))
        orientations.append(np.array(thetas))
    targets = np.array(targets) / 255.
    orientations = np.array(orientations)
    # Could fill in values for these if desired:
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
        # Append color dimension, since these are grayscale
        images = np.array(images)[..., None]
        stim_vals = np.array(stim_vals)[..., None]
    elif layer_type == "MLP":
        # Flatten for MLP input
        images = np.array(images).reshape(len(images), -1)
        stim_vals = np.array(stim_vals).reshape(len(images), -1)
    else:
        raise Exception
    if normalize:
        images = images / 255.
    return images, stim_vals


def plant_batch(N, images, stim_vals, task_weights=[1.0, 1.0],
                targets_only=False, dim=-1, mean=50, std=5,
                sample_var_factors=[1.0, 1.0], logistic_decision=True):
    """Make plants training batch of specified size by combining targets with
    random probes. 'images' is full dataset, preloaded. 'mean', and 'std'
    specify a normal distribution to draw target stimuli from, and 'dim'
    specifies which dimension to draw from. The other dimension remains
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
    decision_bias, decision_scale = calc_decision_params(
        (width_max - width_min) / 2.0)
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
                    np.logical_and(i_w == stim_vals[:, 0],
                                   i_d == stim_vals[:, 1])
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
                    np.logical_and(i_w == stim_vals[:, 0],
                                   i_d == stim_vals[:, 1])
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
                    stim_vals[:, 0] == width_probe,
                    stim_vals[:, 1] == droop_probe
                ).squeeze()
            ][0]
        )
        tp_dist = np.sum(
            np.abs((width_probe, droop_probe) - tval) * task_weights)
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


def plant_batch_modal_prior(N, images, stim_vals, dim=0, means=[25, 75],
                            std=5):
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
            np.where(np.logical_and(
                i_w == stim_vals[:, 0], i_a == stim_vals[:, 1]))[0]
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
            np.where(np.logical_and(
                i_w == stim_vals[:, 0], i_a == stim_vals[:, 1]))[0]
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
        category_dirs = Path(dataset_dir).joinpath(
            "places365_standard", "train").dirs()
    elif train_test == "test":
        category_dirs = Path(dataset_dir).joinpath(
            "places365_standard", "val").dirs()
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
        # How many times this category was drawn:
        n = (cat_inds == cat_ind).sum()
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
        lab = np.array(
            [labels_dict[img_files[i].splitall()[-2]] for i in img_inds])
        # Downsample images
        x1 = [
            np.array(img.resize((128, 128), Image.ANTIALIAS)) for img in imgs]
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
        category_dirs = Path(dataset_dir).joinpath(
            "fruits-360", "Training").dirs()
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


def attention_search_oddball(N, size, grid_count=4, display_type=None):
    """Input is array of objects, similar to classic search
    time experiments (e.g. Treisman). Decision target is location of target
    object. Idea is to show that popout happens when display is easily
    compressed.

    'Oddball' means that the subject does not know ahead of time which feature
    they are looking for, just that it should be unique from distractors.

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
        X = np.array(
            [x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
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
        X = np.array(
            [x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
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
            target_shape, distractor_shape = [
                [0, 1], [1, 0]][np.random.choice(2)]
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
                c for c in combos if c[0] != target_shape or
                c[1] != target_color
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
                            # This ensures that the distractor object is never
                            # the same as the target
                            shapes[i, j], colors[i, j] = combos_allowed[
                                np.random.choice(len(combos_allowed))
                            ]
                            conjunction_counts[
                                (shapes[i, j], colors[i, j])] += 1
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
                if (c[0] != target_shape or c[1] != target_color) and
                c[1] != omitted_color
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
                            # This ensures that the distractor object is never
                            # the same as the target
                            shapes[i, j], colors[i, j] = combos_allowed[
                                np.random.choice(len(combos_allowed))
                            ]
                            conjunction_counts[
                                (shapes[i, j], colors[i, j])] += 1
                if all([v != 1 for v in conjunction_counts.values()]):
                    # Illegal to have any distractors be singletons. There must
                    # be at least one other like it. Passed this check.
                    break
        x = make_display(colors, shapes, target_coord, grid_count)
        X.append(x)
        decision_targets.append(target_coord)
    X = np.array(X)
    decision_targets = np.array(decision_targets)
    probes = np.zeros_like(X)  # Filler
    prob_change = np.zeros((N, 1))  # Filler
    return X, probes, prob_change, decision_targets


def attention_search(N, size, grid_count=4, display_type=None):
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
        X = np.array(
            [x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
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
        X = np.array(
            [x for x in X if np.all(x >= 0) and np.all(x < canvas_size)]).T
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
                c for c in combos if c[0] != target_shape or
                c[1] != target_color
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
                            # This ensures that the distractor object is never
                            # the same as the target
                            shapes[i, j], colors[i, j] = combos_allowed[
                                np.random.choice(len(combos_allowed))
                            ]
                            conjunction_counts[
                                (shapes[i, j], colors[i, j])] += 1
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
                c for c in combos if
                (c[0] != target_shape or c[1] != target_color) and
                c[1] != omitted_color]
            while True:
                conjunction_counts = {(i, j): 0 for i, j in combos}
                for i in range(grid_count):
                    for j in range(grid_count):
                        if (i, j) == target_coord:
                            colors[i, j] = target_color
                            shapes[i, j] = target_shape
                        else:
                            # Pick uniformly from list of allowed distractors.
                            # This ensures that the distractor object is never
                            # the same as the target
                            shapes[i, j], colors[i, j] = combos_allowed[
                                np.random.choice(len(combos_allowed))
                            ]
                            conjunction_counts[
                                (shapes[i, j], colors[i, j])] += 1
                if all([v != 1 for v in conjunction_counts.values()]):
                    # Illegal to have any distractors be singletons. There must
                    # be at least one other like it. Passed this check.
                    break
        x = make_display(colors, shapes, target_coord, grid_count)
        X.append(x)
        decision_targets.append(target_coord)
    X = np.array(X)
    decision_targets = np.array(decision_targets)
    return X, decision_targets


def attention_search_shape(N, size, grid_count=4):
    return attention_search(
        N, size, grid_count=grid_count, display_type="shape")


def attention_search_color(N, size, grid_count=4):
    return attention_search(
        N, size, grid_count=grid_count, display_type="color")


def attention_search_both(N, size, grid_count=4):
    return attention_search(
        N, size, grid_count=grid_count, display_type="both")


def attention_search_both2(N, size, grid_count=4):
    return attention_search(
        N, size, grid_count=grid_count, display_type="both_two_colors")


def generate_training_data(N, size, task_weights=[1.0, 1.0, 1.0],
                           dataset="rectangle", conv=False, data_dir=".",
                           dim=-1, mean=50, std=10, low=0, high=20, alpha=1.0,
                           normalize=True, setsize=[1, 6],
                           logistic_decision=True, attention_search_type=None,
                           parallel=True, train_test="train",
                           pretrained_model=None):
    recall_targets = None  # Some tasks don't use this
    RGB = False  # Whether image has 3 color channels or just one
    if dataset == "gabor_array":
        targets, probes, prob_change, recall_targets = gabor_array(
            N, size, setsize_rng=setsize)
    elif dataset == "gabor_array1":
        targets, probes, prob_change, recall_targets = gabor_array1(
            N, size)
    elif dataset == "gabor_array2":
        targets, probes, prob_change, recall_targets = gabor_array2(
            N, size)
    elif dataset == "gabor_array3":
        targets, probes, prob_change, recall_targets = gabor_array3(
            N, size)
    elif dataset == "gabor_array4":
        targets, probes, prob_change, recall_targets = gabor_array4(
            N, size)
    elif dataset == "gabor_array5":
        targets, probes, prob_change, recall_targets = gabor_array5(
            N, size)
    elif dataset == "gabor_array6":
        targets, probes, prob_change, recall_targets = gabor_array6(
            N, size)
    elif dataset == "attention_search":
        RGB = True
        inputs, recall_targets = attention_search(
            N, size, display_type=attention_search_type)
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
            size, data_dir, layer_type="conv" if conv else "MLP")
        # Make training set by pairing targets with probes and same/different
        # probabilities
        targets, probes, prob_change, recall_targets = plant_batch(
            N, images, stim_vals, task_weights=task_weights, dim=dim,
            mean=mean, std=std, logistic_decision=logistic_decision)
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
            size, data_dir, layer_type="conv" if conv else "MLP")
        # Make training set by pairing targets with probes and same/different
        # probabilities
        targets, probes, prob_change, recall_targets = plant_batch_categorical(
            N, images, stim_vals, categ_dim=dim)
    elif dataset == "plants_modal_prior":
        # Load images into arrays
        images, stim_vals = load_plants(
            size, data_dir, layer_type="conv" if conv else "MLP")
        # Make training set by pairing targets with probes and same/different
        # probabilities
        targets, probes, prob_change, recall_targets = plant_batch_modal_prior(
            N, images, stim_vals, dim=dim)
    elif dataset == "plants_modal_prior_1D":
        # Load images into arrays
        images, stim_vals = load_plants(
            size, data_dir, layer_type="conv" if conv else "MLP")
        # Make training set by pairing targets with probes and same/different
        # probabilities
        targets, probes, prob_change, recall_targets = \
            plant_batch_modal_prior_1D(N, images, stim_vals, dim=dim)
    else:
        raise Exception
    if conv:
        if not RGB:
            targets = targets[..., None]
            probes = probes[..., None]
    else:
        targets = targets.reshape(N, -1)
        probes = probes.reshape(N, -1)
    return targets, probes, prob_change, recall_targets
