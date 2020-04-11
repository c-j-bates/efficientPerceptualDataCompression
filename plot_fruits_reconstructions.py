import os
import numpy as np
from VAE_fruits import make_net
from collections import namedtuple
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from path import Path

CHECKPOINT_DIR = os.environ['MEMNET_CHECKPOINTS']
TRAININGSET_DIR = os.environ['MEMNET_TRAININGSETS']


def build(beta, batch_size):
    vae_args = {
        "activation": "relu",
        "checkpoint_dir": CHECKPOINT_DIR,
        "trainingset_dir": TRAININGSET_DIR,
        "dataset": "fruits360",
        "regenerate_steps": 10000,
        "batch": batch_size,
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


def load_images(img_pths):
    images = [np.array(Image.open(pth).resize((112, 112))) for pth in img_pths]
    images = np.array(images)
    images = images / 255.  # Normalize to [0, 1]
    return images


def reconstruct(net, X, n_samples=5):
    """
    Draw multiple samples for each image in `img_pths`. Reshape output
    so that first dim is which image, second dim is which sample, remaining
    dims are pixels
    """
    Xhat = net.predict(
        np.repeat(X, n_samples, axis=0),
        keep_session=True).reshape([len(X), n_samples] + list(X.shape[1:]))
    Xhat = np.clip(Xhat, 0, 1)
    return Xhat


def format(ax):
    ax.set_xticks([])
    ax.set_xticklabels("")
    ax.set_yticks([])
    ax.set_yticklabels("")
    plt.tight_layout()


def plot(XX, YY, gap=30, img_size=112):
    """
    XX is shape (betas, images, ...)
    YY is shape (betas, images, samples, ...)
    """
    n_betas = len(XX)  # different beta values
    n_targets = len(XX[0])  # target images (inputs to network)
    n_samples = len(YY[0][0])  # samples from decoder
    pixel_width = img_size * (1 + n_samples * n_betas) + gap * 3
    pixel_height = img_size * n_targets
    # Global canvas, on which to place all images
    canvas = Image.new("RGB", (pixel_width, pixel_height))
    canvas.paste(
        Image.fromarray(
            np.uint8(
                np.ones((pixel_height, pixel_width, 3)) * 255.)),
        box=(0, 0))
    # Draw vertical divider lines
    draw = ImageDraw.Draw(canvas)
    for i in range(n_betas):
        draw.line(
            (
                int(gap * (i + 0.5)) + img_size * (1 + i * n_samples),
                0,
                int(gap * (i + 0.5)) + img_size * (1 + i * n_samples),
                pixel_height
            ), fill=(0, 0, 0), width=3
        )
    # Iterate over beta values
    for i, (X, Y) in enumerate(zip(XX, YY)):
        # Iterate over images
        for j, (x, y) in enumerate(zip(X, Y)):
            # Paste target (left-most column of figure)
            canvas.paste(
                Image.fromarray(
                    np.clip(np.uint8(x * 255.), 0, 255)),
                box=(0, j * img_size))
            # Iterate over samples
            for k in range(len(y)):
                x_pos = img_size * (1 + k + n_samples * i) + gap * (i + 1)
                y_pos = j * img_size
                canvas.paste(
                    Image.fromarray(
                        np.clip(np.uint8(y[k] * 255.), 0, 255)),
                    box=(x_pos, y_pos))
    save_pth = "plots/fruits/fruits_reconstructions.pdf"
    canvas.save(save_pth)
    print("Saved to " + save_pth)


n_samples = 5
betas = [1e-7, 0.001, 0.01]
img_pths = [
    "Test/Apple Braeburn/r_99_100.jpg",
    "Test/Apple Golden 2/r_54_100.jpg",
    "Test/Apple Red Delicious/0_100.jpg",
    "Test/Tomato Maroon/1_100.jpg",
    "Test/Tomato 3/0_100.jpg",
    "Test/Tomato 4/10_100.jpg",
    "Test/Banana/36_100.jpg",
    "Test/Banana/86_100.jpg",
    "Test/Banana Red/5_100.jpg",
]
img_pths = [
    Path(TRAININGSET_DIR).joinpath("fruits-360", pth) for pth in img_pths
]
XX = []  # List over beta values
YY = []
for beta in betas:
    X = load_images(img_pths)
    net = build(beta, batch_size=len(X) * n_samples)
    Y = reconstruct(net, X, n_samples=n_samples)
    XX.append(X)
    YY.append(Y)
    del net
    tf.reset_default_graph()  # Must include to delete TF graph
plot(XX, YY)
