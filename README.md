# efficientPerceptualDataCompression
Scripts for training and analyzing neural network models in forthcoming Psychological Review manuscript, "Efficient Data Compression in Perception and Perceptual Memory" (Bates & Jacobs, 2020).

Dependencies:
* tensorflow, version==1.12
* numpy
* scipy
* PIL (can install via: pip install python-pillow)

In addition, there are two image datasets that must be downloaded: the Fruits-360 dataset (https://github.com/Horea94/Fruit-Images-Dataset) and my own dataset of downsampled plant-like images (https://drive.google.com/open?id=170VD85Ke4vExYhcXviqGh-pVy1EIAJlS). For plants-related experiments, the `--image_width` argument in commands below can be supplied with the value 100 or 120, as these correspond to the two different subdirectories in the image set (but note that "plants_setsize" should be three times these values, 300 or 360, since it copies them onto a 3x3 grid).

For all cases below, the following is true:
`--rate_loss_weight` scales the gradient steps corresponding to the KL divergence term in the loss function
`--reconstruction_loss_weights` scales the gradient steps corresponding to the reconstruction (pixel square-error) term in the loss
`--decision_loss_weights` scales the gradient steps corresponding to the decision-loss term (see manuscript)
For both reconstruction and decision loss weights, there are two numbers, the first corresponding to the encoder weights, and the second corresponding to any remaining weights affecting the loss (i.e. those involved in decoding from the latent units). For instance, if the first number is set to 0 for the decision loss, then the weights of the encoder will not be trained with respect to the decision loss.


To train models corresponding to Figure 7 (plants with varying capacities), run:

`python VAE.py --dataset plants --rate_loss_weight <VALUE GREATER THAN ZERO> --reconstruction_loss_weights 1 1 --decision_loss_weights 0 0 --latent 500 --hidden 500 --layer_type MLP --image_width 120 --decoder_layers 2 --encoder_layers 2 --learning_rate 0.0001 --batch 128 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

Note that decision loss is set to zero (i.e. it is ignored). `--encoder_layers` and `--decoder_layers` determine the number of layers in the encoder and decoder respectively.


To train models corresponding to Figure 8 (plants with varying prior distributions), run:

`python VAE.py --dataset plants --dim <0 | 1> --mean 50 --std <10 | 10000> --rate_loss_weight <VALUE GREATER THAN ZERO> --reconstruction_loss_weights 1 1 --decision_loss_weights 0 0 --latent 500 --hidden 500 --layer_type MLP --image_width 120 --decoder_layers 2 --encoder_layers 2 --learning_rate 0.0001 --batch 128 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

where `--mean` and `--std` control the mean and standard deviation of the sampling distribution along either leaf width (`--dim 0`) or leaf angle (`--dim 1`).


To train models corresonding to the set-size experiments, run:

`python VAE.py --dataset plants_setsize<N> --rate_loss <VALUE GREATER THAN ZERO> --decision_loss_weights 0.01 0.01 --dim <0 | 1> --latent 500 --decision_size 100 --hidden 500 --layer_type MLP --image_width <300 | 360> --decoder_layers 2 --encoder_layers 2 --dataset_size 5000 --regenerate_steps 10000 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

where `--dim` is whether the output of the decision module is recall of leaf width (0) or leaf angle (1). Valid arguments to `--dataset` are "plants_setsize1", "plants_setsize2", ..., "plants_setsize6".

To train models corresponding to Figure 11 (only penalizing one stimulus dimension, either leaf width or leaf angle), run:

`python VAE.py --dataset plants --rate_loss 1e-8 --reconstruction_loss_weights 0 1 --task_weights <0 1 | 1 0> --decision_dim 2 --dim <0 | 1> --latent 500 --decision_size 100 --hidden 500 --layer_type MLP --image_width 120 --decoder_layers 2 --encoder_layers 2 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

Setting `--task_weights` to "0 1" makes leaf width the irrelevant dimension, while setting it to "1 0" makes leaf angle the irrelevant dimension.

To train models corresponding to Figure 12 (top) (categorical bias with plants stimuli via categorical loss), run:

`python VAE.py --dataset plants_categorical --rate_loss <VALUE GREATER THAN ZERO> --reconstruction_loss_weights 0.0001 1 --dim <0 | 1> --latent 500 --decision_size 100 --hidden 500 --layer_type MLP --image_width 120 --decoder_layers 2 --encoder_layers 2 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

To train models corresponding to Figure 12 (bottom) (categorical bias with plants stimuli via bimodal prior), run:

`python VAE.py --dataset plants_modal_prior_1D --rate_loss <VALUE GREATER THAN ZERO> --decision_loss_weights 0 0 --dim <0 | 1> --latent 500 --hidden 500 --layer_type MLP --image_width 120 --decoder_layers 2 --encoder_layers 2 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

Note that decision loss is ignored, as this model only depends on the prior distribution. `--dim` specifies which stimulus dimension to apply the bimodal distribution to, leaving the other uniformly distributed.


To train models corresponding to the fruits experiments, run:

`python VAE_fruits.py --rate_loss_weight <VALUE GREATER THAN ZERO> --reconstruction_loss_weights .1 1 --batch 128 --learning_rate 0.0001 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING fruits-360>`


To train models on pop-out task, run:

`python VAE_popout.py --dataset attention_search_shape --dataset_size 1000 --regenerate_steps 1000 --rate_loss_weight <VALUE GREATER THAN ZERO> --reconstruction_loss_weights 1 1 --decision_loss_weights .01 1 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <LOCATION TO SAVE AUTO-GENERATED DATASETS>`

`--dataset` controls which kind of visual search display is generated, and can be:
* "attention_search_shape": all objects are same color but target and distractor differ by shape (used in manuscript)
* "attention_search_color": all objects are same shape but target and distractor differ by color
* "attention_search_both": objects vary by both color and shape, and targets may match distractors along one of those dimensions
* "attention_search_both2": same as attention_search_both, but restricts space of colors to red and blue rather than red, blue, green (used in manuscript)

`--dataset_size` determines the size of the randomly generated training set, while `--regenerate_steps` controls how often the training set is regenerated. If a very large training set cannot fit into memory, one can generate a smaller training set but periodically regenerate it to alleviate overfitting.

To produce the figures in the paper, five scripts have been provided. To reproduce the results in Figure 6, run:

`python analyze.py vae --dataset plants <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --visualize`

To reproduce the results in Figure 7, run:

`python analyze.py vae --dataset plants <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --correlation`

Note the first argument (`vae`) specifies to build the network using VAE.py.

To reproduce the results in Figure 8, run:

`python analyze.py vae --dataset plants_setsize<N> <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --visualize`

To reproduce the results in Figure 9, run:

`python plot_plants_setsize_effects.py`

(Please refer to the file to see or modify which network parameter settings are used.)

To reproduce the results in Figure 10, run:

`python analyze.py vae --dataset plants <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --plants_average_reconstruction`

To reproduce the results in Figure 11, run:

`python analyze.py vae --dataset plants_modal_prior_1D <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --visualize_grid`

`python analyze.py vae --dataset plants_categorical <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --visualize_grid`

To reproduce the results in Figure 12, run:

`python analyze.py vae --dataset plants_modal_prior_1D <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --correlation`

`python analyze.py vae --dataset plants_categorical <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --correlation`

To reproduce the results in Figure 13, run:

`python plot_fruits_reconstructions.py`

To reproduce the results in Figure 14, run:

`python plot_fruits_memory_pca.py`

To reproduce the results in Figure 15, run:

`python analyze.py popout --dataset attention_search_shape <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --visualize`

`python analyze.py popout --dataset attention_search_both2 <COMMAND LINE ARGUMENTS USED WHEN TRAINING CORRESPONDING NETWORK> --visualize`

To reproduce the results in Figure 16, run:

`python plot_popout_setsize_effects.py`

To produce RD-curves for gabor dataset (not included in final manuscript), run:

`python plot_gabor_setsize_effects.py`
