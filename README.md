# efficientPerceptualDataCompression
Scripts for training neural network models in forthcoming Psychological Review manuscript, "Efficient Data Compression in Perception and Perceptual Memory" (Bates, Jacobs).

Dependencies:
* tensorflow==1.12
* numpy
* scipy
* python-pillow

In addition, there are two image datasets that must be downloaded: the Fruits-360 dataset (https://github.com/Horea94/Fruit-Images-Dataset) and my own dataset of downsampled plant-like images (https://drive.google.com/open?id=170VD85Ke4vExYhcXviqGh-pVy1EIAJlS).

For all cases below, the following is true:
`--rate_loss_weight` scales the gradient steps corresponding to the KL divergence term in the loss function
`--reconstruction_loss_weights` scales the gradient steps corresponding to the reconstruction (pixel square-error) term in the loss
`--decision_loss_weights` scales the gradient steps corresponding to the decision-loss term (see manuscript)
For both reconstruction and decision loss weights, there are two numbers, the first corresponding to the encoder weights, and the second corresponding to any remaining weights affecting the loss (i.e. those involved in decoding from the latent units). For instance, if the first number is set to 0 for the decision loss, then the weights of the encoder will not be trained with respect to the decision loss.


To train models corresponding to Figure 7 (plants with varying capacities), run:

`python VAE.py --dataset plants --rate_loss_weight <VALUE GREATER THAN ZERO> --reconstruction_loss_weights 1 1 --decision_loss_weights 0 0 --latent 500 --decision_size 100 --hidden 500 --layer_type MLP --image_width 120 --decoder_layers 2 --encoder_layers 2 --dataset_size 10000 --regenerate_steps 10000 --learning_rate 0.0001 --batch 128 --pregenerate --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

Note that decision loss is set to zero (i.e. it is ignored). `--encoder_layers` and `--decoder_layers` determine the number of layers in the encoder and decoder respectively.


To train models corresponding to Figure 8 (plants with varying prior distributions), run:

`python VAE.py --dataset plants --dim 1 --decision_dim 2 --mean 50 --std 10 --rate_loss_weight <VALUE GREATER THAN ZERO> --reconstruction_loss_weights 1 1 --decision_loss_weights 0 0 --latent 500 --decision_size 100 --hidden 500 --layer_type MLP --image_width 120 --decoder_layers 2 --encoder_layers 2 --dataset_size 10000 --regenerate_steps 10000 --learning_rate 0.0001 --batch 128 --pregenerate --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

where `--mean` and `--std` control the mean and standard deviation of the sampling distribution along either leaf width (`--dim 0`) or leaf angle (`--dim 1`).


To train models corresonding to the set-size experiments, run:

`python VAE.py --dataset plants_setsize<N> --rate_loss <VALUE GREATER THAN ZERO> --decision_loss_weights 0.01 0.01 --dim 1 --latent 500 --decision_size 100 --hidden 500 --layer_type MLP --image_width 360 --decoder_layers 2 --encoder_layers 2 --dataset_size 1000 --regenerate_steps 10000 --checkpoint_dir <DESIRED CHECKPOINTS SAVE DIR> --trainingset_dir <PATH TO DIR CONTAINING 'plant_stimuli'>`

where `--dim` is whether the output of the decision module is recall of leaf width (0) or leaf angle (1). Valid arguments to `--dataset` are "plants_setsize1", "plants_setsize2", ..., "plants_setsize6".


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
