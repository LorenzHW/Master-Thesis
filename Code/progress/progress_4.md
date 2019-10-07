## MNIST
* This experiment is to verify two points: is the TL function correctly implemented and is the encoder learning what it is supposed to learn
* I trained the encoder two times with the triplet loss (no ELBO). The first time I used the default MNIST labels as ground truth and the 
second time I used customized labels based on the entropy of an external NN.

Corresponding notebook:
* [VAE_TL](https://colab.research.google.com/drive/15TGjQRX4du1Ox_SAMWCHGggo2y5DVLOv)

#### Using default MNIST labels for triplet loss
Latent representation:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_4/latent_rep_triplet_loss_mnist_labels.png)

Yellow points are wrongly classified by the external NN:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_4/y_vs_pred_triplet_loss_mnist_labels.png)

#### Using customized labels based on entropy of output layer of an external NN for triplet loss
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_4/latent_rep_triplet_loss_custom_labels.png)

Yellow points are wrongly classified by the external NN:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_4/y_vs_pred_triplet_loss_custom_labels.png)


Observations:
* We see clearly that the triplet loss works correctly for the first case where we use the default MNIST labels
* For the second case where we use customized labels based on the entropy of an external NN the encoder is not learning properly.
A proper representation would be wrongly classified points clustered together (points with high entropy).