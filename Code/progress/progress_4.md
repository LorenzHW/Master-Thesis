## MNIST
* Checking whether triplet loss is implemented correctly: comparing training of encoder on different labels

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