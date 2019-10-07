## MNIST
Corresponding notebook:
* [VAE_TL](https://colab.research.google.com/drive/1eJATlLv6hivc7uOBduFvbQi5HaCXlC_y)

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
* 