## Variational autoencoder overview
![alt text](https://drive.google.com/uc?export=view&id=1v_B-wGfoZRCvl02aMu2jQfqu0m7XLnPp "Variational autoencoder")


#### Latent representation of each number
The following figures represent the ***2D sampled latent vector*** of an autoencoder. Each point represents a number 
from MNIST dataset (10 000 points overall) in the 2D vector space:

[30 epochs](https://github.com/LorenzHW/Master-Thesis/tree/master/Code/variational_autoencoder/weights/30_epochs)|[420 epochs](https://github.com/LorenzHW/Master-Thesis/tree/master/Code/variational_autoencoder/weights/420_epochs)
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/variational_autoencoder/imgages/latent_rep_30_epochs.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/variational_autoencoder/imgages/latent_rep_420_epochs.png)
Train loss: 169.5494    | Train loss: 153.4698
Test loss: -156.79      | Test loss: -151.2663