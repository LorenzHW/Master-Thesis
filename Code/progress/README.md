## MNIST
Corresponding notebooks: 
* [Train MNIST variational autoencoder](https://colab.research.google.com/drive/1SHP5yunom4LZDHRbAPpbGzZPtvpFPLJc)
* [Train MNIST classifier](https://colab.research.google.com/drive/1ExE-VrCrn1OxJR3Sbpil6B0q_UGwLGgS)

Hard facts of VAE training:
* Latent dimension: 2  
* Epochs: 150  
* Train loss: 151,65  
* Validation loss: 153,52  
* Test loss: 153,60

Latent representation of each sample from the test data (10 000 examples)
![alt text](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/latent_rep.png "Logo Title Text 1")


Latent representation of each sample from the test data in correlation with the loss of a classifier.
The darker the point the higher the loss was for that prediction:
![alt text](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/latent_rep_loss.png "Logo Title Text 1")

## Cifar-10
Corresponding notebooks:
* [Train cifar-10 variational autoencoder](https://colab.research.google.com/drive/1U1Fo3YtnAqUiZ3zmaFaELFkOGyKFcoGS)
* [Train cifar-10 MobileNetV2](https://colab.research.google.com/drive/1vXIlagm1hakFnPJ5Fs17WVakJow2ZiL4#scrollTo=L4TFpNygapo0)    

Hard facts of VAE training:
* Latent dimension: 2  
* Epochs: 100  
* Train loss: 1953,92 (Loss is decreasing very slowly)
* Validation loss: 1956,43  
* Test loss: 1960,59

Hard facts of MobileNetV2 training:
* Train accuracy: 99%
* Validation accuracy: 94%  
* Test data accuracy: 93%
* Fine tuning was needed to go from 79% to 93%

TODO: This is a mess. Try transfer learning inside of the encoder
Latent representation of each sample from the test data (10 000 examples). 
![alt text](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/latent_rep_cifar.png "Logo Title Text 1")

Latent representation of each sample from the test data in correlation with the loss of a classifier (transfer learning with MobileNetV2).
The darker the point the higher the loss was for that prediction:
![alt text](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/latent_rep_loss_cifar.png "Logo Title Text 1")

The plotter notebook:
* [Plotting for MNIST and cifar 10](https://colab.research.google.com/drive/1Gofh_CrEp9cYRSxQwpsBEzor8AFmnFtO)