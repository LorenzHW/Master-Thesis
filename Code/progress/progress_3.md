## MNIST
* Compared to [progress_2](https://github.com/LorenzHW/Master-Thesis/blob/master/Code/progress/progress_2.md) I use different
[labels](https://github.com/LorenzHW/Master-Thesis/tree/master/Code/progress/questions) during training of the VAE. 
I used a binary label to differentiate the losses of an external classifier as high or low.
* Compared to [progress_2](https://github.com/LorenzHW/Master-Thesis/blob/master/Code/progress/progress_2.md) I plot on
on the test data again. I am plotting the loss of the external classifier in correlation with latent representation (not the ELBO) just 
like I did in [progress_1](https://github.com/LorenzHW/Master-Thesis/blob/master/Code/progress/progress_1.md).

Corresponding notebooks: 
* [Train MNIST variational autoencoder with triplet loss](https://colab.research.google.com/drive/1KlqlHuqF8-m-FHftM83gWyrU9UOwD8gv)

#### **VAE**
Hard facts of VAE training (not using triplet loss) - **VAE**:
* Latent dimension: 2  
* Epochs: 100  
* Test ELBO loss: 154.1644
* Test triplet loss: 3.9303

#### **VAE_TL**
Hard facts of VAE training with triplet loss and ELBO loss - **VAE_TL**:
* Latent dimension: 2  
* Epochs: 100  
* Test ELBO loss: 154.4483
* Test triplet loss: **1.9043**

Latent representation of both VAEs:

VAE|VAE_TL
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_3/test_data.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_3/test_data_tl.png)

Latent representation of each sample from the test data in correlation with the loss of a classifier. The darker the point the higher the loss was for that prediction:

VAE|VAE_TL
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_3/test_data_external_clf_loss.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_3/test_data_tl_external_clf_loss.png)

Observations:
* We see that the 