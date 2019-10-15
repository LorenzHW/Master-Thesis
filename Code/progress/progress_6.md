## End-to-end run
Putting everything together:

**Step 1**:  
A neural network (NN) trained on MNIST:
* Number of epochs: 7
* Test loss: 0.0783
* Test accuracy: 0.9753

Graph of validation (blue) and train (orange) loss:

![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_6/epoch_loss.svg?sanitize=true)

Graph of validation (blue) and train (orange) accuracy:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_6/epoch_accuracy.svg?sanitize=true)


**Step 2**:   
A variational autoencoder is trained with TL and ELBO:
* Latent dimension: 50
* VAE is used to generate 738 images. Those are exactly the 738 images that the NN is classifying incorrectly.


**Step 3**:    
The neural network (NN) is trained again from scratch with the additional 738 images added to the original dataset.

