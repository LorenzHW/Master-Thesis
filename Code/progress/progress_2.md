## MNIST
Corresponding notebooks: 
* [Train MNIST variational autoencoder with triplet loss](https://colab.research.google.com/drive/1KlqlHuqF8-m-FHftM83gWyrU9UOwD8gv)

#### **VAE**
Hard facts of VAE training (not using triplet loss) - **VAE**:
* Latent dimension: 2  
* Epochs: 100  
* Train reconstruction loss: 152,77  
* Test reconstruction loss: 154,32
* Test triplet loss: 1.6711

Latent representation during training (60 000 samples) every 10 epochs:

Epoch 10|Epoch 20
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_10.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_20.png)
 
Epoch 30|Epoch 40
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_30.png) |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_40.png)
 
Epoch 50|Epoch 60
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_50.png) |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_60.png)

Epoch 70|Epoch 80
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_70.png) |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_80.png)

Epoch 90|Epoch 100
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_90.png) |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_100.png) 


Reconstruction loss during training for every sample every 10 epochs 


Epoch 10|Epoch 20
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_10_loss.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_20_loss.png)

Epoch 30|Epoch 40
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_30_loss.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_40_loss.png)

Epoch 50|Epoch 60
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_50_loss.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_60_loss.png)

Epoch 70|Epoch 80
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_70_loss.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_80_loss.png)

Epoch 90|Epoch 100
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_90_loss.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/epoch_100_loss.png)


#### **VAE_TL**
Hard facts of VAE training with triplet loss and reconstruction loss - **VAE_TL**:
* Latent dimension: 2  
* Epochs: 100  
* Train reconstruction loss: 152.42  
* Test reconstruction loss: 154.61
* Test triplet loss: **1.14**

Latent representation of the two VAEs:

VAE|VAE_TL
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/test_data.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/test_data_tl.png)


Reconstruction loss of the VAEs:

VAE|VAE_TL
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/test_data_loss.png)  |  ![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_2/test_data_tl_loss.png)


One interesting fact is following:
* Before training with a combined loss on 100 epochs, I trained the VAE alternately. First 10 epochs with RC-Loss then 10
epochs with 10 epochs with TL-Loss and so on. This resulted the loss that currently is not used for training to increase again
* Combining both losses for back propagation decreased both losses