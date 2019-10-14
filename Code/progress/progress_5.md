### Triplet Loss Experiment
Corresponding notebook:
* [VAE_TL_EXPERIMENT](https://colab.research.google.com/drive/10C19G2S4p_v7qhOqDB1Kinb9C_w9bTxT)


It turns out that there was a bug inside my code that I did not notice.
So I redid the experiment where I test whether the triplet loss is working properly
with customized labels: I train the encoder of the VAE only with TL. As labels binary values are used (1 and 0).
Where 1 represents a data point with high loss (with respect to the external NN) and 0 represents a data point with low
loss (with respect to the external NN).

Latent representation of the MNIST test data: Every point with the 'X' marker is wrongly
classified. Additionally, the loss for each sample is visualized with the saturation.
The more purple a data point is the lower the loss:

![](https://raw.githubusercontent.com/LorenzHW/Master-Thesis/master/Code/progress/pics/progress_5/triplet_loss_experiment.png)

Observations:
* We see that the encoder tries to distinguish data points with high loss and data points with low loss: On the left side
of the latent representation are mostly data points that are classified correctly and have a low loss. On the right side
we see the opposite.