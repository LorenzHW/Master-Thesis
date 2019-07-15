## Code for my Master Thesis

### Prerequisite:

For **Anaconda** users:
    
    conda env create --file environment_{your-os}.yml
    
For **pip** users:
    
    pip install -r requirements.txt

### Overview:
The directory `variational_autoencoder` contains an implementation of a variational autoencoder and an mnist example.
The folder `variational_autoencoder/weights` contains saved configurations of the model.


### Usage:
To run the mnist example, run:

    python -m variational_autoencoder.mnist

You can either train a new model from scratch or use existing weights (set the corresponding flag inside `variational_autoencoder/mnist.py`).
In case you create a new model from scratch your weights will be saved to `variational_autoencoder/weights`. Additionally,
an image will be generated after each epoch.

### Contribution

If you need to add a dependency, don't forget to update the environment:

    conda env export --no-build > environment_{your-os}.yml
    
And pip: 

    pip freeze > requirements.txt