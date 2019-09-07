## Code for my Master Thesis

### Prerequisite:

For **Anaconda** users:
    
    conda env create --file environment_{your-os}.yml
    
For **pip** users:
    
    pip install -r requirements.txt


### Contribution

If you need to add a dependency, don't forget to update the environment:

    conda env export --no-build > environment_{your-os}.yml
    
And pip: 

    pip freeze > requirements.txt