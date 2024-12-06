# MozzaVID dataset quickstart examples

>A set of functions enabling a quick start in training models on the MozzaVID dataset, as well as evaluation of the performance of models reported in the dataset paper.

### [[Data](https://archive.compute.dtu.dk/files/public/projects/MozzaVID/)] [Paper - TBA] [[Project website](https://papieta.github.io/MozzaVID/)]

## Prerequisites

Complete set of required packages can be installed through the requirements file:

```  
pip install -r requirements.txt
```

File ```requirements_with_versions.txt``` specfies exact versions of the packages, can be used if found relevant. 

Pytorch is commented out in both requirement files, since it may require a system-specific installation. 

## Data and models

Associated dataset can be downloaded from [here](https://archive.compute.dtu.dk/files/public/projects/MozzaVID/)

Models can be downloaded from: TBA

The paths to both models and data have to be adjusted in the ```evaluate_model.py``` and ```train_model.py```

## Example code

A simple model training can be run with the ```train_model.py``` script. Existing models can be evaluated with ```evaluate_model.py```. Both models contain a list of hyperparameters that allow exploring all the variations of the dataset.

## Reference

If you use our dataset, or any of this code for academic work, please consider citing our publication.

TBA

## License

MIT License (see LICENSE file).