# MozzaVID dataset quickstart examples

>A set of functions enabling a quick start in training models on the MozzaVID dataset, as well as evaluation of the performance of models reported in the dataset paper.

## Prerequisites

Complete set of requred packages can be installed through the requirements file:

```  
pip install -r requirements.txt
```

File ```requirements_with_versions.txt``` specfies exact versions of the packages, can be used if relevant. 

Pytorch is commented out in both requirement files, since it requires a system-specific installation. 

## Data and models

Associated dataset and models can be downloaded from [link](todo.com)

The paths to both models nad data have to be adjusted in the ```evaluate_model.py``` and ```train_model.py```

## Example code

A simple model training can be run with the ```train_model.py``` script. Existing models can be evaluated with ```evaluate_model.py```. Both models contain a list of hyperparameters that allow exploring all the variations of the dataset.

## Reference

If you use our dataset, or any of this code for academic work, please consider citing our paper.

TODO

## License

MIT License (see LICENSE file).