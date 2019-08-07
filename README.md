# STCA-DSNN

This is a matlab implementation of STCA on GPUs using CUDA.

STCA is an algorithm for training deep spiking neural networks to resolve the temporal credit assignment problem and classification tasks.

It has been already accepted by *IJCAI-19*.

## Citation

If you use STCA in your reseasrch, please cite the following paper:

```
@inproceedings{ijcai2019-189,
    title     = {STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks},
    author    = {Gu, Pengjie and Xiao, Rong and Pan, Gang and Tang, Huajin},
    booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, {IJCAI-19}},
    publisher = {International Joint Conferences on Artificial Intelligence Organization},
    pages     = {1366--1372},
    year      = {2019},
    month     = {7},
    doi       = {10.24963/ijcai.2019/189},
    url       = {https://doi.org/10.24963/ijcai.2019/189},
}
```

## Requirements

- MATLAB R2016b
- CUDA 9.0
- This version can be run at both Windows and Linux.

## Getting started

- Note that this version can only construct **fully-connected networks**. The following version will support convolutional structures.

### Folder organization

- ***Experiments:*** Experiments completed in the paper (MNIST classification, instrument recognition, and unsegmented sound events detection).

- ***Algorithms:*** Algorithms for training and testing.

- ***Cuda:*** CU and PTX files which are crucial for the parallel processing of this implementation.

- ***Data:*** Some data used in experiments.

- ***Encoding:*** Encoding methods converting other signals into spike domain.

- ***Tools:*** Functions which may be convenient in some cases.

### Formats of data and network weights

To run the codes in the 'Algorithm' folder, we need to construct a struct (called **Data**) to record all the required information in the training set or testing set. This **Data** contains 4 fields as follow:

- ***Labels_name:*** Names of all classes, type: 1 x NClasses cell, each cell is a 'str' constant, Nclass is the number of classes contained in the dataset.

- ***Labels:*** Labels of all samples, type: 1 x NSamples double, each item is an integer (starting from 1), NSamples is the number of samples.

- ***ptn:*** Spiking patterns of all samples, type: NSamples x NAfferents cell, NSamples is the number of samples, NAfferents is the number of spike trains in a spike pattern. In this field, each cell is a vector (type: double) and indicates a spike train (the values in the vector are the independent spike timings, unit: second. If the afferent didn't fire a spike, the cell is []), and the cells in a same row construct an indpendent spike pattern. 

- ***Tmax:*** the max spike timing of each spike pattern, type: NSamples x 1. This field can be directly computed by the function 'get_Tmax' in the 'Tools' folder.

After training, the network weights and some information about training will also be recorded in a struct (called **Structure**) and the testing procedure will load the **Structure** to test the performance. 

## More experimental settings

More experimental settings (e.g., parameters, error curves) can be found from subfolders in the ***Experiments***.
I'm sorry about that the codes in ***Algorithms*** are certainly confused, since different experiments are conducted on different versions of this work. 
I might unify these codes in the future version.

## Contact
If you have some problems on this implementation, please contact me by this e-mail: gupj1202@gmail.com
