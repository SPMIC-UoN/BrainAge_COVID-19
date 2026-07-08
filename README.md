# Brain age modelling, prediction, and its application on COVID-19 data

This repository contains scripts for Brain Age prediction to explore the effects of COVID-19 and the pandemic on brain ageing, as described in this paper: [A. R. Mohammadi-Nejad, M. Craig, E. Cox, X. Chen, R. G. Jenkins, S. Francis, S. N. Sotiropoulos, D. P. Auer, “Brains Under Stress: Unravelling the Effects of the COVID-19 Pandemic on Brain Ageing”, medrxiv, 2024][paper-medrxiv-link].

# Repository Overview
This repository has two sections: the first contains the code for the brain age estimation model, available in the 'predictive_model' folder. The second section includes the code used to produce the figures for the paper, located in the 'analysis' folder.

## 1- Brain age modelling and prediction

This module supports the training of a model to predict age from a set of IDPs or
other measures. The model can then be used to determine a subject's 'brain age'
from a set of the same measured, and hence a 'brain delta' that expresses
how much older or younger their brain appears compared to their actual age.

Based on method described in [S. M. Smith, D. Vidaurre, F. Alfaro-Almagro, T. E. Nichols, K. L. Miller, "Estimation of brain age delta from brain imaging, NeuroImage, 2019][paper-neuroimage-link], multiple models are defined in this paper including:
 - A simple model that has been shown to be biased when the model retains age dependence within the prediction.
 - A corrected 'unbiased' model can be used to address this issue.
 - In addition, quadratic age dependence can be modelled and an alternate approach in which age is regarded as a predictor of IDPs rather than the other way round can be used.

Reproduction of simulation results can be found in the ``predictive_model/examples/`` directory

### User guide

Conceptually, brain age prediction is in two parts - first, we train a model
using known true ages and a set of features (typically IDPs or other metrics)
from some groups of subjects. Once trained, we can then use this model to
predict the ages of a set of subjects given their true ages and values of
the same features used to train the model.

Of course, in practice these two steps are often combined, we train the model
and then use it to predict the ages of the same set of subjects, and may then be
interested in comparing the predicted 'brain age' with the actual ages
of the subjects. 

However the set of subjects used for training and prediction
does not have to be the same, this is used in the cross-validation analysis
in the Smith paper (code to do this is also in the ``predictive_model/examples/`` folder)

## 2- Brain age estimation for COVID-19 data
The repository contains all the necessary resources to reproduce the figures from our paper. Specifically, we have provided a comprehensive data file that includes all the relevant information required for figure generation. Additionally, a Python script is included, which reads the data file and generates the figures as presented in the paper. This script ensures that all preprocessing steps, data transformations, and plotting commands are executed correctly to recreate the figures accurately. By following the instructions in the repository, users can seamlessly reproduce our results and explore the data further.

<!-- References -->

[paper-medrxiv-link]: https://www.medrxiv.org/content/10.1101/2024.07.22.24310790v1
[paper-neuroimage-link]: https://www.sciencedirect.com/science/article/pii/S1053811919305026?via%3Dihub


