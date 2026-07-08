# Brain age modelling and prediction

### Command line interface

The command line interface is ``deltab``

A typical usage would involve the following inputs:

 - A file (e.g. `TRUE_AGES.txt`) containing a listing of true ages for subjects
 - A second file (e.g. `IDPS.txt`) in TSV or CSV format containing the values of the features to use
   in training the model. The features in this file are the columns, each
   subject is a row

We can then run the prediction as:

```
deltab  --train-ages TRUE_AGES.txt --train-features IDPS.txt \
        --predict-model unbiased_quadratic --predict-output BRAIN_AGE.txt --predict age
```

#### Training options

``--feature-nans`` : If set to ``median`` (default), any NaN values in the features for a subject are replace with the median value
of this feature. If ``remove``, subjects with any NaN value in the features are removed. Note that you will then want to use 
``--true-ages-output`` to ensure that you have corresponding true ages to compare with excluding the removed subjects

``--feature-proportion`` : Typically PCA reduction is performed on the feature set and not all features are retained for the
prediction. This option specifies a proportion between 0 and 1 of features to be retained.

``--feature-num`` : An alternative to ``feature-proportion``, this option specifies an exact number of features to retain in PCA

``--feature-var`` : Another alternative to ``feature-proportion``, this retains features to explain the specified proportion of 
total variance

``--kaiser-guttmann`` : Yet another alternative to ``feature-proportion``, this uses the Kaiser-Guttmann criterion to select the
number of features to retain

#### Prediction options

``--predict-model`` : defines the model to use in prediction. For full details see Smith 2019:

 - ``simple`` is a naive regression model that is shown in Smith 2019 to be biased when age dependence is present in the features
 - ``unbiased`` removes age dependence from the features to eliminate the age bias in the ``simple`` model
 - ``unbiased_quadratic`` also models quadratic dependence on true age
 - ``alternate`` is an alternative approach in which age is viewed as a predictor of features rather than the other way round
 - ``alternate_quadratic`` adds quadratic age dependence to the ``alternate`` model


``--predict`` : Specifies what to output, ``age`` to output predicted age or ``delta`` to output the difference between true age
and predicted age

``--predict-output`` : Name of file to write predicted age/delta to

``--true-ages-output`` : Name of file to write true ages used in prediction. Normally this would be identical to the true age input
but if ``--feature-nans`` is set to ``remove``, it will contain the ages only of the subjects that were included

``--predict-ages`` : If the subjects to be predicted are not the same as those used to train the model, this file should contain
the true ages of the prediction subjects

``--predict-features`` : Similarly, when making predictions for a different set of subjects to those used in training, this file
should be CSV or TSV in format and contain the feature values for the prediction subjects.

#### Other options

``--save`` : If specified, the details of the trained model will be saved to a file that can be re-loaded and used for subsequent predictions

``--load`` : If specified, skips the training step and instead loads a previously saved model

``--overwrite`` : Overwrite existing output

``--debug`` : Include debug output messages

### Programmatic interface

See the ``examples`` folder for code samples showing how to use the brain age module in a Python program

<!-- References -->

[paper-medrxiv-link]: https://www.medrxiv.org/content/10.1101/2024.07.22.24310790v1



