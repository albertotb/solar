## Info URLs

* Data:     https://data.nrel.gov/submissions/11
* Info:     https://midcdmz.nrel.gov/oahu_archive/
* Location: https://midcdmz.nrel.gov/oahu_archive/instruments.html
* Map:      https://midcdmz.nrel.gov/oahu_archive/map.jpg


## Models

Here we report the list of models currently beeing tested

* ElasticNet: linear regression plus L1 and L2 regularization.
* Persistence: simple baseline that predicts using the current observation.
* Convolutional (1D): given an ordering of the sensors (by geographical longitude, for instance), applies a
    series of 1D convolutions/locally connected layers to extract correlations between neighboring sensors.
    
They can be found inside the notebooks folder.


## Competition

* Use as data oahu_min.feather.
* Test-set: last four months (Ago-Nov), no retraining from start of test set.

## Dependencies

Create conda environment with:

    conda env create -f environment.yml
