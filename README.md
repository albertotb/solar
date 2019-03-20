## Info URLs

* Data:     https://data.nrel.gov/submissions/11
* Info:     https://midcdmz.nrel.gov/oahu_archive/
* Location: https://midcdmz.nrel.gov/oahu_archive/instruments.html
* Map:      https://midcdmz.nrel.gov/oahu_archive/map.jpg


## Preprocessing data

* Download data using `/src/utils/download_data.py` -> `oahu.feather`
* Resample data from 1s to 1m with `/src/utils/resample_data.py` -> `oahu_min.feather`
* Compute clearsky models with `/src/utils/compute_clearsky.py` -> `oahu_min_cs.pkl`
* Run clearsky notebook `/notebooks/clearsky.ipynb` to comparte the clearsky models and use the best one to normalize the GHI -> `oahu_min_final.pkl`
* Test-set: last four months (Ago-Nov), no retraining from start of test set.

## Models

* Files with name `train_MODEL.py` in `/src/`, so far:
   * ElasticNet
   * DLMs
   * Convolutional (1D) network
   
* Notebooks inside `/notebooks/`:
    * ElasticNet: linear regression plus L1 and L2 regularization.
    * Persistence: simple baseline that predicts using the current observation.
    * Convolutional (1D): given an ordering of the sensors (by geographical longitude, for instance), applies a
        series of 1D convolutions/locally connected layers to extract correlations between neighboring sensors.


## Dependencies

Create conda environment with:

    conda env create -f environment.yml
