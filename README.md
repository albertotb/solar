## Info URLs

* Data:     https://data.nrel.gov/submissions/11
* Info:     https://midcdmz.nrel.gov/oahu_archive/
* Location: https://midcdmz.nrel.gov/oahu_archive/instruments.html
* Map:      https://midcdmz.nrel.gov/oahu_archive/map.jpg


## Preprocessing data

* Download data using `/src/utils/download_data.py` -> `oahu.feather`
* Resample data from 1s to 1m with `/src/utils/resample_data.py` -> `oahu_min.feather`
* Compute clearsky models with `/src/utils/compute_clearsky.py` -> `oahu_min_cs.pkl`:
     * [Pysolar](https://pysolar.readthedocs.io/en/latest/) library
     * [Pvlib](https://pvlib-python.readthedocs.io/en/latest/clearsky.html) library (Ineichen, Haurwitz, simplified Solis)
     * Formulas from UCM (Haurwitz, Kasten)
* Run clearsky notebook `/notebooks/clearsky.ipynb` to compare the clearsky models -> `oahu_min_final.pkl`:
     * Set negative GHI to 0
     * Filter only always-sunlight hours (7:30-17:30)
     * Drop AP3 location (faulty sensor)
     * Normalize the GHI with the best clearsky model to (hopefully) [0, 1] range

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
