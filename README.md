## Info URLs

* Data:     https://data.nrel.gov/submissions/11
* Info:     https://midcdmz.nrel.gov/oahu_archive/
* Location: https://midcdmz.nrel.gov/oahu_archive/instruments.html
* Map:      https://midcdmz.nrel.gov/oahu_archive/map.jpg

## Problems
 1. Given some sensors in certain locations at times t-k, ..., t, predict the output of the sensors at time t+1
    * Only interested in predicting accurately some of them, called control (inner) sensors
    * Rest of the sensors are called proxy (outer) sensors
    * **Problem:** sensors fail and thus data will be missing for certain timesteps
        * Input missing data
        * Transform sensor data into a rectangular grid using the available information at that timestep (learning the spacial structure)
 2. Given a set of N control and proxy sensors and a budget B, decide where to place B new sensors to maximize the predicibility of the control (inner) sensors   


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

* Models inside `/models/` and `/results`:
    * GPconstant_LocCon2D_LSTM_periods9: Use GP as input X. Time distributed Locally Connected 2D and then LSTM. Use 9 periods of time (previous 9 minutes) as feature
    * GPtorch_LocCon2D_LSTM_periods9: Same, but using GP trained in PyTorch.
    * GPconstant_LocCon2D_Dense_periods3: Train with GP as input X. LocCon2D using timestamps as channels. Then Dense Layer. Here the number of parameter scales with periods.

## Dependencies

Create conda environment with:

    conda env create -f environment.yml
