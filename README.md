## Info URLs

* Data:     https://data.nrel.gov/submissions/11
* Info:     https://midcdmz.nrel.gov/oahu_archive/
* Location: https://midcdmz.nrel.gov/oahu_archive/instruments.html
* Map:      https://midcdmz.nrel.gov/oahu_archive/map.jpg


## Competition

* Download data using `/src/utils/download_data.py`
* Resample data from 1s to 1m with `/src/utils/resample_data.py`
* Test-set: last four months (Ago-Nov), no retraining from start of test set.

## Models

* Files with name `train_MODEL.py` in `/src/`, so far:
   * ElasticNet
   * DLMs

## Dependencies

Create conda environment with:

    conda env create -f environment.yml
