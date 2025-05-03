# CSE 572 - Data Mining Final Project

## Disaster and Risk Index Prediction

### Project Structure

- `data/`  
  Create a folder called `noaa_disaster_data` inside the `data` folder.  
  Untar the NOAA dataset into the `noaa_disaster_data` folder, which can be downloaded from:  
  [NOAA Disaster Data](https://drive.google.com/file/d/133KGc7zex7fdMBO-pka8oneaiLTQ0it-/view?usp=sharing).

- `src/`

  - `data_processing.py`  
    The Data Preprocessing script is located under `src/data_processing.py`.  
    You can import the `load_and_merge_disaster_data()` function to load the pre-processed data, which is in the form of a dictionary.

  - `predict.py`  
    Alternatively, there is a script called `predict.py` under `src/`.  
    This script performs the same function as the notebook, but it saves the model predictions in `.pkl` and `.json` formats.

- `notebooks/`

  - `models_prediction.py`  
    This notebook contains different models used for prediction, as stated in the project report.  
    It outputs a predicted incidents CSV file, which is saved under the `results` folder.

- `results/`  
  The output of the predicted incidents CSV file is placed here.

- `streamlit_app.py`  
  This file is for visualization and is hosted on Streamlit. It can be accessed here:  
  [Disaster and Risk Index Prediction](https://disaster-and-risk-index-prediction.streamlit.app/)

## Setup Instructions

1. **Download the NOAA Disaster Data**  
   Download and untar the dataset into the `data/noaa_disaster_data` folder.

2. **Install Dependencies**  
   Ensure you have the required dependencies installed for the scripts and app to run.

3. **Run the Streamlit App**  
   To visualize the disaster risk predictions, run the `streamlit_app.py` file.  
   Access the hosted app via: [Streamlit App](https://disaster-and-risk-index-prediction.streamlit.app/).

## Files Overview

- `data_processing.py`: Contains preprocessing functions for loading and merging disaster data.
- `models_prediction.py`: A Jupyter notebook with models for prediction and output generation.
- `predict.py`: A script that generates model predictions and stores them in `.pkl` and `.json` formats.
- `streamlit_app.py`: A Streamlit app for visualization of results.

## Notes

- Ensure all the paths are correctly set up when running the scripts.
- The model outputs and visualization will be updated periodically, so always check for the latest versions of the predictions.
