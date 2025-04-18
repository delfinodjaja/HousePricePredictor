# House Price Prediction GUI

This Python GUI application predicts house prices using a neural network model trained on the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset from Kaggle.

## Files Included:

1. **HousePrices.ipynb**: Jupyter Notebook file containing data preprocessing and model training code.

2. **GUI.py**: Python script for the graphical user interface (GUI) that allows users to input house features and get a predicted house price.

3. **model.pt**: File containing the saved parameters of the trained neural network model.

4. **scaler_y.pkl and scaler.pkl**: Pickle files containing scaler objects used for consistent scaling among files.

## How to Use:

1. **Data Preprocessing and Model Training:**
   - Refer to the `HousePrices.ipynb` notebook for the data preprocessing steps and model training.
   - This notebook includes code to handle the Kaggle House Prices dataset.

2. **GUI Application:**
   - Run the `GUI.py` script to open the GUI for predicting house prices.
   - The GUI allows users to input various house features such as garage capacity, garage size, basement size, floor sizes, the number of bathrooms, and housing zone.
   - Click the "Predict House Price" button to get a predicted house price based on the trained model.

3. **Model Parameter and Scaler Files:**
   - The `model.pt` file contains the saved parameters of the trained neural network model.
   - The `scaler.pkl` and `scaler_y.pkl` files contain scaler objects used for consistent scaling. Make sure these files are loaded during the model prediction in the GUI.

## Requirements:

- Python 3.x
- Required Python packages:
  - `customtkinter`: A custom-themed tkinter library.
  - `torch`: PyTorch library for neural network operations.
  - `numpy`: NumPy library for numerical operations.
  - `joblib`: Joblib library for saving and loading scaler objects.
  - 'pandas': pandas for loading csv file.
  - 'sklearn': sklearn libary for scaling.

## Acknowledgments:

- The model in this project is trained on the Kaggle House Prices dataset.
- The GUI is built using the `customtkinter` library for a customized appearance.
- The model is built using 'pytorch'
- Data analysis and preprocessing using 'pandas', 'numpy', and 'sklearn'

Feel free to reach out for any questions or improvements!

