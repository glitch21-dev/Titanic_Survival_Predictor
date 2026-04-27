Titanic Survival Predictor

This project builds a machine learning pipeline that predicts whether a passenger aboard the Titanic survived, based on personal and travel-related attributes from the dataset.
It covers the complete workflow from raw data to a trained and evaluated model.

The pipeline begins with data cleaning, where non-essential columns such as ticket numbers, passenger names and cabin identifiers are dropped.
Missing age values are filled in using the median age per passenger class, and the Embarked column is handled before being removed. Once the data is clean, feature engineering is applied to make it more useful for the model. 
This includes encoding gender as a binary value, calculating family size from sibling, spouse, and parent columns, flagging passengers who were travelling alone, and grouping fare and age values into bins.

The cleaned and engineered dataset is then split into training and testing sets. A MinMax scaler is applied to normalise the feature values before training. 
A K-Nearest Neighbors classifier is used as the model, and grid search with cross-validation is run across a range of hyperparameters including the number of neighbors, distance metric, and weighting strategy, to find the best performing configuration. 
The best model is then used to make predictions on the test set, and performance is reported as an accuracy percentage alongside a confusion matrix. The confusion matrix is visualised as a heatmap to make the results easier to interpret.

Tech Stack
- NumPy, for numerical operations and conditional column creation
- pandas, for data loading, cleaning, and manipulation
- scikit-learn, for model training, hyperparameter tuning, scaling, and evaluation
- Matplotlib and Seaborn, for visualising the confusion matrix

Images Below
<img width="1000" height="700" alt="Figure_1" src="https://github.com/user-attachments/assets/1fd8f5ae-08be-417b-85d8-863420c4c1c1" />
............................................................
<img width="1919" height="1079" alt="Screenshot 2026-04-26 233038" src="https://github.com/user-attachments/assets/f7f52b63-8e41-43cf-8ead-51a6e9fd9323" />
.......................................................
<img width="1441" height="465" alt="Screenshot 2026-04-26 233105" src="https://github.com/user-attachments/assets/0acb47d1-7520-4ddf-bafb-fec7be67c1c0" />



