
# Crop Prediction Model

This repository contains a crop prediction model implemented as a Flask web application. The application predicts the most suitable crop based on several environmental parameters such as Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, and rainfall.

## Table of Contents
- [Demo](#demo)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [File Structure](#file-structure)
- [Acknowledgements](#acknowledgements)
- [License](#license)



## Features
- User-friendly web interface for inputting environmental parameters
- Real-time crop prediction
- Trained with a RandomForestClassifier for high accuracy

## Requirements
- Python 3.x
- Flask
- Scikit-learn
- NumPy
- Pandas

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/crop-prediction-model.git
   cd crop-prediction-model
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

4. Ensure you have the model file `model.pickle` in the root directory of the project.

## Usage
1. Run the Flask application:
   ```sh
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000`.

3. Input the required environmental parameters and click the "Predict" button to get the predicted crop.

## Model Training
To train the model, follow these steps:

1. Load the dataset:
   ```python
   data = pd.read_csv("Crop_recommendation.csv")
   ```

2. Split the data into features and labels:
   ```python
   x = data.iloc[:,:-1]  # features
   y = data.iloc[:,-1]   # labels
   ```

3. Split the data into training and testing sets:
   ```python
   from sklearn.model_selection import train_test_split
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   ```

4. Train the RandomForestClassifier model:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier()
   model.fit(x_train, y_train)
   ```

5. Evaluate the model and save it:
   ```python
   from sklearn.metrics import accuracy_score
   predictions = model.predict(x_test)
   accuracy = accuracy_score(y_test, predictions)
   print("Accuracy:", accuracy)

   import pickle
   pickle.dump(model, open("model.pickle", "wb"))
   ```

## File Structure
```
crop-prediction-model/
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── model.pickle
├── Crop_recommendation.csv
└── requirements.txt
```

## Acknowledgements
- [Dataset Source](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- [Flask Documentation](https://flask.palletsprojects.com/)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
