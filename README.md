# Movie-Genre-Classifier
Made Movie Genre Classifier using Random forest algorithm (Machine Learning) and Classifying the dataset and predicting the Model.
# Movie Genre Classification

This project implements a machine learning model to classify movies into genres based on their plot descriptions. The model uses natural language processing techniques and a Random Forest classifier to predict movie genres.

## Project Structure

- `data_loader.py`: Handles loading and preprocessing of the movie dataset
- `text_processor.py`: Implements text preprocessing and feature extraction
- `model.py`: Contains the genre classification model implementation
- `main.py`: Main script to run the complete classification pipeline
- `requirements.txt`: List of Python dependencies

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your data files are in the correct location:
   - `train_data.txt`: Training data file
   - `test_data.txt`: Test data file

2. Run the classification:
```bash
python main.py
```

## Output

The script will generate several output files:
1. `confusion_matrix.png`: Visualization of model performance across different genres
2. `feature_importance.png`: Plot showing the most important features for classification
3. `predictions.txt`: Predicted genres for the test dataset

## Model Details

- Text Processing:
  - Text cleaning and normalization
  - Stop word removal
  - Lemmatization
  - TF-IDF vectorization

- Classification:
  - Random Forest Classifier
  - Cross-validation for model evaluation
  - Feature importance analysis

## Data Format

- Training data format: `ID ::: TITLE ::: GENRE ::: DESCRIPTION`
- Test data format: `ID ::: TITLE ::: DESCRIPTION` 
