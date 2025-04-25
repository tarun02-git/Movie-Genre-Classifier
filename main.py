import os
import numpy as np
from data_loader import DataLoader
from text_processor import TextProcessor
from model import GenreClassifier
import matplotlib.pyplot as plt

def plot_feature_importance(feature_names, importance_scores, top_n=20):
 
    top_indices = np.argsort(importance_scores)[-top_n:]
    
   
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importance_scores[top_indices])
    plt.yticks(range(top_n), feature_names[top_indices])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():

    data_loader = DataLoader('train_data.txt', 'test_data.txt')
    text_processor = TextProcessor(max_features=5000)
    classifier = GenreClassifier(n_estimators=200)
    
    print("Loading data...")
    train_df, test_df = data_loader.load_data()
    
    print("Processing text data...")
    X_train = text_processor.fit_transform(train_df['description'])
    X_test = text_processor.transform(test_df['description'])
    y_train = train_df['genre_encoded']
    
    print("Training model...")
    classifier.train(X_train, y_train)
    
    print("Performing cross-validation...")
    cv_results = classifier.cross_validate(X_train, y_train)
    print(f"Cross-validation scores: {cv_results['scores']}")
    print(f"Mean CV score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']*2:.4f})")
    
    print("\nEvaluating model on training data...")
    eval_results = classifier.evaluate(X_train, y_train, data_loader.label_encoder)
    print("\nClassification Report:")
    print(eval_results['classification_report'])
    

    feature_names = np.array(text_processor.get_feature_names())
    plot_feature_importance(feature_names, eval_results['feature_importance'])
    
    print("\nMaking predictions on test data...")
    test_predictions = classifier.predict(X_test)
    test_predictions_labels = data_loader.label_encoder.inverse_transform(test_predictions)
    

    print("Saving predictions...")
    with open('predictions.txt', 'w', encoding='utf-8') as f:
        for id_, genre in zip(test_df['id'], test_predictions_labels):
            f.write(f"{id_} ::: {genre}\n")
    
    print("\nDone! Results have been saved:")
    print("1. Confusion matrix: confusion_matrix.png")
    print("2. Feature importance plot: feature_importance.png")
    print("3. Predictions: predictions.txt")

if __name__ == "__main__":
    main() 