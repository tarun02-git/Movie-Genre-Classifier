from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class GenreClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y, label_encoder):
       
        y_pred = self.predict(X)
        
    
        report = classification_report(y, y_pred, target_names=label_encoder.classes_)
        
   
        cm = confusion_matrix(y, y_pred)
        
      
        plt.figure(figsize=(15, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
      
        feature_importance = self.model.feature_importances_
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
    
    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv)
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        } 