import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        train_data = []
        with open(self.train_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    id_, title, genre, description = line.strip().split(' ::: ')
                    train_data.append({
                        'id': id_,
                        'title': title,
                        'genre': genre,
                        'description': description
                    })
                except ValueError:
                    continue
                    
        train_df = pd.DataFrame(train_data)
        

        test_data = []
        with open(self.test_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    id_, title, description = line.strip().split(' ::: ')
                    test_data.append({
                        'id': id_,
                        'title': title,
                        'description': description
                    })
                except ValueError:
                    continue
                    
        test_df = pd.DataFrame(test_data)
        

        train_df['genre_encoded'] = self.label_encoder.fit_transform(train_df['genre'])
        
        return train_df, test_df
    
    def get_genre_mapping(self):
        return dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_))) 