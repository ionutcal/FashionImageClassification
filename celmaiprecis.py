import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

class FashionMNISTAnalyzer:
    def __init__(self, train_path, test_path, submission_path):

        train_data = np.load(train_path)
        self.x_train = train_data['x_train']
        self.y_train = train_data['y_train']

        test_data = np.load(test_path)
        self.x_test = test_data["x_test"]
        
        self.submission = pd.read_csv(submission_path)
    
    def preprocess_data(self, subset_ratio=0.8, apply_pca=True, n_components=100):

        x_train, _, y_train, _ = train_test_split(
            self.x_train, self.y_train, 
            train_size=subset_ratio, 
            random_state=42
        )
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, 
            test_size=0.2, 
            random_state=42
        )
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(self.x_test)
        
        if apply_pca:
            pca = PCA(n_components=n_components, random_state=42)
            x_train_scaled = pca.fit_transform(x_train_scaled)
            x_val_scaled = pca.transform(x_val_scaled)
            x_test_scaled = pca.transform(x_test_scaled)
        
        return (
            x_train_scaled, x_val_scaled, 
            y_train, y_val, 
            x_test_scaled
        )
    
    def train_random_forest(self, x_train, y_train):

        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        rf_classifier.fit(x_train, y_train)
        return rf_classifier
    
    def evaluate_model(self, model, x_val, y_val):

        y_pred = model.predict(x_val)
        precision = precision_score(y_val, y_pred, average='macro')
        print(f"Precision: {precision:.4f}")
        return precision
    
    def generate_predictions(self, model, x_test):

        predictions = model.predict(x_test)
        self.submission['Label'] = predictions
        print("Updated sample_submission.csv with predictions.")
        return self.submission

def main():

    train_path = "train.npz"
    test_path = "test.npz"
    submission_path = "sample_submission.csv"

    analyzer = FashionMNISTAnalyzer(train_path, test_path, submission_path)
    
    x_train, x_val, y_train, y_val, x_test = analyzer.preprocess_data(
        subset_ratio=0.8, 
        apply_pca=True, 
        n_components=100
    )
    
    print("Training Random Forest:")
    rf_model = analyzer.train_random_forest(x_train, y_train)
    
    print("Evaluating Random Forest Model:")
    precision = analyzer.evaluate_model(rf_model, x_val, y_val)
    print(f"Final Precision Score: {precision:.4f}")
    
    submission = analyzer.generate_predictions(rf_model, x_test)
    submission.to_csv('final_submission.csv', index=False)

if __name__ == "__main__":
    main()
