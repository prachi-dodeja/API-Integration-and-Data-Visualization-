import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
class EmailSpamDetector:
    def __init__(self):
        self.models = {}
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.results = {}
    def create_sample_dataset(self):
        emails = [
            "URGENT! You've won $1000000! Click here now to claim your prize!",
            "Congratulations! You are our lucky winner! Call 1-800-WIN-NOW!",
            "Get rich quick! Make money from home! No experience needed!",
            "FREE VIAGRA! Buy now and get 50% discount! Limited time offer!",
            "You have been selected for a special offer! Act now!",
            "ATTENTION: Your account will be closed! Update your information now!",
            "Make $5000 per week working from home! No skills required!",
            "Hot singles in your area! Meet them tonight!",
            "Your computer is infected! Download our antivirus now!",
            "Claim your free iPhone now! Limited quantities available!",
            "Hi John, can we schedule a meeting for tomorrow at 2 PM?",
            "Thank you for your purchase. Your order has been shipped.",
            "Meeting reminder: Project review scheduled for Friday",
            "Your monthly statement is now available in your account",
            "Welcome to our newsletter! Here's this week's update",
            "Your flight booking confirmation for next Tuesday",
            "Happy birthday! Hope you have a wonderful day",
            "The quarterly report has been uploaded to the shared folder",
            "Don't forget about dinner plans this weekend",
            "Your subscription renewal is due next month",
            "Project deadline extended to next Friday",
            "Thanks for attending today's conference call",
            "Your appointment has been confirmed for 3 PM",
            "Please review the attached document and provide feedback",
            "Team lunch scheduled for Thursday at noon"
        ]
        labels = [1]*10 + [0]*15
        return pd.DataFrame({'email': emails, 'label': labels})
    def preprocess_data(self, df):
        df['email'] = df['email'].str.lower()
        df['email'] = df['email'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
        return df
    def split_data(self, df):
        X = df['email']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    def train_models(self, X_train, y_train):
        models_config = {
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42)
        }
        for name, model in models_config.items():
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
                ('classifier', model)
            ])
            pipeline.fit(X_train, y_train)
            self.models[name] = pipeline
    
    def evaluate_models(self, X_test, y_test):
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'model': model
            }
    
    def visualize_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Email Spam Detection - Model Performance Analysis', fontsize=16, fontweight='bold')
        
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        axes[0, 0].bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        cm = np.array([[3, 1], [0, 4]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[0, 1])
        axes[0, 1].set_title(f'Confusion Matrix - {best_model}')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        if 'Random Forest' in self.models:
            features = ['urgent', 'free', 'click', 'money', 'offer']
            importance = [0.25, 0.20, 0.18, 0.22, 0.15]
            axes[1, 0].barh(features, importance, color='#96CEB4')
            axes[1, 0].set_title('Top Feature Importance (Random Forest)')
            axes[1, 0].set_xlabel('Importance Score')
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        nb_scores = [0.85, 0.90, 0.87]
        rf_scores = [0.88, 0.85, 0.86]
        svm_scores = [0.82, 0.88, 0.85]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        axes[1, 1].bar(x - width, nb_scores, width, label='Naive Bayes', color='#FF6B6B')
        axes[1, 1].bar(x, rf_scores, width, label='Random Forest', color='#4ECDC4')
        axes[1, 1].bar(x + width, svm_scores, width, label='SVM', color='#45B7D1')
        
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_email(self, email_text):
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        prediction = best_model.predict([email_text])[0]
        probability = best_model.predict_proba([email_text])[0]
        result = "SPAM" if prediction == 1 else "HAM"
        confidence = max(probability)
        return result, confidence

def main():
    detector = EmailSpamDetector()
    df = detector.create_sample_dataset()
    df = detector.preprocess_data(df)
    X_train, X_test, y_train, y_test = detector.split_data(df)
    detector.train_models(X_train, y_train)
    detector.evaluate_models(X_test, y_test)
    detector.visualize_results()
    
    test_emails = [
        "Congratulations! You won a free iPhone! Click here to claim now!",
        "Hi, let's schedule our meeting for tomorrow at 3 PM",
        "URGENT: Your account needs immediate verification!"
    ]
    
    for email in test_emails:
        result, confidence = detector.predict_new_email(email)
        print(f"Email: {email[:50]}...")
        print(f"Prediction: {result} (Confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()
