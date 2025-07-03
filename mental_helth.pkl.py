import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
data = {
    'anxiety': [0, 1, 2, 3, 1, 2, 3, 0],
    'depression': [0, 1, 2, 3, 1, 2, 3, 0],
    'sleep': [0, 1, 2, 3, 2, 2, 3, 0],
    'concentration': [0, 1, 2, 3, 2, 1, 3, 0],
    'risk_level': [0, 1, 2, 2, 1, 2, 2, 0]  # 0=Low, 1=Moderate, 2=High
}

df = pd.DataFrame(data)
X = df.drop('risk_level', axis=1)
y = df['risk_level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'mental_health_model.pkl')
