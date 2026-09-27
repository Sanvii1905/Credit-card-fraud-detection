import os,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix
from src.preprocessing import load_data,prepare_data
df=load_data("data/creditcard_sample.csv")
Xtr,Xte,ytr,yte,scaler=prepare_data(df)
model=LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42)
model.fit(Xtr,ytr); pred=model.predict(Xte)
print("=== Credit Card Fraud Detection ===")
print("Dataset size:",len(df)); print("Fraud transactions:",int(df.Class.sum()))
print("Accuracy :",f"{accuracy_score(yte,pred):.4f}")
print("Precision:",f"{precision_score(yte,pred,zero_division=0):.4f}")
print("Recall   :",f"{recall_score(yte,pred,zero_division=0):.4f}")
print("F1 Score :",f"{f1_score(yte,pred,zero_division=0):.4f}")
print("\nClassification Report:\n",classification_report(yte,pred,zero_division=0))
print("Confusion Matrix:\n",confusion_matrix(yte,pred))
os.makedirs("models",exist_ok=True)
joblib.dump(model,"models/fraud_model.joblib"); joblib.dump(scaler,"models/scaler.joblib")
