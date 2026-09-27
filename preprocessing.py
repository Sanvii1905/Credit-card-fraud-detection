import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
FEATURES=["Time"]+[f"V{i}" for i in range(1,29)]+["Amount"]
def load_data(path): return pd.read_csv(path)
def prepare_data(df,test_size=0.2,random_state=42):
    X=df[FEATURES].copy(); y=df["Class"].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)
    s=StandardScaler()
    return s.fit_transform(Xtr),s.transform(Xte),ytr,yte,s
