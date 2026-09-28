"""Train TF-IDF + Linear SVM without test-set leakage."""
from pathlib import Path
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import clean_text
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'dataset'/'clean_reviews_v2.csv'; MODELS=ROOT/'models'; MODELS.mkdir(exist_ok=True)
df=pd.read_csv(DATA); required={'Clean_Review','Sentiment'}; missing=required-set(df.columns)
if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
X=df['Clean_Review'].fillna('').astype(str).map(clean_text); y=df['Sentiment'].astype(str).str.strip(); mask=X.str.len().gt(0)&y.ne(''); X=X[mask]; y=y[mask]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
tfidf=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,sublinear_tf=True); Xtrv=tfidf.fit_transform(Xtr); Xtev=tfidf.transform(Xte)
model=LinearSVC(class_weight='balanced',random_state=42); model.fit(Xtrv,ytr); pred=model.predict(Xtev)
print(f'Accuracy: {accuracy_score(yte,pred):.4f}'); print(classification_report(yte,pred)); joblib.dump(model,MODELS/'sentiment_svm_model.pkl'); joblib.dump(tfidf,MODELS/'tfidf_vectorizer.pkl'); print('Model and vectorizer saved.')
