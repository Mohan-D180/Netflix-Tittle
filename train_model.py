"""Train the final model (tuned Gradient Boosting from the lab) and save model.joblib."""
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from common import *

raw = pd.read_csv('netflix_titles.csv')
d = build_features(raw)
d['maturity'] = raw['rating'].map(RATING_MAP)
d = d.dropna(subset=['maturity']).drop_duplicates(subset=['title', 'type', 'release_year']).reset_index(drop=True)

# 1) sanity check: same 80/20 split as the lab report
Xtr, Xte, ytr, yte = train_test_split(d[FEATURES], d['maturity'], test_size=0.2, stratify=d['maturity'], random_state=42)
m = make_pipeline().fit(Xtr, ytr); p = m.predict(Xte)
print(f'Hold-out accuracy {accuracy_score(yte, p):.3f} | macro-F1 {f1_score(yte, p, average="macro"):.3f}')

# 2) deployment model: refit on all rows
final = make_pipeline().fit(d[FEATURES], d['maturity'])
genres = sorted(set(g for s in raw['listed_in'].dropna() for g in s.split(', ')))
countries = d['primary_country'].value_counts()
countries = [c for c in countries.index if c != 'Unknown' and countries[c] >= 10]
joblib.dump({'model': final, 'genres': genres, 'countries': countries,
             'metrics': {'accuracy': float(accuracy_score(yte, p)), 'macro_f1': float(f1_score(yte, p, average='macro'))}}, 'model.joblib')
print('Saved model.joblib |', len(genres), 'genres |', len(countries), 'countries')
