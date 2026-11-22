"""Shared feature engineering + pipeline for training and for the Streamlit app."""
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier

CLASSES = ['Kids', 'Teens', 'Mature']
RATING_MAP = {**dict.fromkeys(['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'G'], 'Kids'),
              **dict.fromkeys(['TV-PG', 'PG', 'TV-14', 'PG-13'], 'Teens'),
              **dict.fromkeys(['TV-MA', 'R', 'NC-17'], 'Mature')}
NUM_COLS = ['type_movie', 'release_year', 'year_added', 'month_added', 'years_to_netflix', 'duration_min',
            'n_seasons', 'has_director', 'n_cast', 'n_countries', 'n_genres', 'desc_words', 'title_len']
FEATURES = NUM_COLS + ['primary_country', 'genres', 'description']


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Same engineering as the notebook (row-wise, no leakage)."""
    d = raw.copy()
    d['type_movie'] = (d['type'] == 'Movie').astype(int)
    dur = d['duration'].astype(str).str.extract(r'(\d+)')[0].astype(float)
    d['duration_min'] = np.where(d['type'] == 'Movie', dur, 0)
    d['n_seasons'] = np.where(d['type'] == 'TV Show', dur, 0)
    added = pd.to_datetime(d['date_added'].astype(str).str.strip(), errors='coerce')
    d['year_added'], d['month_added'] = added.dt.year, added.dt.month
    d['years_to_netflix'] = (d['year_added'] - d['release_year']).clip(lower=0)
    d['has_director'] = d['director'].notna().astype(int)
    d['n_cast'] = d['cast'].fillna('').apply(lambda s: len(s.split(',')) if s else 0)
    d['n_countries'] = d['country'].fillna('').apply(lambda s: len([c for c in s.split(',') if c.strip()]))
    d['primary_country'] = d['country'].fillna('Unknown').str.split(',').str[0].str.strip().replace('', 'Unknown')
    d['genres'] = d['listed_in'].str.replace(', ', '|').str.replace(' ', '_')
    d['n_genres'] = d['listed_in'].str.count(',') + 1
    d['desc_words'] = d['description'].str.split().str.len()
    d['title_len'] = d['title'].str.len()
    return d


def make_pipeline(random_state=42):
    prep = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUM_COLS),
        ('country', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=30, sparse_output=False), ['primary_country']),
        ('genres', CountVectorizer(token_pattern=r'[^|]+', lowercase=False, binary=True, min_df=20), 'genres'),
        ('text', Pipeline([('tfidf', TfidfVectorizer(stop_words='english', min_df=3, max_features=3000, sublinear_tf=True)),
                           ('svd', TruncatedSVD(100, random_state=random_state))]), 'description'),
    ], sparse_threshold=0)
    clf = HistGradientBoostingClassifier(learning_rate=0.06, max_depth=4, max_iter=300,
                                         class_weight='balanced', random_state=random_state)
    return Pipeline([('prep', prep), ('clf', clf)])
