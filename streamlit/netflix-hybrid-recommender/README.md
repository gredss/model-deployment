# Content-Based Filtering for Netflix Title Recommendations
## Project Overview

This project implements a content-based recommendation system for Netflix titles using TF-IDF vectorization and cosine similarity. The model generates recommendations by comparing textual metadata such as description, genres, cast, and director. Dimensionality reduction is applied for scalability, and a fallback strategy is designed for handling cold-start queries.

A comprehensive report detailing the methodology and evaluation results is available [here](https://github.com/your-username/model-deployment/blob/main/streamlit/netflix-hybrid-recommender/final-report.pdf)


---

## Step 1: Library Imports and Environment Setup

The following libraries are required for text preprocessing, feature extraction, and model building:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from collections import defaultdict
import pickle
import nltk
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import warnings
warnings.filterwarnings('ignore')
````

Download required NLTK corpora:

```python
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
```

---

## Step 2: Dataset Loading and Cleaning

The dataset is based on the `netflix_titles.csv` file. The `listed_in` column is renamed to `genres`, and missing values in key columns are imputed with the string `'unknown'`.

```python
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={'listed_in': 'genres'}, inplace=True)
    text_cols = ['description', 'director', 'cast', 'country', 'genres']
    df[text_cols] = df[text_cols].fillna('unknown')
    df['rating'] = df['rating'].fillna('unrated')
    
    for col in text_cols:
        df[col] = df[col].apply(clean_text)
    return df
```

---

## Step 3: Text Normalization and Lemmatization

A custom cleaning pipeline is defined to normalize, tokenize, lemmatize, and remove stopwords using NLTK utilities. This is crucial for transforming free-text fields into a structured format suitable for vectorization.

```python
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    tag = tag[0].upper() if tag else ''
    tag_dict = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(lemmatized)
```

---

## Step 4: Feature Extraction and Weighting

TF-IDF vectorization is applied to key metadata fields. Different fields are weighted based on their relevance to content similarity.

```python
desc_vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
gen_vectorizer  = TfidfVectorizer(max_features=500)
dir_vectorizer  = TfidfVectorizer(max_features=300)
cast_vectorizer = TfidfVectorizer(max_features=500)
ctry_vectorizer = TfidfVectorizer(max_features=200)

desc_vec = desc_vectorizer.fit_transform(df['description'])
gen_vec  = gen_vectorizer.fit_transform(df['genres'])
dir_vec  = dir_vectorizer.fit_transform(df['director'])
cast_vec = cast_vectorizer.fit_transform(df['cast'])
ctry_vec = ctry_vectorizer.fit_transform(df['country'])

combined_features = hstack([
    1.0 * desc_vec,
    0.8 * gen_vec,
    0.5 * dir_vec,
    0.4 * cast_vec,
    0.2 * ctry_vec
])
```

---

## Step 5: Dimensionality Reduction and Normalization

To reduce computation and storage costs, the high-dimensional TF-IDF vectors are compressed using Truncated SVD, followed by L2 normalization.

```python
svd = TruncatedSVD(n_components=200, random_state=42)
reduced_features = svd.fit_transform(combined_features)
norm_features = normalize(reduced_features)
```

---

## Step 6: Similarity Search and Indexing

We attempt to use `NearestNeighbors` for efficient similarity search. If the library is not available, we fall back to cosine similarity.

```python
try:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    nn.fit(norm_features)
    USE_ANN = True
except ImportError:
    USE_ANN = False
    cosine_sim = cosine_similarity(norm_features)
```

To support multiple entries with the same title, a dictionary is used to map each lowercase title to its index (or indices):

```python
title_to_indices = defaultdict(list)
for idx, title in enumerate(df['title']):
    title_to_indices[title.lower()].append(idx)
```

---

## Step 7: Recommendation Function

A robust `recommend()` function is defined. It provides top-N content-based recommendations, with fallback behavior for titles not found in the dataset.

```python
def recommend(title, top_n=5, fallback_to_popular=True):
    ...
```

The fallback strategy recommends popular recent titles based on genre similarity, enabling robust support for cold-start scenarios.

---

## Step 8: Model Serialization

All trained components are saved in serialized `.pkl` format to support deployment in web applications or API endpoints.

```python
with open("desc_vectorizer.pkl", "wb") as f:
    pickle.dump(desc_vectorizer, f)

# Repeat for all other components:
# gen_vectorizer.pkl, dir_vectorizer.pkl, cast_vectorizer.pkl, 
# ctry_vectorizer.pkl, svd_model.pkl, nn_model.pkl, 
# norm_features.pkl, title_to_indices.pkl, df_metadata.pkl
```

Only the metadata necessary for recommendation output is preserved (`title`, `director`, `cast`, etc.), excluding the full text vectors.

---

**Next Steps →**

The next phase will involve integrating this backend into a web application using `FastAPI` and building a user interface with `Streamlit`.
