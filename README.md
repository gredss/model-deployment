# Index
## 1. Hotel Booking Status Classifier – Project Summary
[`Click here to see the details`](streamlit/hotel-booking-status-binary-classification)

The Hotel Booking Status Classifier is a machine learning project designed to predict whether a hotel reservation will be honored or canceled. Utilizing structured booking data and the XGBoost algorithm, this model provides reliable predictions to assist hotel management in optimizing operational planning and reducing revenue loss due to cancellations.

### 📌 Key Objectives
- Develop a predictive model capable of classifying booking status using features such as lead time, market segment, customer type, and seasonal factors.
- Provide user-friendly access via a Streamlit web application, enabling real-time predictions based on user-uploaded data.

### 🔍 Core Features
- Comprehensive data preprocessing, including imputation, encoding, and feature scaling
- Feature engineering and selection to enhance model performance and generalizability
- Deployment of an optimized XGBoost classifier, evaluated with metrics such as accuracy, precision, recall, and F1-score
- Structured, object-oriented codebase for scalability and deployment
- Interactive visualization of model results through an intuitive web interface

### 🚀 Live Demo
Try the interactive app here:
👉 https://reservation-flow.streamlit.app/

## 2. Netflix Movie Recommendation
## 3. Obesity Predictor
[`Click here to see the details`](streamlit/netflix-hybrid-recommender)

The Netflix Title Recommender is a hybrid web-based application that delivers intelligent, content-based title recommendations using FastAPI and Streamlit. At its core, it employs TF-IDF vectorization, dimensionality reduction, and cosine-based similarity to match titles based on metadata such as description, genres, cast, and director. The system supports robust fallback behavior for unseen (cold-start) queries and is optimized for real-time response via a production-ready FastAPI backend.

### 📌 Key Objectives
- Develop a personalized content-based filtering model that recommends Netflix titles using natural language metadata.
- Expose the recommendation engine as a scalable RESTful API using FastAPI, enabling stateless, low-latency inference.
- Build an intuitive Streamlit frontend that interacts with the API for end-user accessibility.

### 🔍 Core Features
- Advanced natural language preprocessing: normalization, tokenization, POS tagging, lemmatization, and stopword removal
- Weighted multi-field TF-IDF vectorization over description, genres, director, cast, and country
- Dimensionality reduction with Truncated SVD to ensure computational efficiency
- Scalable similarity search using NearestNeighbors, with fallback genre-based recommendations
- Fully modular backend exposed via FastAPI endpoints for top-N recommendations
- Serialized components (.pkl) for deployable model persistence
- Lightweight Streamlit interface that connects to the FastAPI service, allowing real-time title queries and visual recommendations

### ⚙️ Deployment Architecture
The system follows a clean separation between backend and frontend:
- FastAPI: hosts the core /recommend API endpoint, handling vector retrieval and similarity inference.
- Streamlit: consumes the API and displays structured recommendation results to users via an interactive web UI.
- Pickle: used to load all pre-trained models and vectorizers in a stateless server process.

📄 Final Report
Access the full methodology and system design [here](https://github.com/your-username/model-deployment/blob/main/streamlit/netflix-hybrid-recommender/final-report.pdf).

🚀 Live Demo
Try the deployed application here: 👉 Netflix Recommender Web App


