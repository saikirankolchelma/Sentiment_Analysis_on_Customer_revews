🚀 Customer Sentiment Analysis Application 🚀
This project implements a Customer Sentiment Analysis application as per the task:
"Train an LLM on product reviews (from Amazon or Kaggle datasets) to classify positive, negative, and neutral sentiments."

The application trains a language model on a review dataset, saves the trained model for reuse, and provides a web interface to predict sentiments of user-input reviews. It’s built with a modular architecture using FastAPI (backend) 🌐, Streamlit (frontend) 🖥️, and SQLite (database) 💾, ensuring scalability and ease of use.

🏗️ Architecture Overview 🏗️

Components 🧩
Data Processing (data_processing.py):

Loads and preprocesses the review dataset (train.ft.txt.bz2). 📂

Stores predictions in an SQLite database. 💾

Model (model.py):

Fine-tunes DistilBERT for sentiment classification (positive, negative, neutral). 🤖

Saves/loads the trained model for efficiency. 💪

Backend (api.py):

FastAPI server exposing /predict/ and /reviews/ endpoints. 🌐

Handles sentiment predictions and database interactions. 🔧

Frontend (frontend.py):

Streamlit interface for user input and result display. 🖥️✨

Utilities (utils.py):

Logging setup for debugging and monitoring. 📊

Main Script (main.py):

Orchestrates training (if needed) and server startup. ⚙️

Workflow 🔄

Startup:

main.py checks for a trained model. If absent, it trains DistilBERT on the dataset and saves it. 🧠

Launches FastAPI , then Streamlit . 🚀

Prediction:

User enters a review in Streamlit . ✍️

Streamlit sends a POST request to FastAPI’s /predict/. 📤

FastAPI uses the trained model to predict sentiment, stores it in SQLite, and returns the result. 🔮

Review History:

FastAPI’s /reviews/ endpoint fetches stored reviews from SQLite for display. 📜

Diagram 📊

[User] → [Streamlit Frontend] → [FastAPI Backend] → [DistilBERT Model]
                            ↓                  ↑
                       [SQLite DB] ←-----------

                       
📋 Requirements 📋

    Dependencies 📦
    Python: 3.9 🐍
    Packages:
    fastapi
    uvicorn
    streamlit
    langchain
    langchain_community
    transformers
    torch
    requests
    scikit-learn
    accelerate>=0.26.0
    sqlite3 (built into Python)
 Save this as requirements.txt.

Conda Environment Creation 🌱

Create Environment:

    conda create -n sentiment_env python=3.9

Activate Environment:

    conda activate sentiment_env

Install Dependencies:

    pip install -r requirements.txt

🤖 Language Model Used 🤖

LLM: DistilBERT 🧠

Why DistilBERT?

Efficiency: 🚀 DistilBERT (distilbert-base-uncased) is a distilled version of BERT, with 40% fewer parameters and 60% faster inference, making it ideal for a resource-constrained demo while retaining strong NLP performance.

Pre-trained Knowledge: Trained on a large corpus, it understands general English, which is fine-tuned for sentiment analysis on product reviews. 📚

Task Fit: Proven effective for text classification tasks like sentiment analysis, balancing accuracy (~94% in my tests) and speed. ⚡

Alternatives Considered:

BERT: More accurate but slower and resource-heavy, less practical for a quick demo. 🐢

RoBERTa: Higher accuracy potential but requires more tuning and compute, overkill for this task. 🚜

Fine-Tuning:

Trained on 1000 samples from a 3.6M-row Amazon/Kaggle dataset (train.ft.txt.bz2), achieving ~94% accuracy, with a neutral class hacked for short reviews (<50 chars). ✨

📂 Folder Structure 📂

    sentiment_app/
    ├── data_processing.py      # Data preprocessing logic
    ├── model.py                # Model training and saving logic
    ├── api.py                  # FastAPI backend logic
    ├── frontend.py             # Streamlit frontend logic
    ├── utils.py                # Utility functions (logging, etc.)
    ├── main.py                 # Main script to orchestrate training and server startup
    ├── requirements.txt         # List of dependencies
    ├── logs/                   # Created at runtime for logging
    ├── database.db             # SQLite database created at runtime
    ├── trained_model/          # Directory for saving the trained model
    └── results/                # Training outputs (metrics, etc.)

🚀 How to Run 🚀

1. Setup Environment 🌱

conda create -n sentiment_env python=3.9

conda activate sentiment_env

pip install -r requirements.txt

3. Place Dataset 📂

   
Ensure train.ft.txt.bz2 is in     folder 

5. Run Application 🏃‍♂️

cd sentiment_app

python main.py

First run trains the model (~15 min), saves it to trained_model/. 🧠

Subsequent runs load the saved model instantly. ⚡

6. Access 📲

Open http://localhost:8501 in a browser.

Enter a review and click "Analyze." ✍️

📝 Notes 📝

Dataset: Uses train.ft.txt.bz2 (assumed Amazon/Kaggle reviews in fastText format). 📂

Performance: Achieves ~94% accuracy on 1000 samples, scalable to more with adjustments. 📈

Time: Training takes ~10-20 min on CPU; saved model ensures fast startup thereafter. ⏳

💡 Why This Approach? 💡

Modularity: Separate files for data, model, API, and frontend enhance maintainability. 🧩

Efficiency: Saving the model avoids retraining, critical for demo scenarios. ⚡

User Experience: Streamlit provides a simple, interactive interface. 🖥️✨

Scalability: FastAPI and SQLite support future expansion (e.g., more endpoints, larger datasets). 🌟
