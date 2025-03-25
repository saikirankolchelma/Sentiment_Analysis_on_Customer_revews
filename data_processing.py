import sqlite3
import os
import bz2
import pandas as pd
from utils import setup_logging

logger = setup_logging()

class DataProcessor:
    def __init__(self, db_path="database.db", data_dir="C:/Users/ksaik/OneDrive/Desktop/customer_st_analaysis"):
        self.db_path = db_path
        self.data_dir = data_dir
        self.train_txt = os.path.join(data_dir, "train.txt")
        self._setup_database()

    def _setup_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_text TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL
                )
            ''')
            conn.commit()
            conn.close()
            logger.info("Database initialized.")
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise

    def store_review(self, review_text, sentiment, confidence):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO reviews (review_text, sentiment, confidence) VALUES (?, ?, ?)',
                          (review_text, sentiment, confidence))
            conn.commit()
            conn.close()
            logger.info(f"Stored review: '{review_text}' - {sentiment}")
        except Exception as e:
            logger.error(f"Error storing review: {e}")
            raise

    def get_all_reviews(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reviews")
            rows = cursor.fetchall()
            conn.close()
            logger.info("Fetched all reviews from database.")
            return rows
        except Exception as e:
            logger.error(f"Error fetching reviews: {e}")
            raise

    def decompress_bz2(self, bz2_path, output_path):
        try:
            with bz2.open(bz2_path, 'rt', encoding='utf-8') as bz_file:
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    out_file.writelines(bz_file)
            logger.info(f"Decompressed {bz2_path} to {output_path}")
        except Exception as e:
            logger.error(f"Error decompressing {bz2_path}: {e}")
            raise

    def parse_fasttext(self, file_path):
        try:
            reviews = []
            sentiments = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('__label__'):
                        label, text = line.split(' ', 1)
                        label = label.replace('__label__', '')
                        reviews.append(text.strip())
                        sentiment = "negative" if label == "1" else "positive"  # Fixed mapping
                        sentiments.append(sentiment)
            df = pd.DataFrame({"text": reviews, "sentiment": sentiments})
            df.loc[df["text"].str.len() < 50, "sentiment"] = "neutral"  # Neutral hack
            logger.info(f"Parsed {file_path} into DataFrame with {len(df)} rows")
            return df.sample(n=1000, random_state=42)
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            raise

    def prepare_training_data(self):
        train_bz2 = os.path.join(self.data_dir, "train.ft.txt.bz2")
        if not os.path.exists(self.train_txt):
            self.decompress_bz2(train_bz2, self.train_txt)
        return self.parse_fasttext(self.train_txt)