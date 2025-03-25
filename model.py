from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline, Trainer, TrainingArguments
from langchain_community.llms import HuggingFacePipeline
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import setup_logging
import os

logger = setup_logging()

class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased"):
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            if os.path.exists("trained_model"):
                self.model = DistilBertForSequenceClassification.from_pretrained("trained_model")
                logger.info("Loaded fine-tuned model from 'trained_model'.")
            else:
                self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
                logger.info(f"Initialized fresh {model_name} for training.")
            self.label_map = {"positive": 0, "negative": 1, "neutral": 2}
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def tokenize_function(self, texts):
        return self.tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    def prepare_dataset(self, df):
        try:
            encodings = self.tokenize_function(df["text"].tolist())
            labels = [self.label_map[sent] for sent in df["sentiment"]]
            class SentimentDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels
                def __getitem__(self, idx):
                    item = {key: val[idx] for key, val in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item
                def __len__(self):
                    return len(self.labels)
            return SentimentDataset(encodings, labels)
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise

    def train(self, df):
        try:
            train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
            train_dataset = self.prepare_dataset(train_df)
            eval_dataset = self.prepare_dataset(eval_df)
            
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=2,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=10,
                evaluation_strategy="epoch",
            )
            
            def compute_metrics(pred):
                labels = pred.label_ids
                preds = pred.predictions.argmax(-1)
                acc = accuracy_score(labels, preds)
                return {"accuracy": acc}

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            eval_results = trainer.evaluate()
            logger.info(f"Training completed. Eval results: {eval_results}")
            trainer.save_model("trained_model")
            logger.info("Saved fine-tuned model to 'trained_model'.")
            self.pipe = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=128,
                truncation=True,
                padding=True
            )
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            return eval_results
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def predict(self, review_text):
        try:
            inputs = self.tokenizer(review_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits.softmax(dim=-1)
                sentiment_idx = probs.argmax().item()
                confidence = probs.max().item()
            sentiment = {0: "positive", 1: "negative", 2: "neutral"}[sentiment_idx]
            logger.info(f"Predicted sentiment for '{review_text}': {sentiment} (Confidence: {confidence:.2f}, Probs: {probs.tolist()[0]})")
            return sentiment, confidence
        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            raise

if __name__ == "__main__":
    model = SentimentModel()
    review = "after getting it the cone broke away from the spider only after 10 min of getting it worst buy ever i dont have a amp powerful enough to blow this thing yet it broke and im being tolled i can get a refund or return for the poor put together sub iv gottin i dont advise anyone to waste there time or money here"
    sentiment, confidence = model.predict(review)
    print(f"Sentiment: {sentiment}, Confidence: {confidence}")