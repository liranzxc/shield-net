import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset

class SimpleRAG:
    def __init__(self):
        """
        Initialize the RAG system with a news API key and SentenceTransformer model.
        """
        self.model = SentenceTransformer('all-MiniLM-L12-v2')
        self.articles = []
        self.embeddings = None
        self.csv_path = "./api/data/labeled_data.csv"

        self.fetch_daily_news()
        self.encode_articles()

    def fetch_daily_news(self):
        """
        Mock function to fetch daily news with a mix of good and bad content.
        """
        good_news = [
            "AI revolutionizes healthcare. Researchers are leveraging artificial intelligence to diagnose diseases faster and more accurately.",
            "SpaceX successfully launches Starship. The company's latest mission marks a significant step toward Mars colonization.",
            "Breakthrough in renewable energy. Scientists discover a new method to efficiently store solar energy.",
        ]

        df = pd.read_csv(self.csv_path)
        bad_tweet = df["tweet"].to_list()[:100]
        # Combine good and bad news and shuffle the list
        self.articles = good_news + bad_tweet

    def encode_articles(self):
        """
        Encode the fetched news articles into embeddings using SentenceTransformer and datasets library.
        """
        if not self.articles:
            print("No articles to encode.")
            return

        # Create a Dataset object
        dataset = Dataset.from_dict({"text": self.articles})

        # Encode articles in batches using map
        def embed_batch(batch):
            return {"embeddings": self.model.encode(batch["text"])}

        dataset = dataset.map(embed_batch, batched=True, batch_size=64)

        # Store the embeddings
        self.embeddings = dataset["embeddings"]
        print("Encoded articles into embeddings.")

    def get_context(self, prompt: str, top_k: int = 3, threshold: float = 0.3):
        """
        Retrieve the most relevant articles for a given prompt based on cosine similarity.

        Args:
            prompt (str): The user query or prompt.
            top_k (int): The number of relevant articles to retrieve.
            threshold (float): The minimum similarity score to consider an article relevant.

        Returns:
            List[str]: The most relevant articles or an empty list if no articles meet the threshold.
        """
        if self.embeddings is None:
            print("No embeddings available. Please fetch and encode articles first.")
            return ""

        prompt_embedding = self.model.encode([prompt])
        similarities = cosine_similarity(prompt_embedding, self.embeddings)[0]
        print(np.max(similarities),self.articles[np.argmax(similarities)])
        # Filter based on threshold
        valid_indices = [idx for idx, sim in enumerate(similarities) if sim >= threshold]

        if not valid_indices:
            print(f"No articles found with similarity >= {threshold}")
            return ""

        # Sort the valid indices by similarity and pick the top_k
        valid_indices = sorted(valid_indices, key=lambda idx: similarities[idx], reverse=True)[:top_k]
        relevant_articles = [self.articles[idx] for idx in valid_indices]
        return "\n".join(relevant_articles)
