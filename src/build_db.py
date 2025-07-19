from data_loader import AnimeDataLoader
import pandas as pd
from config.database import index 
from src.generate_embeddings import AnimeEmbeddings

from dotenv import load_dotenv
import numpy as np
load_dotenv()


class AnimeDatabaseBuilder:
    def __init__(self, data_path: str, processed_file_path: str):
        self.data_path = data_path
        self.processed_file_path = processed_file_path
        self.model = AnimeEmbeddings()

    def build_database(self):
        try:
            data_loader = AnimeDataLoader(self.data_path, self.processed_file_path)
            data_loader.generate_processed_data()
            processed_data = pd.read_csv(self.processed_file_path)
            
            if not processed_data.empty:
                for _, row in processed_data.iterrows():
                    description = row['combined_info']
                    embedding = self.model.embeddings.embed_documents([description])[0]
                    upsert_data = [(str(row['id']), embedding, {'description': description, 'image_url': row['image_url']})]
                    index.upsert(
                        vectors=upsert_data,
                    )
                print("Database built successfully.")
            else:
                print("No data to insert into the database.")
        except Exception as e:
            print(f"Error building database: {e}")

if __name__ == "__main__":
    data_path = 'data/anime-dataset-2023.csv'
    processed_file_path = 'data/processed_animedataset.csv'
    builder = AnimeDatabaseBuilder(data_path, processed_file_path)
    builder.build_database()