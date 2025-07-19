import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

class AnimeDataLoader:
    def __init__(self, file_path : str, processed_file_path : str):
        self.file_path = file_path
        self.processed_file_path = processed_file_path
        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            logger.error(f"Error loading data from {self.file_path}: {e}")
            self.data = pd.DataFrame()
        
        
    def generate_processed_data(self):
        if self.data.empty:
            logger.warning("No data to process.")
            return
        
        try:
            # Example processing: dropping duplicates and NaN values
            data = self.data.drop_duplicates().dropna()
            data = data.drop('Other name', axis=1)
            processed_data = pd.DataFrame(columns=['id','combined_info','image_url'])
            
            for index, row in data.iterrows():
                combined_info = []
                for col in data.columns:
                    if col not in ['Image URL', 'anime_id']:
                        combined_info.append(f"{col}: {row[col]}")

                id = row['anime_id']
                image_url = row['Image URL']
                processed_data.loc[len(processed_data)] = [id,' | '.join(combined_info), image_url]


            processed_data.to_csv(self.processed_file_path, index=False)
            
            logger.info(f"Processed data saved to {self.processed_file_path}")

        except Exception as e:
            logger.error(f"Error processing data: {e}")

if __name__ == "__main__":
    AnimeData = AnimeDataLoader('data/anime-dataset-2023.csv', 'data/processed_animedataset.csv')
    AnimeData.generate_processed_data()