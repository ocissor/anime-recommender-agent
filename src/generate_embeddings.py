from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

class AnimeEmbeddings:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    def generate_embeddings(self, texts: list[str]):
        return self.embeddings.embed_documents(texts)
    


