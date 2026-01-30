from transformers import AutoModel
from sentence_transformers import SentenceTransformer
import torch 

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu" 

def get_model(model_id, device=DEVICE): 
    return SentenceTransformer(model_id, trust_remote_code=True).to(device)

def embedd_texts(model, texts, task="text-matching", batch_size=32, show_progress_bar=False):
    return model.encode(texts, task=task, batch_size=batch_size, show_progress_bar=show_progress_bar)