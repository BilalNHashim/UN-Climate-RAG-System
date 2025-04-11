import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def pre_tokenising_process(text: str) -> str:
    # Remove punctuation, preserve accented characters
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = nltk.word_tokenize(text.lower())
    return text

def post_tokenisation_processing(tokens: list) -> list:
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

def get_embeddings(tokens, model):    
    # Get model output (without gradient calculation for efficiency)
    with torch.no_grad():
        model_output = model(**tokens)
    
    # Use the CLS token embedding as the sentence embedding
    # This is a simple approach - in practice, you might use more sophisticated pooling
    sentence_embeddings = model_output.last_hidden_state[:, 0, :]
    
    return sentence_embeddings.cpu().numpy().flatten()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask  # (batch_size, hidden_dim)

# Embedding function
def get_pooled_embeddings(tokens, model, device='cpu'):
    with torch.no_grad():
        model_output = model(**tokens)
    return mean_pooling(model_output, tokens['attention_mask']).cpu().numpy().flatten()