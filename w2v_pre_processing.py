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


### 90% Close to working need to fix
def k_means_cluster_visualisations(embeddings, colour_scheme, number_of_pca_dimensions, n_clusters, new_clusters_column_name):
    colour_map = plt.get_cmap(colour_scheme)
    # Step 1: Stack your embeddings
    X = np.vstack(embeddings.to_numpy())

    # Step 2: Standardize (optional, but helps clustering sometimes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 3: PCA to n dimensions for clustering
    pca = PCA(n_components=number_of_pca_dimensions, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    # Step 4: KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_reduced)

    # Save cluster labels in your dataframe
    df_chunks[new_clusters_column_name] = cluster_labels

    # Step 5: PCA to 2D for plotting only
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_scaled)

    # Step 6: Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap=colour_map, alpha=0.6, s=10)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', s=200, label="Centroids")
    plt.title(f"KMeans Clustering (PCA-reduced {number_of_pca_dimensions}D Embeddings)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()
