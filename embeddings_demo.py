from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, free
sentences = [
    "The cat sat on the mat.",
    "A dog rested on the rug.",
    "The stock market crashed today.",
]
embeddings = model.encode(sentences)
print(f"Each sentence becomes a vector of {len(embeddings[0])} numbers\n")
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
pairs = [
    (0, 1, sentences[0], sentences[1]),
    (0, 2, sentences[0], sentences[2]),
    (1, 2, sentences[1], sentences[2])
]
for i, j, s1, s2 in pairs:
    score = cosine_similarity(embeddings[i], embeddings[j])
    print(f"Similarity: {score:.3f}")
    print(f"  '{s1}'")
    print(f"  '{s2}'\n")