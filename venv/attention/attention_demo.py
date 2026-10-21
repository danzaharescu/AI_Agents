import numpy as np

# ---------------------------------------------------------
# 1. TOKENIZATION: map words to integer IDs
# ---------------------------------------------------------
sentence = "dog cat banana"
words = sentence.split()  # simplest possible tokenizer: split on spaces

vocab = {"dog": 0, "cat": 1, "banana": 2}  # word -> ID lookup table
token_ids = [vocab[w] for w in words]

print("Words:    ", words)
print("Token IDs:", token_ids)
# Words:     ['dog', 'cat', 'banana']
# Token IDs: [0, 1, 2]

# ---------------------------------------------------------
# 2. EMBEDDING: look up a vector for each token ID
# ---------------------------------------------------------
# This table would normally be LEARNED during training.
# Row i = the embedding for vocab ID i.
#                    [is_animal, is_pet, is_food, size]
embedding_table = np.array([
    [1.0, 1.0, 0.0, 0.6],   # ID 0 = dog
    [1.0, 1.0, 0.0, 0.3],   # ID 1 = cat
    [0.0, 0.0, 1.0, 0.2],   # ID 2 = banana
])

X = embedding_table[token_ids]  # turn each ID into its row vector
print("\nEmbeddings (X):")
print(X)

# ---------------------------------------------------------
# 3. ATTENTION on those embeddings
# ---------------------------------------------------------
def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    weights = softmax(scores)
    return weights @ V, weights

output, weights = attention(X, X, X)

print("\nAttention weights:")
print("        ", "  ".join(f"{w:>7}" for w in words))
for i, row in enumerate(weights):
    print(f"{words[i]:>8}", "  ".join(f"{v:.2f}   " for v in row))

print("\nOutput (context-aware embeddings):")
print(np.round(output, 2))