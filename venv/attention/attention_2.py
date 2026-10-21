import numpy as np

# ---------------------------------------------------------
# 1. Toy corpus + vocab
# ---------------------------------------------------------
text = "dog cat animal pet dog cat animal pet banana fruit food banana fruit food"
words = text.split()

vocab = sorted(set(words))
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
V = len(vocab)  # vocab size

print("Vocab:", word2id)

# ---------------------------------------------------------
# 2. Build training pairs: (center_word, context_word)
#    "context" = words within a small window
# ---------------------------------------------------------
window = 2
pairs = []
ids = [word2id[w] for w in words]

for i, center in enumerate(ids):
    for j in range(max(0, i - window), min(len(ids), i + window + 1)):
        if j != i:
            pairs.append((center, ids[j]))

print(f"\n{len(pairs)} training pairs, e.g.:", pairs[:5])

# ---------------------------------------------------------
# 3. Initialize embeddings RANDOMLY (this is the untrained state)
# ---------------------------------------------------------
dim = 2  # tiny, so we can print/plot it easily
np.random.seed(0)
embed_in  = np.random.randn(V, dim) * 0.1   # "center word" vectors
embed_out = np.random.randn(V, dim) * 0.1   # "context word" vectors

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---------------------------------------------------------
# 4. Train with plain gradient descent
#    Task: given center word, predict context word
# ---------------------------------------------------------
lr = 0.05
epochs = 500

for epoch in range(epochs):
    total_loss = 0
    for center_id, context_id in pairs:
        v_center = embed_in[center_id]            # (dim,)
        scores = embed_out @ v_center              # (V,) similarity to every word
        probs = softmax(scores)                    # predicted distribution

        loss = -np.log(probs[context_id] + 1e-9)
        total_loss += loss

        # gradient of softmax cross-entropy
        d_scores = probs.copy()
        d_scores[context_id] -= 1                  # (V,)

        grad_v_center = embed_out.T @ d_scores      # (dim,)
        grad_embed_out = np.outer(d_scores, v_center)  # (V, dim)

        embed_in[center_id]  -= lr * grad_v_center
        embed_out            -= lr * grad_embed_out

    if epoch % 100 == 0:
        print(f"epoch {epoch:4d}  loss {total_loss/len(pairs):.3f}")

# ---------------------------------------------------------
# 5. Inspect the LEARNED embeddings
# ---------------------------------------------------------
print("\nLearned embeddings (word -> vector):")
for w in vocab:
    print(f"  {w:8s} {np.round(embed_in[word2id[w]], 2)}")

# ---------------------------------------------------------
# 6. Check: are similar words actually close together now?
# ---------------------------------------------------------
def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nSimilarity check:")
pairs_to_check = [("dog", "cat"), ("dog", "banana"), ("banana", "fruit")]
for a, b in pairs_to_check:
    sim = cosine_sim(embed_in[word2id[a]], embed_in[word2id[b]])
    print(f"  sim({a}, {b}) = {sim:.3f}")