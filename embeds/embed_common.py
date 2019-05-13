
def find_analogies(w1, w2, w3):
  for w in (w1, w2, w3):
    if w not in word2vec:
      print("%s not in dictionary" % w)
      return

  king = word2vec[w1]
  man = word2vec[w2]
  woman = word2vec[w3]
  v0 = king - man + woman

  distances = pairwise_distances(v0.reshape(1, D), embedding, metric=metric).reshape(V)
  idxs = distances.argsort()[:4]
  for idx in idxs:
    word = idx2word[idx]
    if word not in (w1, w2, w3):
      best_word = word
      break

  print(w1, "-", w2, "=", best_word, "-", w3)