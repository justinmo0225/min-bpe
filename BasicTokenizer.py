from helper import get_stats, merge

class BasicTokenizer:

  def train(self, text, vocab_size, verbose = False):
    ids = list(text.encode("utf-8"))

    self.merges = {}
    self.vocab = {}
    for i in range(256):
      self.vocab[i] = bytes([i])

    for i in range(vocab_size - 256):
      stats = get_stats(ids)
      pair = max(stats, key = stats.get)
      ids = merge(ids, pair, 256 + i)

      self.merges[pair] = 256 + i
      self.vocab[256 + i] = self.vocab[pair[0]] + self.vocab[pair[1]]

  def decode(self, ids):
    byte_seq = b"".join(self.vocab[id] for id in ids)
    return byte_seq.decode("utf-8")

  def encode(self, text):
    ids = list(text.encode("utf-8"))

    while(True):
      stats = get_stats(ids)
      if not any(p in self.merges for p in stats):
        break

      pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
      id = self.merges[pair]
      ids = merge(ids, pair, id)

    return ids