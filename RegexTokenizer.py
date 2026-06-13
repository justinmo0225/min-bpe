import regex
from helper import get_stats, merge

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
  # creates a function to register special tokens
  def register_special_tokens(self, special_tokens):
    self.special_tokens = special_tokens
    for token, id in special_tokens.items():
        self.vocab[id] = token.encode("utf-8")

  def train(self, text, vocab_size, verbose = False):

    self.special_tokens = {}  # initialized even if `register_special_tokens` is never called

    ids = []
    chunks = regex.findall(GPT4_SPLIT_PATTERN, text)
    for part in chunks:
      ids.append(list(part.encode("utf-8")))

    self.merges = {}
    self.vocab = {}
    for i in range(256):
      self.vocab[i] = bytes([i])

    for i in range(vocab_size - 256):
      stats = {}
      for part in ids:
          for pair, count in get_stats(part).items():
              stats[pair] = stats.get(pair, 0) + count

      pair = max(stats, key = stats.get)
      ids = [merge(part, pair, 256 + i) for part in ids]

      self.merges[pair] = 256 + i
      self.vocab[256 + i] = self.vocab[pair[0]] + self.vocab[pair[1]]

  def decode(self, ids):
    byte_seq = b"".join(self.vocab[id] for id in ids)
    return byte_seq.decode("utf-8")

  def encode(self, text):
    pattern = "|".join(regex.escape(token) for token in self.special_tokens)  # joins every special token; regex.escape tells regex to treat special characters literally instead of as regex syntax
    chunks = regex.split(f"({pattern})", text)  # by saving our pattern, it prevents regex from splitting on the special token and throwing it out

    ids = []
    for chunk in chunks:
        # we want to add the special token as is to our ids list
        if chunk in self.special_tokens:
            ids.append(self.special_tokens[chunk])
        # if it is just a normal token, then do BPE
        elif chunk:
            chunk_ids = list(chunk.encode("utf-8"))
            # our previous BPE logic
            while True:
                stats = get_stats(chunk_ids)
                if not any(p in self.merges for p in stats):
                    break
                pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
                chunk_ids = merge(chunk_ids, pair, self.merges[pair])
            ids.extend(chunk_ids)

    return ids