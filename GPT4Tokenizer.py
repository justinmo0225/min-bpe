import regex
import tiktoken
from RegexTokenizer import RegexTokenizer, GPT4_SPLIT_PATTERN
from gpt4 import recover_merges
from helper import get_stats, merge

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):

        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        merges = recover_merges(mergeable_ranks)
        byte_shuffle = {i: enc._mergeable_ranks[bytes([i])] for i in range(256)}
        
        self.merges = merges
        self.vocab = {}
        for i in range(256):
            self.vocab[byte_shuffle[i]] = bytes([i])  # shuffled ID maps back to the original byte
        self.byte_shuffle = byte_shuffle

        # builds the vocab for all merged tokens
        # the bytes that token `rank` represent is equal to the bytes of `p0 + p1`
        for (p0, p1), rank in self.merges.items():
          self.vocab[rank] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
      ids = []
      chunks = regex.findall(GPT4_SPLIT_PATTERN, text)
      for part in chunks:
        ids.append([self.byte_shuffle[b] for b in part.encode("utf-8")])  # each byte `b` in the chunk needs to be replaced by the byte shuffle function

      while(True):
        stats = {}
        for part in ids:
            for pair, count in get_stats(part).items():
                stats[pair] = stats.get(pair, 0) + count
        if not any(p in self.merges for p in stats):
          break

        pair = min(stats, key = lambda p: self.merges.get(p, float("inf")))
        id = self.merges[pair]

        ids = [merge(part, pair, id) for part in ids]

      return [id for part in ids for id in part]