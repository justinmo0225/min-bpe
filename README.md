# minBPE Exercise

A from-scratch implementation of a BPE (Byte Pair Encoding) tokenizer, inspired by [Karpathy's minbpe](https://github.com/karpathy/minbpe) and independently coded as a learning exercise.

## What is BPE?

Byte Pair Encoding is the tokenization algorithm used by GPT-4 and other LLMs. It works by repeatedly merging the most frequent adjacent pairs of tokens until a target vocabulary size is reached.

## File Structure

- `helper.py` — utility functions (`get_stats`, `merge`) used across all tokenizers
- `BasicTokenizer.py` — a simple BPE tokenizer trained on raw UTF-8 bytes
- `RegexTokenizer.py` — extends BasicTokenizer by pre-splitting text with GPT-4's regex pattern, preventing merges across word/punctuation boundaries. Also supports special tokens.
- `GPT4Tokenizer.py` — loads GPT-4's actual merges from tiktoken and reproduces identical token IDs
- `gpt4.py` — helper functions (`bpe`, `recover_merges`) for recovering raw merges from tiktoken, adapted from Karpathy's minbpe

## Usage

```python
from BasicTokenizer import BasicTokenizer

tok = BasicTokenizer()
tok.train("hello world", vocab_size=260)
print(tok.encode("hello world"))  # [259, 32, 119, 111, 114, 108, 100]
print(tok.decode([259, 32, 119, 111, 114, 108, 100]))  # hello world
```

## Matching GPT-4

```python
from GPT4Tokenizer import GPT4Tokenizer

gpt4 = GPT4Tokenizer()
print(gpt4.encode("hello world!!!? (안녕하세요!) lol123 😉"))
# [15339, 1917, 12340, 30, 320, 31495, 230, 75265, 243, 92245, 16715, 28509, 4513, 57037]
```

## References

- [Karpathy's minbpe](https://github.com/karpathy/minbpe)
- [Karpathy's tokenizer lecture](https://www.youtube.com/watch?v=zduSFxRajkE)