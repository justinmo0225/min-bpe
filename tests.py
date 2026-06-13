from BasicTokenizer import BasicTokenizer
from RegexTokenizer import RegexTokenizer
from GPT4Tokenizer import GPT4Tokenizer
import tiktoken

# ---- BasicTokenizer ----
print("=== BasicTokenizer ===")
tok = BasicTokenizer()
tok.train("hello world", vocab_size=260)
print(tok.encode("hello world"))
print(tok.decode(tok.encode("hello world")))

# ---- RegexTokenizer ----
print("\n=== RegexTokenizer ===")
tok2 = RegexTokenizer()
tok2.train("hello world! 123 how're you?", vocab_size=270)
tok2.register_special_tokens({"<|endoftext|>": 100257})
print(tok2.encode("<|endoftext|>hello world"))
print(tok2.decode([259, 264, 33]))

# ---- GPT4Tokenizer ----
print("\n=== GPT4Tokenizer ===")
gpt4 = GPT4Tokenizer()
enc = tiktoken.get_encoding("cl100k_base")

test_text = "hello world!!!? (안녕하세요!) lol123 😉"
your_ids = gpt4.encode(test_text)
tiktoken_ids = enc.encode(test_text)

print(your_ids)
print(tiktoken_ids)
print("Match:", your_ids == tiktoken_ids)