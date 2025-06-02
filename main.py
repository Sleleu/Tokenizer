from src.Tokenizer import Tokenizer
train_text = "aaabdaaabac"

tokenizer = Tokenizer()
tokenizer.train(text=train_text, merge_nb=3)

print("merges:", tokenizer.merges)
print(tokenizer.vocab)
print("ENCODE")
tokens = tokenizer.encode("aaabdaaabac")
print(tokens)
print("DECODE")
string = tokenizer.decode(tokens)
print(string)