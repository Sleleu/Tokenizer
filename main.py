from src.Tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.train("aaabdaaabac", 500)

decoded = tokenizer.decode([256, 32, 257, 33])
print(decoded)
