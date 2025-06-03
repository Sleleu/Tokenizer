from src.Tokenizer import Tokenizer
train_text = "aaabdaaabac"

tokenizer = Tokenizer()
# tokenizer.train(path="data", merge_nb=20000)
# tokenizer.save("model/sleleu_20K")



tokenizer.get_encoding("model/sleleu_20k")
print(tokenizer.encode("<|endoftext|>anabc"))

#print("merges:", tokenizer.merges)
#print("ENCODE")
tokens = tokenizer.encode("aaabdaaabac<|endoftext|><|im_start|>hi")
print(tokens)
#print("DECODE")
string = tokenizer.decode(tokens)
print(string)