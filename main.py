from src.Tokenizer import Tokenizer
train_text = "aaabdaaabac"

tokenizer = Tokenizer()
tokenizer.train(filepath="tests/taylorswift.txt", merge_nb=1000)

#print("merges:", tokenizer.merges)
print(tokenizer.vocab)
with open("output.txt", "w+") as f:
    for k, v in tokenizer.vocab.items():
        str_v = "".join(i for i in list(v.decode('utf-8', errors="replace")))
        if not str_v.isprintable():
            str_v = v
        f.write(f"[{k}] -> [{str_v}]\n")
#print("ENCODE")
tokens = tokenizer.encode("aaabdaaabac<|endoftext|><|im_start|>hi")
#print(tokens)
#print("DECODE")
string = tokenizer.decode(tokens)
#print(string)