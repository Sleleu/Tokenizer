from src.Tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.train("aaabdaaabac", 500)


import sys
import time

start_time = time.time()
ids = [i for i in range(1000000)] + [1, 2, 3]
new_ids = tokenizer.merge(ids, (1, 2), 256)
print(f"my merge time: {time.time() - start_time:.4f} seconds")
# print(ids)
# print(new_ids)

