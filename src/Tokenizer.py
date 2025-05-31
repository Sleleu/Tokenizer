class Tokenizer:

    def __init__(self, vocab={}):
        self.vocab: dict = vocab
        self.vocab_size = None

    def train(self, text, vocab_size):
        encoded_text = self.encode(text)

        # load char in vocab
        # self.vocab = {i : chr(i) for i in range(32, 127)}
        # print(self.vocab)

        self.byte_pair_encoding(text, 500, 3)

    def byte_pair_encoding(self, text, vocab_size, num_merge: int):

        # convert text into raw bytes string
        encoded_text = text.encode("utf-8", errors="replace")

        # convert raw bytes string to a list of integers
        tokens = list(map(int, encoded_text))

        print(f"Text length: {len(text)} | Token list length : {len(tokens)}")
        print("tokens:", tokens)

        for merge in range(num_merge):
            # Get number of occurences for each pair
            occurences = {}
            for i in zip(tokens, tokens[1:]):
                occurences[i] = occurences.get(i, 0) + 1
            print("occurences:", occurences)

            # Get the top pair (max occurences value)
            top_pair = max(occurences, key=lambda k: occurences[k])
            print("top_pair:", top_pair)

            new_ids = self.merge(tokens, top_pair, 255 + merge)
            tokens = new_ids
            for i in tokens:
                print(chr(i), end="")
            print()
            for i in new_ids:
                print(chr(i), end="")
            print()

            idx = 256 + merge
            self.vocab[top_pair] = idx

    def merge(self, ids: list, top_pair: tuple, new_id: int):
        if not ids:
            return []

        new_ids = []
        added_pair = False
        for pair in zip(ids, ids[1:]):
            if added_pair:
                added_pair = False
                continue
            if pair == top_pair:
                new_ids.append(new_id)
                added_pair = True
            else:
                new_ids.append(pair[0])

        # add the last id if it's not part of a top pair
        if not added_pair:
            new_ids.append(ids[-1])
        return new_ids

    def encode(self, text):
        return text.encode("utf-8")

    def decode(self, ids):
        pass
