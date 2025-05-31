class Tokenizer:

    def __init__(self, vocab={}):
        self.vocab: dict = vocab
        self.vocab_size = None

    def train(self, text, vocab_size):
        print("Begin training phase")
        encoded_text = self.encode(text)

        # load char in vocab
        self.vocab = {i: bytes([i]) for i in range(256)}
        print(self.vocab)

        # convert text into raw bytes string
        encoded_text = text.encode("utf-8", errors="replace")

        # convert raw bytes string to a list of integers
        tokens = list(map(int, encoded_text))
        tokens_initial_len = len(tokens)
        num_merge = 3

        for _ in range(num_merge):
            best_pair = self.get_best_pair(tokens)

            id = len(self.vocab)
            if best_pair not in self.vocab.values():
                self.vocab[id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                print(f"New token added in vocabulary, id: {id} | token: {best_pair}")

            tokens = self.merge(tokens, best_pair, id)
            # print(self.vocab)

            # self.byte_pair_encoding(text, 3)
        print("initial token length:", tokens_initial_len)
        print("after merge:", len(tokens))
        print("compression ratio:", tokens_initial_len / len(tokens))

        print(self.vocab)
        print("Training complete")

    def get_best_pair(self, tokens: list[int]) -> tuple[int, int]:
        # Get number of occurences for each pair
        occurences = {}
        for i in zip(tokens, tokens[1:]):
            occurences[i] = occurences.get(i, 0) + 1

        # Get the best pair (max occurences value)
        best_pair = max(occurences, key=lambda k: occurences[k])

        return best_pair

    def merge(self, ids: list, best_pair: tuple, new_id: int):
        if not ids:
            return []

        new_ids = []
        added_pair = False
        for pair in zip(ids, ids[1:]):
            if added_pair:
                added_pair = False
                continue
            if pair == best_pair:
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

    def decode(self, ids: list[int]):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
