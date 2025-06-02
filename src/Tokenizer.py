class Tokenizer:

    def __init__(self, vocab={}):
        self.vocab: dict = vocab
        self.vocab_size = None
        self.merges = {}
        self.special_token_ids = []

    def train(self, text=None, filepath=None, merge_nb=500):
        # load char in vocab
        self.vocab = {i: bytes([i]) for i in range(256)}

        if filepath: 
            print("FILEPATH EXIST")
            with open(filepath) as file:
                text = file.read()
        encoded_text = text.encode("utf-8")
        tokens = list(map(int, encoded_text))
        # tokens_initial_len = len(tokens)

        for _ in range(merge_nb):
            occurences = self.find_occurences(tokens)
            if not occurences:
                break # no occurences left
            best_pair = self.get_best_pair(occurences)

            id = len(self.vocab)
            self.merges[best_pair] = id
            if best_pair not in self.vocab.values():
                self.vocab[id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                print(f"New token added in vocabulary, id: {id} | token: {best_pair}")

            tokens = self.merge(tokens, best_pair, id)

        # print("initial token length:", tokens_initial_len)
        # print("after merge:", len(tokens))
        # print("compression ratio:", tokens_initial_len / len(tokens))
        
        special_tokens = (b'<|endoftext|>',
                          b'<|im_start|>',
                          b'<|im_end|>')
        for special_token in special_tokens:
            self.special_token_ids.append(len(self.vocab))
            self.vocab[len(self.vocab)] = special_token

        print("Training complete")

    def find_occurences(self, tokens: list[int]) -> dict[tuple, int]:
        # Get number of occurences for each pair
        occurences = {}
        for i in zip(tokens, tokens[1:]):
            occurences[i] = occurences.get(i, 0) + 1
        return occurences

    def get_best_pair(self, occurences: dict[tuple, int]) -> tuple[int, int]:
        # Get the best pair (max occurences value)
        best_pair = max(occurences, key=lambda k: occurences[k])
        return best_pair

    def merge(self, ids: list, best_pair: tuple, new_id: int) -> list[int]:
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

    def encode_special_tokens(self, tokens: list[int]) -> list[int]:
        special_tok_ids = self.special_token_ids
        tok_to_replace = []

        # gather all list[bytes] of special tokens by id
        for i in special_tok_ids:
            tok_to_replace.append(list(self.vocab[i]))

        try:
            idx = 0
            while idx <= len(tokens):
                idx = tokens.index(60, idx) # search '<'
                # for each '<' found, check if next bytes match with any special token
                for i, special_tok_id in enumerate(special_tok_ids):
                    if tokens[idx : idx + len(tok_to_replace[i])] == tok_to_replace[i]:
                        tokens[idx : idx + len(tok_to_replace[i])] = [special_tok_id]
                        break
                idx += 1
        # index() return value error if no '<' is found in text
        except ValueError:
            print("'<' not found in text")
            pass # not an error, pass
        return tokens



    def encode(self, text) -> list[int]:
        tokens = list(text.encode("utf-8"))
        tokens = self.encode_special_tokens(tokens)

        while True:
            # map pair keys
            stats = self.find_occurences(tokens)
            if not stats: # no merge left
                break
            pair = min(stats, key=lambda k: self.merges.get(k, float("inf")))
            if pair not in self.merges: # no merge left
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
