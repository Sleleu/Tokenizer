import regex as re
import os
from src.DataManager import DataManager

class Tokenizer:

    def __init__(self, vocab={}, pattern=None):
        self.vocab: dict = vocab
        self.vocab_size = None
        self.merges = {}
        self.special_token_ids = []
        if pattern is None:
            # gpt-4 pattern
            self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        else:
            self.pattern = pattern
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text=None, path=None, merge_nb=500):
        # load char in vocab
        self.vocab = {i: bytes([i]) for i in range(256)}

        if path: 
            text = DataManager.load(path)
        
        if self.pattern is None:
            encoded_text = text.encode("utf-8")
            tokens = [list(map(int, encoded_text))]
        else:
            text_chunks = re.findall(self.compiled_pattern, text)
            tokens = [list(chunk.encode("utf-8")) for chunk in text_chunks]


        for i in range(merge_nb):
            if i % (merge_nb / 10000) == 0:
                print(f"Merge {i} / {merge_nb}")

            occurences = {}
            for chunk_token in tokens:   
                self.find_occurences(chunk_token, occurences)
            if not occurences:
                break # no occurences left
            best_pair = self.get_best_pair(occurences)
            id = 256 + i
            self.merges[best_pair] = id
            if best_pair not in self.vocab.values():
                self.vocab[id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
                    #print(f"New token added in vocabulary, id: {id} | token: {best_pair}")

            tokens = [self.merge(token_chunk, best_pair, id) for token_chunk in tokens]
        
        special_tokens = (b'<|endoftext|>',
                          b'<|im_start|>',
                          b'<|im_end|>')
        for special_token in special_tokens:
            self.special_token_ids.append(len(self.vocab))
            self.vocab[len(self.vocab)] = special_token

        print("Training complete")

    def find_occurences(self, tokens: list[int], occurences: dict) -> dict[tuple[int, int], int]:
        # Get number of occurences for each pair
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
            pass # not an error, pass
        return tokens



    def encode(self, text) -> list[int]:
        tokens = list(text.encode("utf-8"))
        tokens = self.encode_special_tokens(tokens)

        while True:
            # map pair keys
            stats = self.find_occurences(tokens, occurences={})
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
    
    def save(self, folderpath: str) -> None:

        # take tail path to add a vocab/merges name
        last_folder_path = os.path.split(folderpath)[1]
        vocab_name = last_folder_path + "_vocab.txt"
        merges_name = last_folder_path + "_merges.txt"
        special_tok_ids_name = last_folder_path + "_special_tokens_ids.txt"

        DataManager.vocab_save(self.vocab, folderpath, vocab_name)
        DataManager.merges_save(self.merges, folderpath, merges_name)
        DataManager.special_tok_ids_save(self.special_token_ids, folderpath, special_tok_ids_name)

    def get_encoding(self, folderpath: str) -> None:
        """
        Given a folder path, load these elements in `Tokenizer` instance:
        - "foldername/foldername_vocab.txt" -> self.vocab : `dict[int, bytes]`
        - "foldername/foldername_merges.txt" -> self.merges : `dict[tuple[int, int], int]`
        - "foldername/foldername_merges.txt" -> self.special_token_ids : `list[int]`
        """
        self.vocab = DataManager.vocab_load(folderpath)
        self.merges = DataManager.merges_load(folderpath)
        self.special_token_ids = DataManager.special_tok_ids_load(folderpath)

