import os
import ast

class DataManager:

    @staticmethod
    def load(path: str) -> str:
        if os.path.isfile(path):
            return DataManager._load_file(path)
        if os.path.isdir(path):
            return DataManager._load_folder(path)
        raise FileNotFoundError(f"'{path}' not found")

    @staticmethod
    def _load_file(filepath: str) -> str:
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"'{filepath}' not found")
        with open(filepath, "r", encoding="utf-8") as f:
            print(filepath)
            return f.read()

    @staticmethod
    def _load_folder(folderpath: str) -> str:
        if not os.path.isdir(folderpath):
            raise FileNotFoundError(f"'{folderpath}' not found")

        texts = []
        for name in sorted(os.listdir(folderpath)):
            full = os.path.join(folderpath, name)
            if os.path.isdir(full):
                sub = DataManager._load_folder(full)
                if sub:
                    texts.append(sub)
            elif os.path.isfile(full):
                texts.append(DataManager._load_file(full))

        return "\n\n".join(texts)

    # TODO: change save/load model logic
    # ==== SAVE MODEL METHODS ====
    @staticmethod
    def vocab_save(vocab: dict[int, bytes], folderpath: str, filename: str) -> None:
        name = os.path.join(folderpath, filename)
        with open(name, "w+") as f:
            for k, bytes_v in vocab.items():
                str_v = "".join(i for i in list(bytes_v.decode('utf-8', errors="replace")))
                if str_v.isprintable():
                    f.write(f"[{k}] -> [{str_v}]\n")
                else:
                    f.write(f"[{k}] -> [{bytes_v}]\n")

    @staticmethod
    def merges_save(merges: dict[tuple[int, int], int], folderpath: str, filename: str) -> None:
        name = os.path.join(folderpath, filename)
        with open(name, "w+") as f:
            for k, v in merges.items():
                f.write(f"{k} -> {v}\n")


    @staticmethod
    def special_tok_ids_save(special_tok_ids: list[int], folderpath: str, filename: str) -> None:
        name = os.path.join(folderpath, filename)
        with open(name, "w+") as f:
            for id in special_tok_ids:
                f.write(f"{id}\n")
    
    # ==== LOAD MODEL METHODs ====
    @staticmethod
    def vocab_load(folderpath: str) -> dict[int, bytes]:
        vocab = {}
        last_folder_path = os.path.split(folderpath)[1]
        filename = os.path.join(folderpath, last_folder_path) + "_vocab.txt"
        with open(filename, "r") as f:
            for line in f:
                token_id, value = map(lambda x: x.strip('[]'), line.strip().split(" -> "))
                if value.startswith("b'") and value.endswith("'"):
                    value_bytes = ast.literal_eval(value)
                    vocab[int(token_id)] = bytes(value_bytes)
                else:
                    vocab[int(token_id)] = value.encode("utf-8")
        return vocab

    @staticmethod
    def merges_load(folderpath: str) -> dict[tuple[int, int], int]:
        merges = {}
        last_folder_path = os.path.split(folderpath)[1]
        filename = os.path.join(folderpath, last_folder_path) + "_merges.txt"
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                merge_id, new_id = line.split(" -> ")
                merge_id = tuple(map(int, merge_id.strip('()').split(", ")))
                merges[merge_id] = int(new_id)
        return merges
        

    @staticmethod
    def special_tok_ids_load(folderpath: str) -> list[int]:
        special_tok_ids = []
        last_folder_path = os.path.split(folderpath)[1]
        filename = os.path.join(folderpath, last_folder_path) + "_special_tokens_ids.txt"
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                special_tok_ids.append(int(line))
        return special_tok_ids
                