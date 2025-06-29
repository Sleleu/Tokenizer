import pytest
from src.Tokenizer import Tokenizer

@pytest.mark.parametrize(
    "ids, best_pair, new_id, expected",
    [
        ([1, 2, 3, 4, 1, 2, 5], (1, 2), 42, [42, 3, 4, 42, 5]),
        ([1, 2, 3, 4, 1, 2], (1, 2), 42, [42, 3, 4, 42]),
        ([1, 2, 1, 2], (1, 2), 42, [42, 42]),
        ([1, 2, 1, 2, 100, 100], (1, 100), 42, [1, 2, 1, 2, 100, 100]),
        ([], (1, 2), 42, []),
    ],
)
def test_merge(
    ids: list[int],
    best_pair: tuple[int, int],
    new_id: int,
    expected: list[int],
):
    tokenizer = Tokenizer()
    result = tokenizer.merge(ids, best_pair, new_id)
    assert result == expected

@pytest.mark.parametrize(
    "tokens, expected",
    [
        # "<|endoftext|>" --> 259
        ([60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62], [259]),
        
        # "<|endoftext|> aa <|endoftext|>" --> "259, 32, 97, 97, 32, 259" (token 'aa' not merged yet)
        ([60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62,
          32, 97, 97, 32,
          60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62],
          
          [259, 32, 97, 97, 32, 259]),

        # "<|endoftext|>a" --> '259, 97'
        ([60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62, 97], [259, 97]),

        # "<|endoftext|>a" --> "259, 97"
        ([60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62, 97], [259, 97]),

        # "a<|endoftext|>" --> "97, 259"
        ([97, 60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62], [97, 259]),

        # nothing happen
        ([1, 2, 1, 2, 100, 100], [1, 2, 1, 2, 100, 100]),

        # "<|endoftext|>a<|im_start|>a<|im_end|>a<|endoftext|><<><><||"
        ([60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62,
          97,
          60, 124, 105, 109, 95, 115, 116, 97, 114, 116, 124, 62,
          97,
          60, 124, 105, 109, 95, 101, 110, 100, 124, 62,
          97,
          60, 124, 101, 110, 100, 111, 102, 116, 101, 120, 116, 124, 62,
          60, 60, 62, 60, 62, 60, 124, 124],
          [259, 97, 260, 97, 261, 97, 259, 60, 60, 62, 60, 62, 60, 124, 124]),

        # empty tokens
        ([], []),
    ],
)
def test_encode_special_tokens(
    tokens: list[int],
    expected: list[int],
):
    tokenizer = Tokenizer()
    tokenizer.train("aaabdaaabac", merge_nb=3)

    result = tokenizer.encode_special_tokens(tokens)
    assert result == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # only special token: "<|endoftext|>" -> [259]
        ("<|endoftext|>", [259]),

        # no tokens: "abc" -> [97, 98, 99]
        ("abc", [97, 98, 99]),

        # 'abc' followed by a special token: "abc<|endoftext|>" -> [97, 98, 99, 259]
        ("abc<|endoftext|>", [97, 98, 99, 259]),

        # consecutive tokenisation : "aaabdaaabac" -> 
        # following this merge table : {(97, 97): 256, 
        #                               (256, 97): 257,
        #                               (257, 98): 258}
        #    [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
        # => [256, 97, 98, 100, 256, 97, 98, 97, 99]
        # => [257, 98, 100, 257, 98, 97, 99]
        # => [258, 100, 258, 97, 99]
        ("aaabdaaabac", [258, 100, 258, 97, 99]),

        # same example with special tokens
        ("<|endoftext|>aaabdaaabac<|endoftext|>", [259, 258, 100, 258, 97, 99, 259]),

        # multiple special tokens + encoding
        ("<|endoftext|>a<|im_start|>aa<|im_end|>a<|endoftext|><<><><||aaab", 
         [259, 97, 260, 256, 261, 97, 259, 60, 60, 62, 60, 62, 60, 124, 124, 258]),

        # empty tokens : "" -> []
        ("", []),
    ],
)
def test_encode(
    text: str,
    expected: list[int],
):
    tokenizer = Tokenizer()
    tokenizer.train("aaabdaaabac", merge_nb=3)

    result = tokenizer.encode(text)
    assert result == expected

@pytest.mark.parametrize(
    "param_tokens, expected",
    [
        ('', ''),
        ([97], 'a'),
        ([240, 159, 166, 144, 240, 159, 166, 144], '🦐🦐'),
    ],
)
def test_decode(
    param_tokens: list[int],
    expected: str,
):
    tokenizer = Tokenizer()
    with open ("src/Tokenizer.py", 'r') as f: 
        text = f.read()
    
    tokenizer.train('abc')
    tokens = tokenizer.encode("src/Tokenizer.py")
    assert tokenizer.decode(tokens) == text
    assert tokenizer.decode(param_tokens) == expected