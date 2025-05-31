import pytest
from src.Tokenizer import Tokenizer

@pytest.fixture
def tokenizer():
    return Tokenizer()

@pytest.mark.parametrize(
    "ids, pair, new_id, expected",
    [
        ([1, 2, 3, 4, 1, 2, 5], (1, 2), 42, [42, 3, 4, 42, 5]),
        ([1, 2, 3, 4, 1, 2], (1, 2), 42, [42, 3, 4, 42]),
        ([1, 2, 1, 2], (1, 2), 42, [42, 42]),
        ([1, 2, 1, 2, 100, 100], (1, 100), 42, [1, 2, 1, 2, 100, 100]),
        ([], (1, 2), 42, []),
    ],
)
def test_merge(
    tokenizer: Tokenizer,
    ids: list[int],
    pair: tuple[int, int],
    new_id: int,
    expected: list[int],
):
    result = tokenizer.merge(ids, pair, new_id)
    assert result == expected
