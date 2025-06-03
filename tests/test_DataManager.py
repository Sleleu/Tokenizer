import pytest
from src.DataManager import DataManager

@pytest.mark.parametrize(
    "path, expected",
    [
        ("tests/test_data/a.txt", "a\naa\naaa"),
        ("tests/test_data", "a\naa\naaa\n\nbbb\nbb\nb\n\n\nc\ncc\n\n\n\neee"),
    ],
)
def test_load(
    path: str,
    expected: str,
):
    text = DataManager.load(path)
    assert text == expected
