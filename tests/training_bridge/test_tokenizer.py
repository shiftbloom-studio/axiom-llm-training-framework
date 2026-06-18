from pathlib import Path

from hcaps.training_bridge.tokenizer import WhitespaceTokenizer


def test_whitespace_tokenizer_roundtrip(tmp_path: Path) -> None:
    texts = ["Hello world", "Testing tokenizer"]
    tok = WhitespaceTokenizer()
    tok.fit(texts)
    encoded = [tok.encode(t) for t in texts]
    decoded = [tok.decode(ids) for ids in encoded]
    assert decoded[0] == "hello world"
    assert decoded[1] == "testing tokenizer"
    # Test save/load
    file_path = tmp_path / "tok.json"
    tok.save(file_path)
    tok2 = WhitespaceTokenizer.load(file_path)
    assert tok2.decode(tok2.encode("hello world")) == "hello world"
