
from localai.tokenizer import CharTokenizer


def test_encode_decode():
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["안녕하세요"])

    ids = tokenizer.encode("안녕")

    assert len(ids) == 2
    assert tokenizer.decode(ids) == "안녕"


def test_unknown_character():
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["가나다"])

    ids = tokenizer.encode("가XYZ")

    assert ids[1] == tokenizer.stoi[tokenizer.UNK]
    assert ids[2] == tokenizer.stoi[tokenizer.UNK]


def test_save_load(tmp_path):
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(["한글 테스트"])

    path = tmp_path / "vocab.json"
    tokenizer.save(path)

    loaded = CharTokenizer.load(path)

    assert loaded.stoi == tokenizer.stoi
    assert loaded.decode(
        loaded.encode("한글")
    ) == "한글"
