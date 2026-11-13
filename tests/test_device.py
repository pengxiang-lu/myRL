import pytest
from g1_rl.device import resolve_device


def test_cuda_unavailable_is_error(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        resolve_device("cuda")
    assert str(resolve_device("auto")) == "cpu"
    assert str(resolve_device("cpu")) == "cpu"


def test_invalid_device_index(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_device("cuda:1")


@pytest.mark.parametrize("name", ["mps", "not-a-device"])
def test_unsupported_device(name):
    with pytest.raises(ValueError):
        resolve_device(name)
