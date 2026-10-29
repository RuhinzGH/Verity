"""NLIVerifier batches pairs shortest first; results must still come back in input order."""
from types import SimpleNamespace

import numpy as np
import torch

from verity import verify


class _Enc(dict):
    def to(self, device):
        return self


def _stub_verifier(batch_size: int) -> verify.NLIVerifier:
    """A verifier whose 'model' returns logits [n, -n, 0] for a claim of n characters."""
    v = object.__new__(verify.NLIVerifier)
    v.tok = lambda ev, cl, **kw: _Enc(n=torch.tensor([float(len(c)) for c in cl]))
    v.model = lambda n: SimpleNamespace(logits=torch.stack([n, -n, torch.zeros_like(n)], 1))
    v.cols, v.max_length, v.batch_size, v.torch = [0, 1, 2], 512, batch_size, torch
    return v


def test_logits_keep_input_order():
    pairs = [("e" * 40, "c" * 7), ("e", "c" * 2), ("e" * 90, "c" * 5), ("e" * 3, "c" * 9), ("e", "c")]
    z = _stub_verifier(batch_size=2).logits(pairs)
    assert z[:, 0].tolist() == [7, 2, 5, 9, 1]
    assert z[:, 1].tolist() == [-7, -2, -5, -9, -1]


def test_predict_is_softmax_of_logits():
    pairs = [("long evidence " * 5, "a claim"), ("e", "another, longer claim")]
    v = _stub_verifier(batch_size=1)
    np.testing.assert_allclose(v.predict(pairs), verify.softmax(v.logits(pairs)), rtol=1e-6)
    assert v.logits([]).shape == (0, 3)
