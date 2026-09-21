import pytest
from dev.shim import RawStore, chunks, compress, shortlist


class Drop:
    def predict(self, *args):
        return {'choice': {'index': 1, 'probabilities': [.01, .98, .01]},
                'noul': .01, 'score': {'value': .1}}


def test_preserves_errors_and_exact_original(tmp_path):
    text = 'progress heartbeat\n' * 500 + 'ERROR src/cache.py:42 failed\n'
    store = RawStore(tmp_path)
    result = compress(text, 'Fix the failure', Drop(), store, 1000, True)
    assert result['compressed']
    assert 'ERROR src/cache.py:42 failed' in result['text']
    assert store.get(result['raw_ref']) == text
    assert ''.join(x[2] for x in chunks(text)) == text


def test_shadow_mode_is_lossless_and_store_rejects_traversal(tmp_path):
    store = RawStore(tmp_path)
    assert compress('a'*9000, 'task', Drop(), store)['text'] == 'a'*9000
    with pytest.raises(ValueError):
        store.get('../outside')


def test_uncertain_router_preserves_full_schemas():
    class Uncertain:
        def predict(self, *args):
            return {'choice': {'probabilities': [.25]*4}}
    tools = [{'name': str(i), 'inputSchema': {'required': ['path']}} for i in range(4)]
    assert shortlist(tools, 'task', Uncertain(), experimental=True) == tools
