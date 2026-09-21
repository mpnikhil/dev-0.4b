import copy
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, BertConfig, BertModel
from dev.data import validate, encode, collate
from dev.model import ThreeHeadEncoder, multitask_loss


@pytest.fixture
def tokenizer():
    words = ['[UNK]', '[PAD]', 'assertion', 'collection', 'failed', 'passed', 'unknown', 'unrelated', 'partial', 'direct']
    tok = Tokenizer(WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]')


def row(kind='choice'):
    q = {'type': kind, 'instructions': 'assertion failed'}
    if kind != 'noul':
        q['criteria'] = ['failed', 'passed']
    return {'schema_version': 2, 'group': 'public/repo', 'state': 'assertion failed', 'question': q, 'label': 0}


def test_question_and_criteria_change_inputs_and_pooled_spans(tokenizer):
    a = row(); b = copy.deepcopy(a); b['question']['instructions'] = 'collection failed'
    assert encode(a, tokenizer, 128)['input_ids'] != encode(b, tokenizer, 128)['input_ids']
    a['question']['criteria'].reverse()
    encoded = encode(a, tokenizer, 128)
    ids = encoded['input_ids']
    selected = [[ids[i] for i, v in enumerate(mask) if v] for mask in encoded['candidate_mask']]
    assert selected == [[tokenizer.convert_tokens_to_ids('passed')], [tokenizer.convert_tokens_to_ids('failed')]]
    with pytest.raises(ValueError, match='do not truncate'):
        encode(a, tokenizer, 4)


def test_mixed_primitives_variable_rubrics_and_padding_invariance(tokenizer):
    torch.manual_seed(5)
    rows = [row('choice'), row('score'), row('noul')]
    long = row('score'); long['question']['criteria'] = ['unrelated', 'partial', 'direct']; long['label'] = 2
    rows.append(long)
    encoded = [encode(r, tokenizer, 128) for r in rows]
    batch = collate(encoded, tokenizer.pad_token_id)
    model = ThreeHeadEncoder(BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                             num_hidden_layers=1, num_attention_heads=2, intermediate_size=32))).eval()
    result = model(**batch)
    assert result['score'][1].softmax(-1)[2] == 0
    alone = model(**collate([encoded[1]], tokenizer.pad_token_id))
    assert torch.allclose(result['score'][1, :2], alone['score'][0], atol=1e-6)
    loss = multitask_loss(result, batch['labels']); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for head in (model.choice, model.score, model.noul) for p in head.parameters())
    assert model.encoder.embeddings.word_embeddings.weight.grad is not None


def test_old_schema_and_bad_labels_rejected():
    with pytest.raises(ValueError, match='schema_version'):
        validate({'objective': 'old contract'})
    bad = row('score'); bad['label'] = 4
    with pytest.raises(ValueError, match='Invalid score'):
        validate(bad)
    bad = row(); bad['question']['criteria'] = ['duplicate', 'duplicate']
    with pytest.raises(ValueError, match='unique'):
        validate(bad)
