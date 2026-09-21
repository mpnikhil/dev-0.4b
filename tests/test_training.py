import json
import sys
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, Qwen2Config, Qwen2Model, PreTrainedTokenizerFast
from dev import train
from dev.inference import Predictor


@pytest.mark.parametrize('causal', [False, True])
def test_train_export_and_reload_without_network(tmp_path, monkeypatch, causal):
    base = tmp_path/'base'
    vocab = {'[UNK]':0, '[PAD]':1, '[EOS]':2, 'keep':3, 'drop':4, 'error':5, 'noise':6}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.save_pretrained(base)
    if causal:
        backbone = Qwen2Model(Qwen2Config(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=2, intermediate_size=32))
    else:
        backbone = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                       num_attention_heads=2, intermediate_size=32))
    backbone.save_pretrained(base)
    for split in ['train','validation']:
        rows = []
        for i, state in enumerate(['error', 'noise']):
            for kind in ['choice', 'score', 'noul']:
                q = {'type': kind, 'instructions': 'find error'}
                if kind != 'noul':
                    q['criteria'] = ['keep', 'drop'] if kind == 'choice' else ['noise', 'error', 'keep']
                rows.append(dict(schema_version=2, group=split, state=state, question=q,
                                 label=i if kind == 'choice' else 1-i))
        (tmp_path/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    out = tmp_path/'checkpoint'
    argv = ['train', '--model', str(base), '--train', str(tmp_path/'train.jsonl'),
            '--validation', str(tmp_path/'validation.jsonl'), '--output', str(out),
            '--device','cpu','--epochs','1','--accumulation','1']
    if causal:
        argv += ['--lora-rank','2']
    monkeypatch.setattr(sys, 'argv', argv)
    train.main()
    predictor = Predictor(out, device='cpu')
    result = predictor.predict('find error', 'error', ['keep','drop'])
    answers = predictor.answer('error', {
        'short': {'type': 'score', 'instructions': 'find error', 'criteria': ['noise', 'error']},
        'long': {'type': 'score', 'instructions': 'find error', 'criteria': ['noise', 'error', 'keep']},
    })
    assert len(answers['short']['probabilities']) == 2
    assert len(answers['long']['probabilities']) == 3
    assert abs(sum(answers['short']['probabilities']) - 1) < 1e-5
    assert len(result['choice']['probabilities']) == 2
    assert abs(sum(result['choice']['probabilities'])-1) < 1e-5
    assert 0 <= result['noul'] <= 1
    assert result['calibrated'] is False
    assert (out/'trainer.pt').exists()


def test_train_with_brier_calibration(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    vocab = {'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'keep': 3, 'drop': 4, 'error': 5, 'noise': 6}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token='[UNK]'))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.save_pretrained(base)
    backbone = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                   num_attention_heads=2, intermediate_size=32))
    backbone.save_pretrained(base)
    for split in ['train', 'validation']:
        rows = []
        for i, state in enumerate(['error', 'noise']):
            for kind in ['choice', 'score', 'noul']:
                q = {'type': kind, 'instructions': 'find error'}
                if kind != 'noul':
                    q['criteria'] = ['keep', 'drop'] if kind == 'choice' else ['noise', 'error', 'keep']
                rows.append(dict(schema_version=2, group=split, state=state, question=q,
                                 label=i if kind == 'choice' else 1 - i))
        (tmp_path / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    out = tmp_path / 'calibrated_checkpoint'
    argv = ['train', '--model', str(base), '--train', str(tmp_path / 'train.jsonl'),
            '--validation', str(tmp_path / 'validation.jsonl'), '--output', str(out),
            '--device', 'cpu', '--epochs', '1', '--accumulation', '1',
            '--loss-mode', 'brier', '--freeze-encoder']
    monkeypatch.setattr(sys, 'argv', argv)
    train.main()
    predictor = Predictor(out, device='cpu')
    result = predictor.predict('find error', 'error', ['keep', 'drop'])
    assert result['calibrated'] is True
    answers = predictor.answer('error', {
        'choice': {'type': 'choice', 'instructions': 'find error', 'criteria': ['keep', 'drop']},
    })
    assert answers['choice']['calibrated'] is True

