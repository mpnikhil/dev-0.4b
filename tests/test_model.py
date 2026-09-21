import torch
from transformers import BertConfig, BertModel, Qwen2Config, Qwen2Model
from dev.model import ThreeHeadEncoder, multitask_loss


def batch():
    return dict(input_ids=torch.tensor([[1, 2, 3, 4, 0], [1, 2, 5, 6, 7]]),
                attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
                candidate_mask=torch.tensor([[[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
                                             [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]]).float(),
                candidate_valid=torch.tensor([[True, False], [True, True]]))


def test_encoder_heads_backpropagate_and_mask_invalid_candidates():
    torch.manual_seed(1)
    model = ThreeHeadEncoder(BertModel(BertConfig(vocab_size=20, hidden_size=16, num_hidden_layers=1,
                                                num_attention_heads=2, intermediate_size=32)))
    result = model(**batch())
    assert result['choice'][0].softmax(-1)[1] == 0
    assert result['score'][0].softmax(-1)[1] == 0
    labels = {'choice': torch.tensor([0, 1]), 'score': torch.tensor([0, 1]), 'noul': torch.tensor([1, 0])}
    loss = multitask_loss(result, labels)
    loss.backward()
    assert torch.isfinite(loss)
    for head in (model.choice, model.score, model.noul):
        assert all(p.grad is not None for p in head.parameters())
    assert model.encoder.embeddings.word_embeddings.weight.grad is not None


def test_causal_pooling_uses_final_nonpadding_token_and_sees_late_evidence():
    torch.manual_seed(2)
    config = Qwen2Config(vocab_size=20, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, intermediate_size=32)
    model = ThreeHeadEncoder(Qwen2Model(config)).eval()
    assert model.pooling == 'last'
    b = batch()
    before = model(**b)['noul']
    b['input_ids'][0, 3] = 9
    after = model(**b)['noul']
    assert not torch.allclose(before[0], after[0])
    assert torch.allclose(before[1], after[1])
    b['input_ids'][0, 4] = 18  # masked padding must not affect the result
    assert torch.allclose(after, model(**b)['noul'])


def test_universal_single_head_unification():
    config = BertConfig(vocab_size=20, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    model = ThreeHeadEncoder(BertModel(config))
    # Assert true parameter and module identity
    assert model.score is model.choice
    assert model.noul is model.choice
    assert model.head is model.choice
    # Single-head state_dict only contains choice.* parameters
    head_param_names = [n for n, _ in model.named_parameters() if not n.startswith('encoder.')]
    assert head_param_names == ['choice.0.weight', 'choice.0.bias', 'choice.2.weight', 'choice.2.bias']
    # Verify Noul probability equals 2-way Softmax probability
    b = batch()
    result = model(**b)
    probs_2way = result['choice'].softmax(-1)
    sig_noul = result['noul'].sigmoid()
    assert torch.allclose(probs_2way[1, 1], sig_noul[1], atol=1e-6)


def test_multitask_brier_loss_and_calibration():
    torch.manual_seed(42)
    config = BertConfig(vocab_size=20, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)
    model = ThreeHeadEncoder(BertModel(config))
    result = model(**batch())
    labels = {'choice': torch.tensor([0, 1]), 'score': torch.tensor([0, 1]), 'noul': torch.tensor([1, 0])}
    loss = multitask_loss(result, labels, loss_mode="brier")
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    for head in (model.choice, model.score, model.noul):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


