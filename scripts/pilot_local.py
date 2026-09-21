"""Bounded local MPS full-finetuning pilot; synthetic labels, no quality claims."""
import json
from pathlib import Path
import statistics
import time
import torch
from transformers import AutoTokenizer
from dev.model import ThreeHeadEncoder, multitask_loss


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process')
    torch.manual_seed(17)
    name = 'answerdotai/ModernBERT-base'
    model = ThreeHeadEncoder.pretrained(name).to('mps')
    tokenizer = AutoTokenizer.from_pretrained(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    length = 512
    tokens = tokenizer.encode('Objective: diagnose test failure. Command: pytest. Output: FAILED tests/test_cache.py AssertionError expected 200 got 500. ', add_special_tokens=False)
    ids = (tokens*(length//len(tokens)+1))[:length]
    ids[-1] = tokenizer.eos_token_id or ids[-1]
    batch = {'input_ids': torch.tensor([ids]*2, device='mps'),
             'attention_mask': torch.ones(2, length, dtype=torch.long, device='mps'),
             'candidate_mask': torch.zeros(2, 3, length, device='mps'),
             'candidate_valid': torch.ones(2, 3, dtype=torch.bool, device='mps')}
    for c in range(3):
        batch['candidate_mask'][:, c, length-5+c] = 1
    labels = {'choice': torch.tensor([0,1], device='mps'), 'score': torch.tensor([2,1], device='mps'),
              'noul': torch.tensor([1,0], device='mps')}
    timings = []
    for step in range(5):
        torch.mps.synchronize(); started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = multitask_loss(model(**batch), labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.mps.synchronize()
        elapsed = time.perf_counter()-started
        if step >= 2:
            timings.append(elapsed)
        print(json.dumps({'step': step+1, 'seconds': elapsed, 'loss': loss.item()}), flush=True)
    model.eval()
    inference = []
    with torch.inference_mode():
        for i in range(5):
            torch.mps.synchronize(); started = time.perf_counter()
            model(**batch)
            torch.mps.synchronize()
            if i >= 2:
                inference.append(time.perf_counter()-started)
    report = {'purpose':'throughput_and_memory_only','device':'MPS','model':name,
              'precision':'fp32','attention':'eager','batch_size':2,'length':512,
              'method':'full_finetune','optimizer_step_s':statistics.median(timings),
              'batch_inference_s':statistics.median(inference),
              'current_allocated_gib':torch.mps.current_allocated_memory()/2**30,
              'driver_allocated_gib':torch.mps.driver_allocated_memory()/2**30,
              'memory_note':'Current allocations after inference, NOT peak training memory'}
    out = Path('runs/local-pilot'); out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
