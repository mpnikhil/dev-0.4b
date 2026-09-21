"""GPU throughput/memory pilot. Synthetic labels: never a quality evaluation.

Executed remotely after project sources have been packed by build_pilot.py.
"""
import gc
import os
import json
import statistics
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from dev.model import ThreeHeadEncoder, multitask_loss


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Pilot requires CUDA')
    torch.manual_seed(17)
    output = Path('/content/jev-pilot')
    output.mkdir(exist_ok=True)
    report = {'purpose': 'throughput_and_memory_only', 'labels': 'synthetic_not_quality_evidence',
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'runs': []}
    models = [('answerdotai/ModernBERT-base', 0), ('Qwen/Qwen2.5-Coder-0.5B', 16)]
    if os.environ.get('PILOT_QWEN_ONLY') == '1':
        models = models[1:]
    for name, rank in models:
        model = ThreeHeadEncoder.pretrained(name).cuda()
        model.encoder.config.use_cache = False
        if rank:
            model.encoder = get_peft_model(model.encoder, LoraConfig(r=rank, lora_alpha=32,
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        tokenizer = AutoTokenizer.from_pretrained(name)
        source = 'Objective: diagnose test failure. Command: pytest. Output: FAILED tests/test_cache.py AssertionError expected 200 got 500. '
        tokens = tokenizer.encode(source, add_special_tokens=False)
        for length in [512, 2048]:
            ids = (tokens * (length//len(tokens)+1))[:length]
            ids[-1] = tokenizer.eos_token_id or ids[-1]
            batch = {'input_ids': torch.tensor([ids]*2, device='cuda'),
                     'attention_mask': torch.ones(2, length, dtype=torch.long, device='cuda'),
                     'candidate_mask': torch.zeros(2, 3, length, device='cuda'),
                     'candidate_valid': torch.ones(2, 3, dtype=torch.bool, device='cuda')}
            for c in range(3):
                batch['candidate_mask'][:, c, length-5+c] = 1
            labels = {'choice': torch.tensor([0, 1], device='cuda'),
                      'score': torch.tensor([2, 1], device='cuda'), 'noul': torch.tensor([1, 0], device='cuda')}
            torch.cuda.reset_peak_memory_stats()
            timings = []
            model.train()
            for step in range(8):
                torch.cuda.synchronize(); started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = multitask_loss(model(**batch), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite pilot loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                torch.cuda.synchronize()
                if step >= 2:
                    timings.append(time.perf_counter()-started)
            record = {'model': name, 'method': 'LoRA-16' if rank else 'full_finetune',
                      'length': length, 'batch_size': 2, 'precision': 'bf16_autocast_fp32_weights',
                      'attention': 'eager', 'optimizer_step_s': statistics.median(timings),
                      'examples_per_s': 2/statistics.median(timings),
                      'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            model.eval()
            inference = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for step in range(8):
                    torch.cuda.synchronize(); started = time.perf_counter()
                    model(**batch)
                    torch.cuda.synchronize()
                    if step >= 2:
                        inference.append(time.perf_counter()-started)
            record['batch_inference_s'] = statistics.median(inference)
            report['runs'].append(record)
            (output/'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(record), flush=True)
            del batch, labels, loss
        del model, optimizer, tokenizer
        gc.collect(); torch.cuda.empty_cache()
    print('PILOT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
