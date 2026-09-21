import json
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import load_file
from .model import ThreeHeadEncoder
from .data import SCHEMA_VERSION, encode, collate
from .train import device_for, move


class Predictor:
    def __init__(self, checkpoint, device='auto'):
        path = Path(checkpoint)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(repo_id=str(checkpoint)))
        self.metadata = json.loads((path / 'run.json').read_text())
        if self.metadata.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Checkpoint predates question-conditioned schema v2; retraining is required')
        self.device = device_for(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path / 'tokenizer', local_files_only=True)
        config = AutoConfig.from_pretrained(path / 'encoder', local_files_only=True)
        if self.metadata.get('lora_rank'):
            from peft import PeftModel
            base = AutoModel.from_pretrained(self.metadata['model'], revision=self.metadata.get('revision'), attn_implementation='sdpa')
            self.model = ThreeHeadEncoder(PeftModel.from_pretrained(base, path / 'adapter'))
            missing, unexpected = self.model.load_state_dict(load_file(path / 'heads.safetensors'), strict=False)
            if unexpected or any(not k.startswith('encoder.') for k in missing):
                raise ValueError('Incomplete head checkpoint')
        else:
            self.model = ThreeHeadEncoder(AutoModel.from_config(config, attn_implementation='sdpa'))
            self.model.load_state_dict(load_file(path / 'model.safetensors'), strict=False)
        self.model.to(self.device).eval()
        self.max_length = self.metadata['max_length']
        # Per-readout temperature scaling (Guo et al. 2017), fit post-hoc by calibrate_temperature.py.
        # Absent key => T=1.0 (no-op). Keys match q['type']: 'noul' | 'choice' | 'score'.
        self.temperatures = self.metadata.get('temperatures') or {}

    @torch.inference_mode()
    def answer(self, state, questions, command='', batch_size=4):
        """Batch independent question/state pairs.

        Per-readout temperature scaling is applied to the logits when the checkpoint
        carries fitted temperatures; a readout with no fitted temperature is returned
        uncalibrated (answer['calibrated'] is per-readout).

        Caller IDs are routing keys only, never model inputs. Evidence selection and
        whole-output coverage checks belong to the wrapper, not this raw classifier.
        """
        if not isinstance(questions, dict) or not questions or batch_size < 1:
            raise ValueError('Require nonempty questions mapping and positive batch_size')
        items = list(questions.items())
        answers = {}
        for start in range(0, len(items), batch_size):
            part = items[start:start + batch_size]
            rows = [encode({'schema_version': SCHEMA_VERSION, 'group': 'inference', 'state': state,
                            'command': command, 'question': q}, self.tokenizer, self.max_length) for _, q in part]
            result = self.model(**move(collate(rows, self.tokenizer.pad_token_id), self.device))
            for i, (qid, q) in enumerate(part):
                kind = q['type']
                temperature = self.temperatures.get(kind, 1.0)
                # New (temperature) scheme: per-readout truth by presence of a fitted T.
                # Legacy checkpoints carry no `temperatures` block, so fall back to the
                # run.json status flag (keeps older Brier/SFT checkpoints reporting as before).
                calibrated = (kind in self.temperatures) if self.temperatures \
                    else self.metadata.get('status') == 'calibrated'
                answer = {'type': kind, 'calibrated': calibrated}
                if kind == 'noul':
                    p_yes = (result[kind][i] / temperature).sigmoid().item()
                    answer['probability_yes'] = p_yes
                    answer['confidence'] = max(p_yes, 1.0 - p_yes)
                else:
                    probabilities = (result[kind][i, :len(q['criteria'])] / temperature).softmax(-1).cpu().tolist()
                    answer['probabilities'] = probabilities
                    if kind == 'choice':
                        idx = max(range(len(probabilities)), key=probabilities.__getitem__)
                        answer['index'] = idx
                        answer['criterion'] = q['criteria'][idx]
                        answer['confidence'] = probabilities[idx]
                    else:
                        mu = sum(j * p for j, p in enumerate(probabilities))
                        var = sum((j - mu) ** 2 * p for j, p in enumerate(probabilities))
                        max_var = ((len(probabilities) - 1) / 2.0) ** 2
                        answer['value'] = mu
                        answer['variance'] = var
                        answer['confidence'] = max(0.0, min(1.0, 1.0 - (var / max_var))) if max_var > 0 else 1.0
                answers[qid] = answer
        return answers

    def predict(self, objective, state, options):
        """Compatibility adapter for the experimental, default-passthrough MCP shim."""
        answers = self.answer(state, {
            'choice': {'type': 'choice', 'instructions': objective, 'criteria': options},
            'score': {'type': 'score', 'instructions': f'How useful is this evidence for: {objective}',
                      'criteria': ['Unrelated', 'Background only', 'Some useful context', 'Direct supporting evidence', 'Indispensable evidence']},
            'noul': {'type': 'noul', 'instructions': f'Does this contain evidence that must be preserved verbatim for: {objective}',
                     'criteria': ['No', 'Yes']},
        })
        return {'choice': answers['choice'], 'score': answers['score'],
                'noul': answers['noul']['probability_yes'],
                'calibrated': self.metadata.get('status') == 'calibrated'}
