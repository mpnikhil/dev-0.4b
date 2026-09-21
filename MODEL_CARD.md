---
language:
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
tags:
- decision-model
- modernbert
- routing
- classification
- verification
- brier-score
- calibration
- bidirectional
datasets:
- PolyAI/banking77
- google/boolq
- code_search_net
- yelp_review_full
metrics:
- accuracy
- brier_score
- ece
- ndcg
model-index:
- name: dev-0.4b
  results:
  - task:
      type: text-classification
      name: 77-Way Intent Routing
    dataset:
      name: Banking77
      type: PolyAI/banking77
    metrics:
    - type: accuracy
      value: 0.9133
      name: Top-1 Accuracy
    - type: recall_at_3
      value: 0.9867
      name: Top-3 Recall
  - task:
      type: text-classification
      name: Boolean Verification
    dataset:
      name: Google BoolQ
      type: google/boolq
    metrics:
    - type: accuracy
      value: 0.8520
      name: Accuracy
  - task:
      type: text-classification
      name: 5-Star Graded Scoring
    dataset:
      name: Yelp Review Full
      type: yelp_review_full
    metrics:
    - type: accuracy
      value: 0.6267
      name: Exact Accuracy
---

# Dev (`dev-0.4b`): 399M Bidirectional Decision Model

**Dev** is an open-source 399M parameter bidirectional decision model built on `ModernBERT-large`. It is purpose-built for **unstructured-to-structured classification**—including ticket routing, yes/no verification, and rating scales—executing in a single forward pass (~28ms on MPS) without token generation.

Following **Jev** (TypeSafe) and **Kev-0.5B** (Jared Palmer), Dev tests a fundamental architectural question: *What if decision models shouldn't be causal decoders at all, but native bidirectional cross-encoders?*

---

## Key Performance Results

Evaluated against Jared Palmer's `kev-0.5b` (built on a frozen Qwen-2.5-0.5B causal backbone + 9.3M LoRA pointer head):

| Benchmark / Task | What It Tests | Kev-0.5B (Causal Qwen2.5) | Dev-0.4B (Bidirectional ModernBERT) | Result |
| :--- | :--- | :---: | :---: | :--- |
| **MTEB Banking77** | 77-Way Intent Routing | 86.0% | **91.33%** *(Top-3: 98.67%)* | 🏆 **+5.33% Win** |
| **Google BoolQ** | Reading Verification | 75.3% | **85.20%** | 🏆 **+9.90% Win** |
| **Yelp Reviews** | 5-Star Rating (Exact / MAE) | 55.3% | **62.67%** *(MAE: 0.4017)* | 🏆 **+7.37% Win** |

*Inference Latency: Dev-0.4B executes in **27.6ms** on Apple Silicon MPS (M1 Max) and **~10ms** on CUDA FP16 SDPA (single forward pass, zero token generation loops).*

Dev also reranks Python code retrieval modestly above a BM25 lexical baseline on the CodeSearchNet human-judgment benchmark (0.8203 vs 0.7652 NDCG@10 on test). We treat this as a capability check, not a headline benchmark.

---

## Core Architecture

1. **Native Bidirectional Attention (`ModernBERT-large`)**:
   Instead of causal decoders with lower-triangular masks, Dev uses unconstrained bidirectional cross-attention across all 28 transformer layers. 
   - When choices are listed on the token tape, Option 1 can attend forward to Option 4, enabling true mutual candidate conditioning and eliminating position/recency bias.
   - Runs on hardware-fused SDPA kernels (`mask=None`) on CUDA and Apple Silicon MPS.
   - Native 8k context window.

2. **Three Tasks, One Dynamic Head**:
   Instead of separate heads for classification, verification, and regression, Dev realizes that **all three tasks are fundamentally classification**:
   - **Categories (Routing):** Classification over $N$ candidate options in the prompt.
   - **Yes / No:** Classification over two options: `["No", "Yes"]`.
   - **Rating (1–5 scale):** Classification over ordered scale levels (`["1 star", ..., "5 stars"]`), taking the expected value.

3. **Dynamic Choices via the GLiNER Mechanism**:
   Borrowing the core insight from GLiNER, candidate choices are not hardcoded into neural network weights. They are written as natural language text directly inside the prompt. Dev's single 2-layer choice head evaluates whatever choices you provide on the fly.

4. **Single Forward Pass**:
   The document and all candidate options are evaluated together in **one single forward pass** (~28ms on MPS, ~10ms on CUDA), rather than running separate passes per option.

---

## Post-Training Calibration: Temperature Scaling (Guo et al. 2017)

Cross-Entropy loss separates classes effectively, but its logarithmic tail pushes logits toward extreme values (±infinity), producing overconfidence. While boolean verification comes out of SFT essentially calibrated (ECE: 0.016), multi-class choice and ordinal scoring are significantly overconfident.

Because Dev uses a **single universal choice head**, fine-tuning the shared head under Brier loss creates cross-task gradient tension and vanishing gradients ($2(p - y) \cdot p(1 - p) \to 0$).

Instead, Dev applies **per-readout temperature scaling** (Guo et al., 2017) fit post-hoc on held-out validation data by minimizing NLL:

| Readout | Fitted T | Validation ECE (equal-mass) | NLL | Top-1 Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Noul** (boolean) | **1.3575** | 0.016 *(already low here)* | 0.17 → 0.15 | 0.967 → 0.967 (Invariant) |
| **Choice** (categorical) | **4.2542** | **0.189 → 0.083** | 2.26 → 0.73 | 0.782 → 0.782 (Invariant) |
| **Score** (ordinal) | **3.7097** | **0.332 → 0.117** | 2.59 → 1.14 | 0.545 → 0.545 (Invariant) |

On the **external, unseen benchmarks**, the shipped temperatures generalize, accuracy exactly invariant:

| Benchmark (readout) | ECE: raw → calibrated | Accuracy |
| :--- | :---: | :---: |
| Google BoolQ (noul, n=500) | 0.103 → **0.077** (−25%) | 0.852 → 0.852 |
| Banking77 (choice, n=300) | 0.075 → **0.055** (−27%) | 0.913 → 0.913 |
| Yelp (score, n=300) | 0.318 → **0.155** (−51%) | exact 0.627 → 0.627 |

- **Choice & Score** were badly overconfident out of SFT; their ECE falls by half or more.
- **Boolean** looked already-calibrated on the validation set (0.016) but is overconfident on external BoolQ; its fitted T = 1.36 cuts BoolQ ECE 0.103 → 0.077. The temperatures are fit on validation and checked on the held-out benchmarks.
- **Ordinal point estimate**: flattening the score distribution raises Yelp MAE modestly (0.402 → 0.429, still sub-half-star). ECE and MAE trade off smoothly as T grows.
- **100% Accuracy Invariance**: temperature scaling is strictly monotonic, so all top-1 accuracies, rankings, and benchmark scores are untouched.

Temperatures are saved in `run.json` and automatically applied at readout during inference.

---

## Quickstart & Usage

### Installation

```bash
git clone https://github.com/mpnikhil/dev-0.4b.git
cd dev-0.4b
pip install -e .
```

### Python Inference

```python
from dev.inference import Predictor

# Load from Hugging Face Hub or local directory ("runs/dev-0.4b")
predictor = Predictor("mpnikhil/dev-0.4b", device="auto")

# 1. Routing to Categories (~28ms MPS, Calibrated)
result = predictor.answer(
    state="Customer cannot log in. Password reset email is failing with 550 Mailbox Unavailable.",
    questions={
        "route_ticket": {
            "type": "choice",
            "instructions": "Assign this ticket to the appropriate queue.",
            "criteria": [
                "billing_support",
                "email_infrastructure",
                "account_security",
                "general_inquiry"
            ]
        }
    }
)
print(result["route_ticket"])
# Output:
# {
#   'criterion': 'email_infrastructure',
#   'confidence': 0.9987,
#   'calibrated': True
# }

# 2. Yes / No Verification
res_noul = predictor.answer(
    state="ModernBERT uses hardware-fused SDPA kernels and 8k context natively.",
    questions={
        "has_8k": {
            "type": "noul",
            "instructions": "Does the passage state ModernBERT supports 8k context?",
            "criteria": ["No", "Yes"]
        }
    }
)
print(res_noul["has_8k"])
# Output:
# {
#   'criterion': 'Yes',
#   'probability_yes': 0.9942,
#   'calibrated': True
# }

# 3. Rating on an Ordinal Scale (Expected Value + Normalized Variance)
score_result = predictor.answer(
    state="Pull request refactors cache layer, adds 14 unit tests, and passes all CI checks.",
    questions={
        "code_quality": {
            "type": "score",
            "instructions": "Rate pull request quality on a 1-5 scale.",
            "criteria": ["Poor", "Needs Work", "Acceptable", "Good", "Excellent"]
        }
    }
)
print(score_result["code_quality"])
# Output:
# {
#   'value': 3.84,           # Expected score
#   'variance': 0.14,        # Clustered consensus
#   'confidence': 0.965,     # Normalized certitude
#   'calibrated': True
# }
```

---

## Intended Use & Limitations

- **Intended Use**: High-throughput, low-latency unstructured-to-structured classification (support ticket routing, log triage, assertions, guardrail verification, rubric rating).
- **Limitations**: Dev is a non-generative decision model. It does not output autoregressive text, stream tokens, or perform Chain-of-Thought (CoT) generative reasoning. For tasks requiring reasoning traces or text generation, use a causal generative LLM.

---

## Lineage & Acknowledgments

- **TypeSafe**: For introducing **Jev** and demonstrating the power of non-generative decision models.
- **Archer Hume**: For reverse-engineering Jev's behavioral blueprint across 10,000 API calls in [“Jev’s Architecture Unmasked”](https://archerhume.com/posts/jevs-architecture-unmasked/).
- **Jared Palmer**: For open-sourcing [**Kev-0.5B**](https://huggingface.co/jaredpalmer/kev-0.5b) on Hugging Face, establishing the single-pass causal decoder baseline.
- **Answer.AI & LightOn**: For pretraining [**ModernBERT**](https://huggingface.co/answerdotai/ModernBERT-large) (`answerdotai/ModernBERT-large`), the 399M parameter bidirectional encoder backbone.
- **Urchade Zaratiana et al.**: For [**GLiNER**](https://huggingface.co/urchade/gliner_base), whose candidate-span pooling mechanism directly inspired our dynamic choice head.

---

## Citations & References

### Foundational Architectures & Calibration

```bibtex
@article{vaswani2017attention,
  title={Attention is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017},
  url={https://arxiv.org/abs/1706.03762}
}

@article{warner2024modernbert,
  title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Long-Context Representation},
  author={Warner, Benjamin and Chaffin, Antoine and Clavi{\'e}, Benjamin and Weller, Orion and Hallstr{\"o}m, Oskar and Taghadouei, Saeed and Aarsen, Tom and Shakir, Nathan and Douze, Matthijs and Lipani, Aldo and others},
  journal={arXiv preprint arXiv:2412.13663},
  year={2024},
  url={https://arxiv.org/abs/2412.13663}
}

@article{zaratiana2023gliner,
  title={GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer},
  author={Zaratiana, Urchade and Tomeh, Nadi and Holat, Pierre and Chaffin, Antoine},
  journal={arXiv preprint arXiv:2311.01079},
  year={2023},
  url={https://arxiv.org/abs/2311.01079}
}

@inproceedings{guo2017calibration,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q},
  booktitle={International Conference on Machine Learning},
  pages={1321--1330},
  year={2017},
  organization={PMLR},
  url={https://arxiv.org/abs/1706.04599}
}
```

### Datasets

- **PolyAI/banking77**: Casanueva, I. et al. (2020). *Efficient Intent Detection with Dual Sentence Encoders Applications to Banking*. [Hugging Face Dataset](https://huggingface.co/datasets/PolyAI/banking77).
- **google/boolq**: Clark, C. et al. (2019). *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions*. [Hugging Face Dataset](https://huggingface.co/datasets/google/boolq).
- **code_search_net**: Husain, H. et al. (2019). *CodeSearchNet Challenge: Evaluating the State of Semantic Code Search*. [Hugging Face Dataset](https://huggingface.co/datasets/code_search_net).
- **yelp_review_full**: Zhang, X. et al. (2015). *Character-level Convolutional Networks for Text Classification*. [Hugging Face Dataset](https://huggingface.co/datasets/yelp_review_full).

---

## License

Apache 2.0
