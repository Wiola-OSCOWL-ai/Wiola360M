# Wiola-360M

Wiola-360M is a 360 million parameter causal language model based on the **Wiola Architecture**, designed for efficient language modeling while maintaining strong performance at a relatively small parameter count. The model is implemented in PyTorch and follows the Hugging Face Transformers interface.

The architecture introduces several efficiency-oriented components including Gated Cross-Layer Attention (GCLA), Dual Stream Feed Forward (DSFF), Adaptive Token Merging (ATM), and Spiral Rotary Positional Encoding (SRPE).

This repository contains the pretrained Wiola-360M model weights.

---

## Model Details

| Property | Value |
|----------|-------|
| Model | Wiola-360M |
| Parameters | ~360 Million |
| Architecture | Decoder-only Transformer |
| Framework | PyTorch |
| Library | Hugging Face Transformers |
| License | MIT |
| Organization | OSCOWL AI |

---

## Architecture

Wiola introduces several architectural improvements over a conventional decoder-only transformer.

### Gated Cross-Layer Attention (GCLA)

Allows each decoder layer to access compressed contextual information from previous layers through learnable gating.

### Dual Stream Feed Forward (DSFF)

Each transformer block contains both narrow and wide feed-forward streams whose outputs are combined for improved efficiency.

### Adaptive Token Merging (ATM)

Adjacent redundant tokens are merged during training to reduce unnecessary computation while preserving semantic information.

### Spiral Rotary Positional Encoding (SRPE)

An extension of rotary positional embeddings that introduces a spiral radial component for richer positional representations.

---

## Files

The repository contains:

```
model.safetensors
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
LICENSE
README.md
```

---

## Installation

Create a Python environment and install the required libraries.

```bash
pip install torch transformers sentencepiece safetensors
```

---

## Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Wiola-OSCOWL-ai/Wiola360M"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)
```

---

## Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Wiola-OSCOWL-ai/Wiola360M"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "Explain why transformers are effective language models."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Training

Wiola-360M was trained as a causal language model using next-token prediction.

Training features include:

- Mixed precision training
- AdamW optimizer
- Cosine learning rate schedule
- Gradient accumulation
- Checkpoint saving
- Validation-based best checkpoint selection

---

## Intended Uses

Wiola-360M is intended for

- Research
- Language model experimentation
- Text generation
- Educational purposes
- Benchmarking efficient transformer architectures

---

## Limitations

As with all language models,

- outputs may contain factual inaccuracies,
- responses should not be treated as professional advice,
- generated text may reflect biases present in the training corpus.

Human verification is recommended for important applications.

---

## Citation

If you use Wiola-360M in your research, please cite the Wiola paper.

```bibtex
@misc{chowdhury2026wiolaarchitectureefficientsmall,
      title={The Wiola Architecture for Efficient Small Language Models},
      author={Aryuemaan Kumar Chowdhury and Afreen Shaik and Yaparla Bhargavi and Brahma Kumar},
      year={2026},
      eprint={2607.01394},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.01394},
}
```

---

## Copyright

Copyright (c) 2026 OSCOWL AI

The Wiola architecture, pretrained models, implementation, documentation, and associated materials are Copyright © 2026 OSCOWL AI.

Permission is granted under the MIT License contained in this repository.

---

## Authors

**Aryuemaan Kumar Chowdhury**

OSCOWL AI

with contributions from

- Afreen Shaik
- Yaparla Bhargavi
- Brahma Kumar

---

## License

This project is released under the MIT License.

See the LICENSE file for details.
