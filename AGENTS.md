# AGENTS.md - Guide for Agentic Coding Agents

## Build / Lint / Test Commands

### Installation

```bash
# Install from source (editable mode)
pip install -e .

# Install from PyPI
pip install -U qwen-tts

# Optional: FlashAttention 2 for reduced GPU memory
pip install -U flash-attn --no-build-isolation
# For machines with <96GB RAM:
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

### Running Tests

The project uses example scripts as tests. These live in the `examples/` directory:

```bash
# Run specific test scripts directly
python examples/test_model_12hz_base.py
python examples/test_model_12hz_custom_voice.py
python examples/test_model_12hz_voice_design.py
python examples/test_tokenizer_12hz.py

# Run a single test script (pattern)
python examples/<test_file>.py

# Launch web UI demo
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
qwen-tts-demo --help
```

### Fine-tuning

```bash
cd finetuning

# Prepare data for training
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# Run fine-tuning
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_name
```

## Code Style Guidelines

### File Headers

Every source file must start with the Apache 2.0 license header and encoding comment:

```python
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### Import Organization

Order imports: standard library → third-party → local. Group third-party imports by library.

```python
import base64
import os
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModel

from ..core.models import Qwen3TTSConfig
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `Qwen3TTSModel`, `VoiceClonePromptItem`)
- **Functions and variables**: snake_case (e.g., `generate_custom_voice`, `ref_audio`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_JOBS`)
- **Private members**: underscore prefix (e.g., `_maybe`, `_title_case_display`)

### Type Hints

Always use type hints for function signatures and important return types:

```python
from typing import List, Optional, Tuple, Union

def generate_custom_voice(
    text: Union[str, List[str]],
    language: Union[str, List[str]],
    speaker: Optional[str] = None,
) -> Tuple[List[np.ndarray], int]:
    ...
```

Audio input type alias is used extensively:
```python
AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]
```

### Error Handling

Use descriptive error messages and raise appropriate exception types:

```python
if audio_input is None:
    raise ValueError("audio_input cannot be None")

if sr <= 0:
    raise ValueError(f"Invalid sample rate: {sr}. Must be positive.")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")
```

### Docstrings

Use Google/NumPy-style docstrings:

```python
class Qwen3TTSModel:
    """
    A HuggingFace-style wrapper for Qwen3 TTS models.

    Args:
        model: The underlying Qwen3TTSForConditionalGeneration model.
        processor: The model processor.
        generate_defaults: Optional default generation parameters.

    Returns:
        Tuple[List[np.ndarray], int]: List of waveforms and sample rate.
    """
```

### Device and Dtype Handling

Models commonly support device_map and dtype arguments:

```python
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

Supported dtypes: `torch.bfloat16`, `torch.float16`, `torch.float32`.

### Audio Processing

- Use `soundfile` (sf) for WAV I/O
- Use `librosa` for audio loading/resampling
- Return audio as `float32` numpy arrays
- Always return sample rate alongside waveforms:

```python
import soundfile as sf
sf.write("output.wav", waveform, sample_rate)

wavs, sr = model.generate_custom_voice(text="Hello", language="English")
```

### Package Structure

- `qwen_tts/core/` - Core model and tokenizer implementations
- `qwen_tts/inference/` - High-level API wrappers (Qwen3TTSModel, Qwen3TTSTokenizer)
- `qwen_tts/cli/` - Command-line interface tools
- `examples/` - Example usage and test scripts
- `finetuning/` - Fine-tuning utilities

### Model Loading

Use HuggingFace AutoModel/AutoProcessor pattern for consistency:

```python
from transformers import AutoConfig, AutoModel, AutoProcessor

config = AutoConfig.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, **kwargs)
processor = AutoProcessor.from_pretrained(model_path)
```

### Key Public APIs

- `Qwen3TTSModel.from_pretrained()` - Load TTS models
- `Qwen3TTSTokenizer.from_pretrained()` - Load tokenizer
- `model.generate_custom_voice()` - Custom voice generation
- `model.generate_voice_design()` - Voice design
- `model.generate_voice_clone()` - Voice cloning
- `model.create_voice_clone_prompt()` - Build reusable clone prompts
- `tokenizer.encode()` / `tokenizer.decode()` - Audio codec operations

### Python Version

Minimum Python version is 3.10 (supports 3.10, 3.11, 3.12, 3.13). Use type hints compatible with these versions.