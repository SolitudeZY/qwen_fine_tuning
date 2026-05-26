#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen35-ft

python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.version.cuda)"
python -c "import torchvision; from torchvision.ops import nms; print('torchvision nms ok')"
python -c "from transformers import enable_full_determinism; print('transformers ok')"
python -c "from peft import PeftModel; print('peft ok')"
python -c "import swift; print('swift ok')"
python -c "from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn; print('causal_conv1d ok')"