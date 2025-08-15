# steps extracted from 
# https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_llama/tutorial_accelerate_hf_llama_with_te.ipynb
# utils.py and te_llama.py are copied from v2.5

# Import necessary packages, methods and variables
from utils import *


# Provide Huggingface Access Token
hyperparams.hf_access_token = ""
assert hyperparams.hf_access_token, "Provide a HF API Access Token!"

# Provide a directory to cache weights in to avoid downloading them every time.
# (By default, weights are cached in `~/.cache/huggingface/hub/models`)
# hyperparams.weights_cache_dir = ""

# For Llama 2, uncomment this line (also set by default)
hyperparams.model_name = "meta-llama/Llama-2-7b-hf"

# For Llama 3, uncomment this line
# hyperparams.model_name = "meta-llama/Meta-Llama-3-8B"

hyperparams.mixed_precision = "bf16"


# Init the model and accelerator wrapper
model = init_te_llama_model(hyperparams)

if False:
    accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)

    # Finetune the model
    finetune_model(model, hyperparams, accelerator, train_dataloader, optimizer, lr_scheduler)
