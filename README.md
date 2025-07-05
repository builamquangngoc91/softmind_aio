# softmind_aio

## Qwen3-1.7B Continual Pretraining

This project contains code for continual pretraining of Qwen3-1.7B model on legal corpus data.

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download dataset:
   ```bash
   gdown 1OnxJ_UeJ_YXRX0E1U7lBIxqI9phaI6wq
   ```

3. Set Hugging Face token (optional, for model upload):
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```

4. Run training:
   ```bash
   python track5_1_7b_continue_pretraining.py
   ```

### Files

- `track5_1_7b_continue_pretraining.py` - Main training script
- `requirements.txt` - Required dependencies
- `track5-1-7b-countinue-pretraining.ipynb` - Original Jupyter notebook
