# softmind_aio

## Qwen3-1.7B Continual Pretraining

This project contains code for continual pretraining of Qwen3-1.7B model on legal corpus data.

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download datasets:
   ```bash
   gdown 1OnxJ_UeJ_YXRX0E1U7lBIxqI9phaI6wq
   gdown 1GatkZT0nepRMC0G2lUxofP_9yKThwVlC
   ```

3. Set Hugging Face token (required for model upload):
   ```bash
   export HF_TOKEN=your_huggingface_token
   ```

4. Run training:
   ```bash
   python track5_1_7b_continue_pretraining.py
   python finetune_track5.py
   ```

### Files

- `track5_1_7b_continue_pretraining.py` - Continual pretraining script
- `finetune_track5.py` - Supervised fine-tuning script
- `requirements.txt` - Required dependencies
- `track5-1-7b-countinue-pretraining.ipynb` - Original continual pretraining notebook
- `finetune_track5.ipynb` - Original fine-tuning notebook
