# Model paths and names
LOCAL_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
LORA_PATH = "../lora/"
DATA_PATH = "../data/tmp/"
OUTPUT_PATH="../outputs/"

# Training parameters
MAX_SEQ_LENGTH = 2048
RANK = 64
MAX_ITER_STEPS = 100
EPOCHS = 0
SAMPLE_LEN="25k"

# Kaggle upload configuration
MODEL_SLUG = "qwen3-4b-jigsaw-acrc-lora---"
VARIATION_SLUG = "06"

###--------------------------------###
DATASET_ID="000"
BASE_MODEL=LOCAL_MODEL_PATH.split("/")[-1].replace(".", "p")
TRAIN_DIR=f"{BASE_MODEL}_lora_fp16_r{RANK}_s{SAMPLE_LEN}_e_{EPOCHS}_msl{MAX_SEQ_LENGTH}-{DATASET_ID}"
print("TRAIN_DIR",TRAIN_DIR)