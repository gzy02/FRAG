
############################################
# region os env
import os
with open('.env', 'r') as f:
    for line in f:
        try:
            key, value = line.strip().split('=')
            os.environ[key] = value
        except:
            break
# endregion

############################################
# region LLM config
ollama_models = [
    "llama2:7b-chat-q5_K_M",
    "llama3:70b-instruct-q5_K_M",
    "llama3:8b-instruct-q5_K_M",
    "llama2:70b-chat-q5_K_M"
]

commercial_models = [
    "gpt-3.5-turbo-0125",
    "gpt-4o-mini-2024-07-18",
]

local_models = {
    "llama3-8b": "/back-up/LLMs/llama3/Meta-Llama-3-8B-Instruct/",
    "llama3-70b": "/back-up/LLMs/llama3/Meta-Llama-3-70B-Instruct/",
    "llama2-7b": "/back-up/LLMs/models/Llama-2-7b-chat-hf/",
    "llama2-70b": "/back-up/LLMs/Llama-2-70b-chat-ms/",
    "llama2-13b": "/back-up/LLMs/models/Llama-2-13b-chat-hf/",
}

reasoning_model = "llama3-8b"
hop_pred_model = reasoning_model
max_reasoning_paths = 64
tensor_parallel_size = 1
gpu_memory_utilization = 0.4
temperature = 0.01
max_tokens = 256
stop_tokens = ["\n", "<|eot_id|>", "</s>"]
quantization = None  # "fp8"
dtype = "auto"  # "bfloat16"
enforce_eager = True
# endregion

############################################
# region Datasets config
supported_datasets = ["CWQ", "webqsp",]

reasoning_dataset = "CWQ"
reasoning_type = "PPR2"  # PPR4
experiment_type = "MainExperiment"

dataset_dir = f"/back-up/gzy/dataset/AAAI/{experiment_type}/{reasoning_dataset}/{reasoning_type}/"
test_file = dataset_dir + "test_name.jsonl"
# endregion

############################################
# region SentenceModel config
emb_model_dir = "sentence-transformers/all-MiniLM-L6-v2"
rerank_model_dir = "BAAI/bge-reranker-v2-m3"

# endregion

############################################
# region Evaluation config
hr_top_k = 10
# endregion
