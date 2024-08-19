from time import time
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
import re
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID, label, feature = 'ID', 'Label', 'Description'
reasoning_types = ["PPR2", "PPR4"]
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model_dir = "text-classification-model/checkpoint-1380/"
exp_type = "MainExperiment"  # MultiHopExperiment, MainExperiment
reasoning_models = [
    "gpt-4o-mini-2024-07-18",
    "llama2-7b",
    "llama3-8b",
    "llama3-70b",
    "gpt-3.5-turbo-0125",
    "llama2-70b",
]

reasoning_datasets = {
    "webqsp": 1628,
    "CWQ": 3531,
}


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts=None, labels=None):
        self.texts = texts if texts is not None else []
        self.labels = labels if labels is not None else []
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length', max_length=128)
        item = {key: torch.tensor(val)
                for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def add_data(self, texts, labels):
        self.texts.extend(texts)
        self.labels.extend(labels)


def format_question(question):
    question = question[0].upper() + question[1:]
    question = question if question[-1] == "?" else question + "?"
    return question


def find_last_number(input_string):
    numbers = re.findall(r'\d+', input_string)
    return numbers[-1] if numbers else None


def run(reasoning_model, reasoning_dataset, sample_size):
    reasoning_model = reasoning_model.replace(":", "-")
    # log_file = f"./{reasoning_dataset}_{reasoning_model}.log"
    answer_file = f"{reasoning_model}_zero_shot_HopPred.json"
    save_dir = "feedback-model/"+reasoning_dataset+"/"+reasoning_model
    log_dir = '/tmp/logs/pred/'+reasoning_dataset+"/"+reasoning_model

    training_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_predict=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=1e-4,
        logging_dir=log_dir,
        disable_tqdm=True,
        # save_only_model=True,
        # load_best_model_at_end=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=2, id2label=id2label, label2id=label2id
    ).to(device)

    def inference(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                           padding='max_length', max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        return predictions.tolist()

    question_list = [dict(), dict()]
    ans_dicts = [dict(), dict()]
    reasoning_paths = [dict(), dict()]
    for index, reasoning_type in enumerate(reasoning_types):
        dataset_dir = f"dataset/AAAI/{exp_type}/{reasoning_dataset}/{reasoning_type}/"
        test_file = dataset_dir + answer_file
        json_info = json.load(open(test_file))
        for info in json_info["eval_info"]:
            qid = info["id"]
            ans_dicts[index][qid] = info
            question_list[index][qid] = format_question(info["question"])
            reasoning_paths[index][qid] = info["ReasoningPaths"]

    questions = {qid: question_list[0][qid]
                 for qid in question_list[0] if qid in question_list[1]}

    print(f"model: {reasoning_model}")
    print(f"reasoning_dataset: {reasoning_dataset}")
    print(f"sample_size: {sample_size}")

    questions_df = pd.DataFrame(columns=[ID, feature, 'llm_index'])
    random_questions = random.sample(list(questions.items()), len(questions))
    for qid, question in random_questions:
        if len(questions_df) < sample_size:
            predictions = inference([question])
            pred_index = predictions[0]
            llm_pred = ans_dicts[pred_index][qid]["HopPredOutput"]
            num = llm_pred.count("->") // 2
            if num not in [1, 2, 3, 4]:
                continue
            questions_df = questions_df._append(
                {ID: qid, feature: question, 'llm_index': 1 if num > 2 else 0}, ignore_index=True)
            if len(questions_df) == sample_size:
                dataset = SentimentDataset(tokenizer, questions_df[feature].tolist(
                ), questions_df['llm_index'].tolist())
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                )
                trainer.train()
                break

    # ckpt_dirs = os.listdir(save_dir)
    # ft_model_dir = os.path.join(save_dir, ckpt_dirs[-1])
    classifier = pipeline("sentiment-analysis", tokenizer=tokenizer,
                          model=model, device=device)
    predictions = classifier(list(questions.values()))
    predictions = [label2id[pred['label']] for pred in predictions]

    json_info = {
        "basic_info": {
            "Dataset": reasoning_dataset,
            "llm": reasoning_model
        },
        "eval_info": [
        ]
    }

    hits = 0
    for qid, pred_index in zip(questions.keys(), predictions):
        hits += ans_dicts[pred_index][qid]["HR"]
        json_info["eval_info"].append(ans_dicts[pred_index][qid])
    print(hits, "/", len(questions))
    print(hits/len(questions)*100)
    print()
    output_path = f"dataset/AAAI/{exp_type}/{reasoning_dataset}/{reasoning_model.replace(':','-')}_feedback_answers.json"
    json.dump(json_info, open(output_path, "w"))

    return hits


if __name__ == "__main__":
    random.seed(0)

    for reasoning_dataset, length in reasoning_datasets.items():
        samples = [length//4]

        for reasoning_model in reasoning_models:
            max_hit = 0
            max_size = 0
            try:
                for sample_size in samples:
                    cur_hit = run(reasoning_model,
                                  reasoning_dataset, sample_size)
                    if cur_hit > max_hit:
                        max_hit = cur_hit
                        max_size = sample_size
            except Exception as e:
                print(e)
                continue
            print("------")
            print("FindBest:", reasoning_dataset,
                  reasoning_model, max_hit, max_size)
            print("------")
            print()
