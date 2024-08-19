import pandas as pd
from transformers import pipeline
import torch
import json
from time import time
torch.set_float32_matmul_precision('high')


def format_question(question):
    question = question[0].upper()+question[1:]
    question = question if question[-1] == "?" else question+"?"
    return question


model_path = "text-classification-model/checkpoint-1380/"
reasoning_types = ["PPR2", "PPR4"]
exp_type = "MainExperiment"  # MultiHopExperiment, MainExperiment
reasoning_models = [
    "gpt-4o-mini-2024-07-18",
    "llama2-7b",
    "llama3-8b",
    "llama3-70b",
    "gpt-3.5-turbo-0125",
    "llama2-70b",
]
reasoning_datasets = [
    "webqsp",
    "CWQ",
]


def run(reasoning_dataset, reasoning_model):
    answer_file = f"{reasoning_model.replace(':','-')}_zero_shot_answers.json"
    output_path = f"dataset/AAAI/{exp_type}/{reasoning_dataset}/{reasoning_model.replace(':','-')}_nofeedback_answers.json"

    print(reasoning_dataset)
    print(answer_file)
    question_list = [dict(), dict()]
    ans_dicts = [dict(), dict()]

    for index, reasoning_type in enumerate(reasoning_types):
        dataset_dir = f"dataset/AAAI/{exp_type}/{reasoning_dataset}/{reasoning_type}/"

        test_file = dataset_dir + answer_file
        json_info = json.load(open(test_file))
        for info in json_info["eval_info"]:
            qid = info["id"]
            ans_dicts[index][qid] = info
            question_list[index][qid] = format_question(info["question"])
    classifier = pipeline("sentiment-analysis",
                          model=model_path, device="cuda")
    questions = {qid: question_list[0][qid]
                 for qid in question_list[0] if qid in question_list[1]}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    t = time()
    predictions = classifier(list(questions.values()))
    print(time()-t)
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
        json_info["eval_info"].append(ans_dicts[pred_index][qid])
        hits += ans_dicts[pred_index][qid]["HR"]
    print(hits, "/", len(questions))
    print()
    json.dump(json_info, open(output_path, "w"))


if __name__ == "__main__":
    for reasoning_dataset in reasoning_datasets:
        for reasoning_model in reasoning_models:
            try:
                run(reasoning_dataset, reasoning_model)
            except:
                print("No Found")
