from config import test_file, reasoning_dataset, reasoning_model, dataset_dir
from utils.Tools import get_query
import json
from utils.LLM import LLM
from utils.Evaluation import  eval_f1,  eval_hit
from utils.PromptTemplate import REASONING_INPUT

if __name__ == "__main__":
    print("Dataset:", reasoning_dataset)
    print(test_file)
    print("Reasoning Model:", reasoning_model)

    path_file = f"{reasoning_dataset}_paths.json"
    print(path_file)
    reasoning_paths_file = dataset_dir + path_file
    json_info = json.load(open(reasoning_paths_file))
    basic_info = json_info["basic_info"]
    basic_info["llm"] = reasoning_model
    eval_info = []
    reason_infos = {info["id"]: info for info in json_info["eval_info"]}
    reasoning_llm = LLM(reasoning_model)
    sample = 0
    hits = 0
    questions = []
    for query in get_query(test_file):
        sample += 1
        info = reason_infos[query.qid]
        reasoning_paths = info["ReasoningPaths"]
        llm_question = REASONING_INPUT.format(
            paths=reasoning_paths, question=query.question)
        questions.append(llm_question)
    answers = reasoning_llm.batch_invoke(questions)
    for query in get_query(test_file):
        info = reason_infos[query.qid]

        print("Question: ", query.question)
        print("Answers:", query.answers)
        llm_answer = answers.pop(0)
        print("LLM Answer:", llm_answer)
        info["llm_answer"] = llm_answer

        f1, acc, recall = eval_f1([llm_answer], query.answers)
        hit = eval_hit(llm_answer, query.answers)
        print("Eval-Reasoning: ACC =", acc)
        print("Eval-Reasoning: F1 =", f1)
        print("Eval-Reasoning: Recall =", recall)
        print("Eval-Reasoning: HR =", hit)
        info["ACC"] = acc
        info["F1"] = f1
        info["Recall"] = recall
        info["HR"] = hit
        hits += hit

        eval_info.append(info)

    print(hits, "/", sample)
    answers_file = f"{dataset_dir}{reasoning_model.replace(':','-')}_zero_shot_answers.json"
    with open(answers_file, "w") as fp:
        json.dump(
            {
                "basic_info": basic_info,
                "eval_info": eval_info
            }, fp, indent=4)
