from utils.LLM import LLM
from utils.PromptTemplate import HOP_INPUT
import json
from config import reasoning_dataset, reasoning_model, hop_pred_model, reasoning_type, dataset_dir, stop_tokens
#stop_tokens.append("<|eot_id|>")
reasoning_model = reasoning_model.replace(":", "-")
answer_file = f"{reasoning_model}_zero_shot_answers.json"
input_file = dataset_dir + answer_file
output_file = dataset_dir + \
    answer_file.replace("_answers.json", "_HopPred.json")

llm = LLM()
llm_inputs = []
print(input_file)
print(output_file)
with open(input_file) as f:
    json_info = json.load(f)
    llm_outputs = []
    for info in json_info["eval_info"]:
        qid = info["id"]
        llm_input = HOP_INPUT.format(
            paths=info["ReasoningPaths"],
            question=info["question"]
        )
        llm_inputs.append(llm_input)
        # llm_output = llm.invoke(llm_input)
        # print(llm_output)
        # llm_outputs.append(llm_output)
    llm_outputs = llm.batch_invoke(llm_inputs)
    for index, info in enumerate(json_info["eval_info"]):
        info["HopPredOutput"] = llm_outputs[index]

    with open(output_file, "w") as f:
        json.dump(json_info, f)
