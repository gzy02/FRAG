from config import test_file, reasoning_dataset, dataset_dir, reasoning_model
from utils.Tools import get_query
from pre_retrieval import *
from retrieval import *
from post_retrieval import *
from pipeline.RecordPipeline import RecordPipeline
import json
if __name__ == "__main__":
    eval_info = []

    module_pipe = RecordPipeline(PreRetrievalModuleBGE(64),
                                 RetrievalModuleBFS(2), PostRetrievalModuleBGE(32))
    print(module_pipe)
    basic_info = {"Dataset": reasoning_dataset,
                  "test_file": test_file, "PreRetrievalModule": str(module_pipe.preRetrieval), "RetrievalModule": str(module_pipe.retrieval), "PostRetrievalModule": str(module_pipe.postRetrieval)}

    sample = 0
    for query in get_query(test_file):
        sample += 1

        query, res_dict = module_pipe.run(query)
        eval_info.append(res_dict)

    print(sample)
    with open(f"{dataset_dir}{reasoning_dataset}_paths.json", "w") as fp:
        json.dump(
            {
                "basic_info": basic_info,
                "eval_info": eval_info
            }, fp, indent=4)
