# FRAG pipeline
## Step 0: Data preprocessing
Get `test_name.jsonl` using PPR algorithm.

Refer to [data preprocessing](data_preprocess/README.md) for details.

## Step 1: Get reasoning paths
```python
python getPaths.py
```
This script will generate reasoning paths for each query. The paths are generated using`RecordPipeline(PreRetrievalModuleBGE(64), RetrievalModuleBFS(2), PostRetrievalModuleBGE(32))` for simple query, `RecordPipeline(PreRetrievalModuleBGE(64), RetrievalModuleDij(4), PostRetrievalModuleBGE(32))` for complex query.

## Step 2: Reasoning using paths and LLM
```python
python Reason.py
```

## Step 3: FRAG
Download the Reasoning-aware model from [here](https://huggingface.co/), and run
```python
python FRAG.py
```

## Step 4: FRAG_F
set `stop_tokens = ["\n"]`, and run
```python
python getHopPred.py
```

After getting the hop prediction for FRAG-Simple and FRAG-Complex, run
```python
python FRAG_F.py
```
for final FRAG_F prediction.