
############################################
# region Reasonging Module
PERSONA = """You are an expert reasoner with a deep understanding of logical connections and relationships. Your task is to analyze the given reasoning paths and provide clear and accurate answers to the questions based on these paths."""

REASONING_TEMPLATE = """Based on the reasoning paths, please answer the given question.

Reasoning Paths:
{paths}

Question:
{question}

## Output:

"""
REASONING_INPUT = PERSONA+ REASONING_TEMPLATE
# endregion

############################################
# region Hop-estimate Module
PERSONA_HOP = """You are an expert reasoner with a deep understanding of logical connections and relationships. Your task is to analyze the given reasoning paths and provide accurate reasoning path to the questions based on these paths."""


REASONING_TEMPLATE_HOP = """Based on the reasoning paths, please extract the correct reasoning path. If NO correct reasoning path, please just reply "NO".

Reasoning Paths:
{paths}

Question: {question}

Correct reasoning path: 
"""
HOP_INPUT = PERSONA_HOP + REASONING_TEMPLATE_HOP
# endregion
