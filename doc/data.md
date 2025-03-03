# Evaluation Data
This Respository support two ways to load your evaluation datasset.
## HuggingFace Dataset

## Provide Evaluation Data File
If you want to provide your own dataset, please follow the columns below to format your dataset:
```json
{
    "name": "the name of the dataset",
    "type": "please specify one of the following types: MCQA-OQ, MCQA-Prob, OPQA, Mix",
    "few_shot": "the split use for few shot.",
    "metric": "the metric for calculate the score.",
    "data": {
      "train": "list of data",
      "validation": "list of data",
      "test": "list of data"
    }
}
```

Below are the details for each row of the data:
```json
{
  "system_prompt": "the column name in the dataset containing the system prompts. (default: None)",
  "question_prompt": "the column name in the dataset containing the prompts. (default: None)",
  "question": "the column name in the dataset containing the questions. (default: question)",
  "answer": "the column name in the dataset containing the OPQA answer. (default: answer)",
  "options": "dict of options for MCQA. Ex: {\"A\": 1, \"B\": 2, \"C\": 3} (default: None)",
  "correct_options": "list of answers in the options from MCQA (default: correct_option)",
  "messages": "the column name in the dataset containing the messages. [{\"role\": \"user\", \"content\": \"123\"}, {\"role\": \"assistant\", \"content\": \"123\"}] (default: None)"
}
```