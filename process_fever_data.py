from datasets import DatasetDict, Dataset

import json
import random
import itertools

def read_json_lines_file(file):
    with open(file) as f:
        for line in f:
            inst = json.loads(line)
            yield inst

def get_instances(split, num_max_instances=None):

    file = f"fever_data/fever_retrieval_{split}.jsonl"
    instances = list()

    for data in itertools.islice(read_json_lines_file(file), 0, num_max_instances):

        # discard instances with no positive passage or no negative passage
        num_positive_passages = len(data["positive_passages"][0])
        num_negative_passages = len(data["negative_passages"][0])
        if (num_positive_passages < 1) or (num_negative_passages < 1):
            continue

        # we select the first positive passage
        positive_passage = list(data["positive_passages"][0][0].items())[0] 

        # we sample 1 negative passage randomly
        hard_negative_passage = list(random.choice(data["negative_passages"][0]).items())[0] 

        instance = {
            "query": data["query"],
            "positive_passage":  {
                "title": positive_passage[0],
                "passage": positive_passage[1]
            },
            "hard_negative_passage":  {
                "title": hard_negative_passage[0],
                "passage": hard_negative_passage[1]
            }
        }

        instances.append(instance)

    return instances

train_instances = get_instances("train")
validation_instances = get_instances("dev")
test_instances = get_instances("test")

dataset = DatasetDict({
    "train": Dataset.from_list(train_instances),
    "dev": Dataset.from_list(validation_instances),
    "test": Dataset.from_list(validation_instances)
})

dataset.push_to_hub("tomhodemon/fever_data", private=True)