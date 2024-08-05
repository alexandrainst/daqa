from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (
    chain_of_thought, generate, self_critique, system_message
)

SYSTEM_MESSAGE = system_message("Svar på følgende spørgsmål ved hjælp af den givne kontekst. Svaret skal være kort, højst en sætning.")

def record_to_sample(record):
    return Sample(
        input=f"{record['context']}\n{record['question']}",
        target=record["answer"],
        metadata=dict(title=record["title"]),
    )

@task
def daqa():
    return Task(
        dataset=hf_dataset(
            path="alexandrainst/daqa",
            split="train",
            sample_fields=record_to_sample,
            limit=50
            ),
        plan=[
            SYSTEM_MESSAGE,
            generate()
        ],
        scorer=model_graded_fact()
    )

@task
def daqa_hard():
    return Task(
        dataset=hf_dataset(
            path="alexandrainst/daqa-hard",
            split="train",
            sample_fields=record_to_sample,
            # limit=50
            ),
        plan=[
            SYSTEM_MESSAGE,
            generate()
        ],
        scorer=model_graded_fact()
    )