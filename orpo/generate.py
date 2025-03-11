import emoji
from distilabel.models.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    FormatTextGenerationDPO,
    GroupColumns,
    LoadDataFromHub,
    StepInput,
    StepOutput,
    step,
)
from distilabel.steps.tasks import TextGeneration


def contains_emoji(text):
    return any(char in emoji.EMOJI_DATA for char in text)


@step(inputs=["generation"], outputs=["ratings"], step_type="global")
def emoji_classifier(inputs: StepInput) -> "StepOutput":
    for gen in inputs:
        gen["ratings"] = [1 if contains_emoji(g) else 0 for g in gen["generations"]]
    yield inputs


generation_model_1 = TransformersLLM(
    model="unsloth/Llama-3.2-3B-Instruct",
    generation_kwargs={"temperature": 0.7, "max_new_tokens": 256},
)

generation_model_2 = TransformersLLM(
    model="unsloth/Llama-3.2-1B-Instruct",
    generation_kwargs={"temperature": 0.7, "max_new_tokens": 256},
)

with Pipeline(name="generate-dataset") as pipeline:
    load_dataset = LoadDataFromHub(
        repo_id="openai/gsm8k",
        batch_size=32,
        num_examples=1000,
        pipeline=pipeline,
        config="main",
        output_mappings={"question": "instruction"},
    )

    generate_responses = [
        TextGeneration(
            llm=generation_model_1,
            name="text_generation_emoji",
            system_prompt=(
                "You are a helpful AI Assistant writer. "
                "You will be provided with an instruction and true answer."
                "Please reformat the answer using markdown AND insert emojis for clarity."
                "You can use emojis to format buletpoints, emphaise emotions"
                "Messages without emojis will be rejected."
                "Long messages are discouraged"
            ),
            template="QUESTION: {{ instruction }}\nANSWER: {{ answer }}",
            pipeline=pipeline,
            num_generations=1,
            columns=["answer", "instruction"],
        ),
        TextGeneration(
            llm=generation_model_2,
            name="text_generation_no_emoji",
            system_prompt=(
                "You are a helpful AI Assistant writer. "
                "Please reformat the answer using markdown AND. DO NOT insert emojis"
                "Messages with emojis will be rejected."
                "Long messages are discouraged"
            ),
            template="ANSWER: {{ answer }}",
            pipeline=pipeline,
            num_generations=1,
            columns=["answer"],
        ),
    ]

    classify_responses = emoji_classifier(pipeline=pipeline)

    group_responses = GroupColumns(
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"],
        pipeline=pipeline,
    )

    format_dpo = FormatTextGenerationDPO(
        pipeline=pipeline,
    )

    load_dataset >> generate_responses[0] >> group_responses
    load_dataset >> generate_responses[1] >> group_responses

    group_responses >> classify_responses >> format_dpo


if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    distiset.save_to_disk(distiset_path="emoji-dataset")
