import torch
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

base_model.eval()

peft_model = PeftModel.from_pretrained(
    base_model, "Qwen2-0.5B-PEFT-ORPO/checkpoint-375"
)
peft_model = peft_model.merge_and_unload()
peft_model = peft_model.to(device)

peft_model.eval()


def generate_text(model, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)


text_generator = pipeline(
    "text-generation", model=peft_model, tokenizer=tokenizer, device=device
)

prompt = "Explain the theory of relativity in simple terms."

print(
    text_generator(prompt, max_length=512, truncation=True)[0]["generated_text"],
)
