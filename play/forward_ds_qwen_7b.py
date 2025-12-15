import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_path = "models/DS-qwen-7B/DeepSeek-R1-Distill-Qwen-7B"
print(f"Loading model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Simple forward pass
prompt = "Hello, how are you?"
print(f"\nPrompt: {prompt}")

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
print("\nGenerating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated text:\n{generated_text}")
