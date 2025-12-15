from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pickle

# Storage for activations
activations = {}

def get_activation_hook(name):
    """Create a hook function to capture activations."""
    def hook(module, input, output):
        # Store activation, move to CPU and convert to numpy
        if isinstance(output, tuple):
            activations[name] = output[0].detach().cpu().numpy()
        else:
            activations[name] = output.detach().cpu().numpy()
    return hook

tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ".",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model.to("mps")

# Register hooks on all layers (AFTER model is loaded)
# hooks = []
# for name, module in model.named_modules():
#     if len(list(module.children())) == 0:  # Only leaf modules
#         hook = module.register_forward_hook(get_activation_hook(name))
#         hooks.append(hook)

text = "tell me how much is 2+2"
messages = [{"role": "user", "content": text}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(input_text, return_tensors="pt").to("mps")

# Run forward pass to capture activations
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)

# Remove hooks
for hook in hooks:
    hook.remove()

# Save activations to file
print(f"\nSaving {len(activations)} activation layers...")
with open("activations.pkl", "wb") as f:
    pickle.dump(activations, f)
print("Activations saved to activations.pkl")

# Also save as numpy arrays (optional - for easier loading)
np.savez_compressed("activations.npz", **activations)
print("Activations also saved to activations.npz")
