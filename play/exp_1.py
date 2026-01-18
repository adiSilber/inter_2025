from play.interface import Experiment, ModelGenerationConfig, SamplingParams, ActivationCapturer
from play.generate_normal import GenerateNormal
import torch
from typing import Dict, List





class EverythingCapturer(ActivationCapturer):
    def __init__(self):
        super().__init__()
        self.model = None
        self.hook_handles = []

    def bind(self, model: torch.nn.Module):
        """Analyze the model and prepare hooks."""
        self.model = model

    def _make_hook_fn(self, layer_name: str):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # For transformer layers, output is typically a tuple (hidden_states, ...)
            # We capture the hidden states (first element)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Clone to avoid keeping references to computation graph
            if layer_name not in self.activations:
                self.activations[layer_name] = []
            self.activations[layer_name].append(hidden_states.clone().detach())
        
        return hook_fn

    def __enter__(self):
        """Register hooks on all transformer layers."""
        if self.model is None:
            raise ValueError("Model not bound. Call bind() first.")
        
        # Clear previous activations
        self.activations.clear()
        
        # Register hooks on all transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Standard transformer architecture (Qwen2, Llama, etc.)
            for i, layer in enumerate(self.model.model.layers):
                layer_name = f"layer_{i}"
                handle = layer.register_forward_hook(self._make_hook_fn(layer_name))
                self.hook_handles.append(handle)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style architecture
            for i, layer in enumerate(self.model.transformer.h):
                layer_name = f"layer_{i}"
                handle = layer.register_forward_hook(self._make_hook_fn(layer_name))
                self.hook_handles.append(handle)
        else:
            # Fallback: register hook on the entire model
            handle = self.model.register_forward_hook(self._make_hook_fn("model_output"))
            self.hook_handles.append(handle)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()





MODEL_PATH = "model_weights/"

if __name__ == "__main__":
    activation_capturer = ActivationCapturer()
    experiment = Experiment(
        model_generation_config=ModelGenerationConfig(
            model_path=MODEL_PATH,
            sampling_params=SamplingParams(
                max_new_tokens=1000
            )
        )
    )
    generate_normal = GenerateNormal(experiment)
    generate_normal.generate(batch_size=1)