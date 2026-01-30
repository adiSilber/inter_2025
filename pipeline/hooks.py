import torch
from pipeline.interface import ActivationCapturer, DataPoint, GenerationMode


class AttentionMapCapturerClipActivations(ActivationCapturer):
    """Captures attention maps and can intervene by clipping attention to question tokens."""
    
    def __init__(self):
        super().__init__()
        self.hooks = []
        self.model = None
        # self._v_cache = {}  # Store V tensors per layer

    def capturer(self, mode, datapoints: list[DataPoint], question_to_clip_indecies: list[int]=[]):    # adisi changed - currently I clip all question indices
        # Yonatan: I returned this function to keep the code open for chanegs, if needed, config correctly outside 
        # This wat we also don't break the interface with the parent class
        self.question_to_clip_indecies = question_to_clip_indecies
        return super().capturer(mode, datapoints)

    # def capturer(self, mode, datapoints: list[DataPoint], **kwargs):
    #     return super().capturer(mode, datapoints)

    def bind(self, model: torch.nn.Module):
        """Analyze the model and prepare to capture attention maps."""
        self.model = model
        
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise ValueError("Could not find model layers for attention capture")
        
        # Initialize activation storage for each layer
        for layer_idx in range(len(layers)):
            self.activations[f"layer_{layer_idx}_attention"] = []
            self.activations[f"layer_{layer_idx}_attention_clipped"] = []
        
        try:
            if hasattr(self.model, 'config'):
                setattr(self.model.config, 'attn_implementation', 'eager')
        except Exception:
            pass
    
    def attach_hooks(self):
        """Register hooks to capture V and attention, then intervene."""
        if self.model is None:
            raise ValueError("Must call bind() before using context manager")
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Could not find model layers")
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Qwen/DeepSeek attention modules typically return a tuple where
                # output[1] are the attention weights when `output_attentions=True`.
                attention_weights = None
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights = output[1]

                # If weights are missing, raise instructive error so caller can
                # ensure `output_attentions=True` or that model config was set.
                if attention_weights is None:
                    raise ValueError(
                        f"Layer {layer_idx}: attention weights are None. "
                        "Ensure the model is called with `output_attentions=True` "
                        "or set `model.config.output_attentions = True` and use an "
                        "attn implementation that exposes weights (e.g. 'eager')."
                    )

                # Offload to CPU and detach to save GPU memory and break graph
                try:
                    cpu_attn = attention_weights.detach().cpu()
                except Exception:
                    # If not a tensor, store as-is
                    cpu_attn = attention_weights

                self.activations[f"layer_{layer_idx}_attention"].append(cpu_attn)
                if self.generation_mode == GenerationMode.AFTER_INJECTION:
                    # question_indexes = range(len(self.datapoints[0].question_formatted_contents_tokenized))
                    for i in self.question_to_clip_indecies:
                        attention_weights[:, :, :, i] = torch.clamp(attention_weights[:, :, :, i], max=1e-4)
                    try:
                        clipped = attention_weights.detach().cpu()
                    except Exception:
                        # If not a tensor, store as-is
                        print("could not detach attention weights, saving unclipped weights")
                        clipped = attention_weights

                    self.activations[f"layer_{layer_idx}_attention_clipped"].append(clipped)

            return hook
        
        # Register hooks
        for layer_idx, layer in enumerate(layers):
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention
            else:
                continue
            
            hook_handle = attn_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook_handle)
        
        return self
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class AttentionMapCapturer(ActivationCapturer):
    """Captures attention maps from all layers and heads during model forward pass."""
    
    def __init__(self):
        super().__init__()
        self.hooks = []
        self.model = None
    
    def bind(self, model: torch.nn.Module):
        """Analyze the model and prepare to capture attention maps."""
        self.model = model
        
        # Find all layers with attention modules
        # For transformers, typically model.layers[i].self_attn or similar
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise ValueError("Could not find model layers for attention capture")
        
        # Initialize activation storage for each layer
        for layer_idx in range(len(layers)):
            layer_name = f"layer_{layer_idx}_attention"
            self.activations[layer_name] = []
        # Ensure model forwards return attentions by default when possible.
        try:
            if hasattr(self.model, 'config'):
                # Prefer to enable attention outputs so hooks can see weights.
                # setattr(self.model.config, 'output_attentions', True)
                # Some implementations expose an attn implementation flag; prefer eager.
                try:
                    setattr(self.model.config, 'attn_implementation', 'eager')
                except Exception:
                    pass
        except Exception:
            pass
    
    def attach_hooks(self):
        """Register hooks to capture attention outputs."""
        if self.model is None:
            raise ValueError("Must call bind() before using context manager")
        
        # Find layers and register hooks
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
        else:
            raise ValueError("Could not find model layers")
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Qwen/DeepSeek attention modules typically return a tuple where
                # output[1] are the attention weights when `output_attentions=True`.
                attention_weights = None
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights = output[1]

                # If weights are missing, raise instructive error so caller can
                # ensure `output_attentions=True` or that model config was set.
                if attention_weights is None:
                    raise ValueError(
                        f"Layer {layer_idx}: attention weights are None. "
                        "Ensure the model is called with `output_attentions=True` "
                        "or set `model.config.output_attentions = True` and use an "
                        "attn implementation that exposes weights (e.g. 'eager')."
                    )

                # Offload to CPU and detach to save GPU memory and break graph
                try:
                    cpu_attn = attention_weights.detach().cpu()
                except Exception:
                    # If not a tensor, store as-is
                    cpu_attn = attention_weights

                self.activations[f"layer_{layer_idx}_attention"].append(cpu_attn)

            return hook
        
        # Register hooks on attention modules
        for layer_idx, layer in enumerate(layers):
            # Different model architectures use different names
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attention'):
                attn_module = layer.attention
            else:
                continue
            
            hook_handle = attn_module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook_handle)
        
        return self
    
    def remove_hooks(self):
        """Remove hooks after capture."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
