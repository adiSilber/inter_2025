

import torch
from torch import nn
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
        """Monkey-patch eager_attention_forward to clip attention BETWEEN softmax and @V.

        The original used register_forward_hook which runs AFTER attn_output is
        already computed, so the clipping never actually affected the model.
        By patching eager_attention_forward we insert clipping right before
        the attn_weights @ value_states matmul.
        """
        if self.model is None:
            raise ValueError("Must call bind() before using context manager")

        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        self._original_eager_fn = qwen2_mod.eager_attention_forward
        capturer = self

        def patched_eager_attention_forward(module, query, key, value,
                                            attention_mask, scaling, dropout=0.0, **kwargs):
            key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
            value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)

            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

            # Capture original (unclipped) attention weights
            capturer.activations[f"layer_{module.layer_idx}_attention"].append(attn_weights.detach().cpu())

            # Clip and record clipped weights when in AFTER_INJECTION mode
            if capturer.generation_mode == GenerationMode.AFTER_INJECTION:
                clipped = attn_weights.clone()
                for i in capturer.question_to_clip_indecies:
                    clipped[:, :, :, i] = torch.clamp(clipped[:, :, :, i], max=1e-4)
                capturer.activations[f"layer_{module.layer_idx}_attention_clipped"].append(clipped.detach().cpu())
                attn_weights = clipped  # model now uses clipped weights

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        qwen2_mod.eager_attention_forward = patched_eager_attention_forward
        return self

    def remove_hooks(self):
        """Restore the original eager_attention_forward."""
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
        if hasattr(self, '_original_eager_fn'):
            qwen2_mod.eager_attention_forward = self._original_eager_fn
            del self._original_eager_fn

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
