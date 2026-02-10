from sklearn.calibration import partial
import torch
from torch import nn
from transformers import LlamaModel
from pipeline.interface import ActivationCapturer, DataPoint, Experiment, GenerationMode
from typing import Optional,Any
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
class AttentionMapCapturerClipActivations(ActivationCapturer):
    """Captures attention maps and can intervene by clipping attention to question tokens."""

    def __init__(self, layers_to_clip=None, capture_weights=True, clip_max_val=1e-4):
        super().__init__()
        self.hooks = []
        self.layers: nn.ModuleList
        self.layers_to_clip = (
            layers_to_clip  # None = all layers, set/list = specific layers
        )
        self.capture_weights = capture_weights  # False = skip storing tensors
        self.clip_max_val = clip_max_val

    def capturer(
        self,
        modes: list[GenerationMode],
        datapoints: list[DataPoint],
        question_to_clip_indecies: list[int] = [],
        clip_max_val=None,
        **kwargs,
    ):
        # Yonatan: I returned this function to keep the code open for chanegs, if needed, config correctly outside
        # This wat we also don't break the interface with the parent class
        self.question_to_clip_indecies = question_to_clip_indecies
        self.clip_max_val = (
            clip_max_val if clip_max_val is not None else self.clip_max_val
        )
        return super().capturer(modes, datapoints, **kwargs)

    # def capturer(self, mode, datapoints: list[DataPoint], **kwargs):
    #     return super().capturer(mode, datapoints)

    def bind(self, model: torch.nn.Module,**kwargs):
        """Analyze the model and prepare to capture attention maps."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers  # type: ignore
        elif hasattr(model, "layers"):
            self.layers = model.layers  # type: ignore
        else:
            raise ValueError("Could not find model layers for attention capture")

        # Initialize activation storage for each layer
        for layer_idx in range(len(self.layers)):
            self.activations[f"layer_{layer_idx}_attention"] = []
            self.activations[f"layer_{layer_idx}_attention_clipped"] = []

        try:
            if hasattr(model, "config"):
                setattr(model.config, "attn_implementation", "eager")
                setattr(model.config, "_attn_implementation", "eager")
        except Exception:
            pass

    def attach_hooks(self) -> None:
        """Monkey-patch eager_attention_forward to clip attention BETWEEN softmax and @V.

        The original used register_forward_hook which runs AFTER attn_output is
        already computed, so the clipping never actually affected the model.
        By patching eager_attention_forward we insert clipping right before
        the attn_weights @ value_states matmul.
        """
        if self.layers is None:
            raise ValueError("Must call bind() before using context manager")

        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

        self._original_eager_fn = qwen2_mod.eager_attention_forward
        capturer = self

        def patched_eager_attention_forward(
            module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
        ):
            key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
            value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)

            attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=dropout, training=module.training
            )

            # Capture original (unclipped) attention weights
            if capturer.capture_weights:
                capturer.activations[f"layer_{module.layer_idx}_attention"].append(
                    attn_weights.detach().cpu()
                )

            # Clip and record clipped weights when in AFTER_INJECTION mode
            if capturer.generation_mode == [GenerationMode.AFTER_INJECTION] and (
                capturer.layers_to_clip is None
                or module.layer_idx in capturer.layers_to_clip
            ):
                clipped = attn_weights.clone()
                for i in capturer.question_to_clip_indecies:
                    clipped[:, :, :, i] = torch.clamp(
                        clipped[:, :, :, i], max=capturer.clip_max_val
                    )
                if capturer.capture_weights:
                    capturer.activations[
                        f"layer_{module.layer_idx}_attention_clipped"
                    ].append(clipped.detach().cpu())
                attn_weights = clipped  # model now uses clipped weights

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        qwen2_mod.eager_attention_forward = patched_eager_attention_forward

    def remove_hooks(self) -> None:
        """Restore the original eager_attention_forward."""
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

        if hasattr(self, "_original_eager_fn"):
            qwen2_mod.eager_attention_forward = self._original_eager_fn
            del self._original_eager_fn


class AttentionMapCapturer(ActivationCapturer):
    """Captures attention maps from all layers and heads during model forward pass."""

    def __init__(self):
        super().__init__()
        self.hooks = []
        self.layers: nn.ModuleList

    def bind(self, model: torch.nn.Module,**kwargs):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers  # type: ignore
        elif hasattr(model, "layers"):
            self.layers = model.layers  # type: ignore
        else:
            raise ValueError("Could not find model layers for attention capture")

        # Initialize activation storage for each layer
        for layer_idx in range(len(self.layers)):
            layer_name = f"layer_{layer_idx}_attention"
            self.activations[layer_name] = []
        # Ensure model forwards return attentions by default when possible.
        try:
            if hasattr(model, "config"):
                # Prefer to enable attention outputs so hooks can see weights.
                # setattr(self.model.config, 'output_attentions', True)
                # Some implementations expose an attn implementation flag; prefer eager.
                try:
                    setattr(model.config, "attn_implementation", "eager")
                except Exception:
                    pass
        except Exception:
            pass

    def attach_hooks(self) -> None:
        """Register hooks to capture attention outputs."""
        if self.layers is None:
            raise ValueError("Must call bind() before using context manager")

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
        for layer_idx, layer in enumerate(self.layers):
            # Different model architectures use different names
            if hasattr(layer, "self_attn"):
                attn_module = layer.self_attn
            elif hasattr(layer, "attention"):
                attn_module = layer.attention
            else:
                continue

            hook_handle = attn_module.register_forward_hook(make_hook(layer_idx))  # type: ignore
            self.hooks.append(hook_handle)

    def remove_hooks(self) -> None:
        """Remove hooks after capture."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class AttentionHeadClipper(ActivationCapturer):

    def __init__(
        self,
    ):
        self.hooks_handles = []
        self.current_modes = []
        self.model: Optional[HookedTransformer] = None # type: ignore

    def bind(self, model: torch.nn.Module,**kwargs):
        if not isinstance(model, HookedTransformer):
            raise ValueError(
                "AttentionHeadClipper requires a TransformerLens 'HookedTransformer'. "
                "Standard HF models mask the internal attention pattern."
            )
        if not "experiment" in kwargs:
            raise ValueError("Experiment config must be passed to AttentionMapCapturer via bind() kwargs")
        self.experiment: Experiment = kwargs["experiment"]
        self.clip_max_val = self.experiment.clip_max_val
        self.heads_to_clip = self.experiment.activation_head_clipping if self.experiment.activation_head_clipping is not None else {} 
        super().bind(model,**kwargs)

    def capturer(self, modes: list[GenerationMode], datapoints: list[Any], **kwargs):
        self.current_num_pad_tokens = kwargs.get("num_pad_tokens", [])
        return super().capturer(modes, datapoints, **kwargs)

    def question_indecies_to_clip(self, datapoint: DataPoint, num_pad_tokens: int) -> list[int]:
        start = self.experiment.model_generation_config.question_prompt_template.prompt_tokens_before_content
        end = len(datapoint.question_formatted_contents_tokenized)-self.experiment.model_generation_config.question_prompt_template.prompt_tokens_after_content
        start += num_pad_tokens
        end += num_pad_tokens
        return list(range(start, end))
    def attach_hooks(self) -> None:
        if self.datapoints is None or self.current_modes is None:
            raise ValueError("Must have datapoints and modes set before attaching hooks")
        def _clipping_hook_fn(tensor, hook:HookPoint, head_indices : list[int], max_val, datapoints:list[DataPoint], modes:list[GenerationMode],pads:list[int]):
            for dp_index, dp in enumerate(datapoints):
                
                
                target_dict = {}
                if dp.should_capture_activations:
                    if modes[dp_index] == GenerationMode.UPTO_INJECTION:
                        target_dict = dp.activations_upto_injection
                    elif modes[dp_index] == GenerationMode.INJECTION:
                        target_dict = dp.activations_injection
                    elif modes[dp_index] == GenerationMode.AFTER_INJECTION:
                        target_dict = dp.activations_after_injection

                    target_dict[f"layer_{hook.layer}_attention"].append(tensor[dp_index, :, 0, :].detach().cpu()) # shape (batch,heads,query,key) and we only have one query


                if modes[dp_index] != GenerationMode.AFTER_INJECTION and modes[dp_index] != GenerationMode.UPTO_INJECTION:
                    continue
                token_indecies = self.question_indecies_to_clip(dp, pads[dp_index])
                tensor[dp_index, head_indices, :, token_indecies] = torch.clamp(
                    tensor[dp_index, head_indices, :, token_indecies],
                    max=max_val,
                )

                if dp.should_capture_activations and len(head_indices) > 0:
                    target_dict[f"layer_{hook.layer}_attention_clipped"].append(tensor[dp_index, :, 0, :].detach().cpu()) 
            
            return tensor
    
        if self.model is None:
            raise ValueError("Must call bind() before using context manager")

        for layer_idx in range(self.model.cfg.n_layers):
            hook_name = f"blocks.{layer_idx}.attn.hook_pattern"
            hook = partial(_clipping_hook_fn, head_indices=self.heads_to_clip.get(layer_idx, []), max_val=self.clip_max_val,datapoints=self.datapoints, modes=self.current_modes, pads=self.current_num_pad_tokens)
            self.model.add_hook(hook_name, hook)

    def remove_hooks(self) -> None:
        if self.model:
            self.model.remove_all_hook_fns(including_permanent=False)






