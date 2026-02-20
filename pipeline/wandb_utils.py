"""Utilities for Weights & Biases logging of experiment config and metrics."""

from typing import Any, Dict

from pipeline.interface import (
    Experiment,
    ModelGenerationConfig,
    JudgeGenerationConfig,
    SamplingParams,
)
from pipeline.dataset_loaders import aggregate_shuffle_strategy


def experiment_config_for_wandb(
    experiment: Experiment,
    output_dir: str,
) -> Dict[str, Any]:

    def _class_name(obj: Any) -> str | None:
        if obj is None:
            return None
        return type(obj).__name__

    def _sampling_params_dict(sp: SamplingParams | None) -> Dict[str, Any]:
        if sp is None:
            return {}
        return {
            "temperature": sp.temperature,
            "top_k": sp.top_k,
            "top_p": sp.top_p,
            "take_dumb_max": sp.take_dumb_max,
            "max_new_tokens": sp.max_new_tokens,
        }

    mc = experiment.model_generation_config
    jc = experiment.judge_generation_config
    dataset = experiment.dataset
    strategy_name = dataset.strategy.name

    config: Dict[str, Any] = {
        "experiment_name": experiment.name,
        "runner_name": experiment.runner_name,
        "unique_id": experiment.unique_id,
        "seed": experiment.seed,
        "output_dir": output_dir,
        "num_datapoints": len(experiment.datapoints),
        # Model
        "model_name": mc.model_name,
        "model_path": mc.model_path,
        "model_should_stop": _class_name(mc.should_stop),
        "model_get_injection": _class_name(mc.get_injection),
        "model_global_stop": _class_name(mc.global_stop),
        "model_question_prompt_template": _class_name(mc.question_prompt_template),
        "model_dtype": str(mc.dtype),
        "model_sampling_params": _sampling_params_dict(mc.sampling_params),
        # Judge
        # Dataset
        "dataset_loaders": [type(d).__name__ for d in dataset.loaders],
        "dataset_strategy": strategy_name,
        "dataset_seed": getattr(dataset, "seed", None),
        "activation_head_clipping": str(experiment.activation_head_clipping),
    }
    if jc:
        config.update(
            {
                "judge_name": jc.judge_name,
                "judge_model_path": jc.judge_model_path,
                "judge_prompt": _class_name(jc.judge_prompt),
                "judge_dtype": str(jc.dtype),
                "judge_sampling_params": _sampling_params_dict(jc.sampling_params),
            }
        )
    if experiment.activation_capturer is not None:
        cap = experiment.activation_capturer
        config["activation_capturer"] = _class_name(cap)
        for attr in ("clip_max_val", "layers_to_clip", "capture_weights"):
            if hasattr(cap, attr):
                config[f"activation_capturer_{attr}"] = getattr(cap, attr)
    return config
