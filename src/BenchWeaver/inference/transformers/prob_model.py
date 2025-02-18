import torch
from typing import Any, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoConfig
from ...hparams.model_args import ModelArguments
from ...hparams.finetuning_args import FinetuningArguments

def load_model(model_args: "ModelArguments", finetuning_args: "FinetuningArguments") -> "PreTrainedModel":
        init_kwargs = get_init_kwargs(model_args)
        model:"PreTrainedModel" = AutoModelForCausalLM.from_pretrained(**init_kwargs)
        model.requires_grad_(False)
        model = casting_compute_type(model, model_args)
        model.eval()
        if model_args.print_param_status:
            for name, param in model.named_parameters():
                print(
                    "name: {}, dtype: {}, device: {}, trainable: {}".format(
                        name, param.dtype, param.device, param.requires_grad
                    )
                )
        return model
    
def casting_compute_type(model: "PreTrainedModel", model_args: "ModelArguments") -> "PreTrainedModel":
    if model_args.infer_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            print("BF16 not supported, switching to float16.")
            target_dtype = torch.float16
        else:
            target_dtype = torch.bfloat16
    elif model_args.infer_dtype == "fp16":
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32
    print(f"Casting model compute type to {target_dtype}.")
    
    for param in model.parameters():
        if param.data.dtype != target_dtype:
            print(f"Converting parameter of shape {param.data.shape} to {target_dtype}")
            param.data = param.data.to(target_dtype)

    return model
        
def print_parm_status(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

def get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    init_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": getattr(model_args, "token", None),
        "device_map": getattr(model_args, "device_map", "auto"),
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    init_kwargs["config"] = config
    init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
    return init_kwargs