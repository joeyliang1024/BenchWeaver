from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase
from ...hparams.model_args import ModelArguments
from ...extras.logging import get_logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig, ProcessorMixin
    
logger = get_logger(__name__)

class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    
def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)
    try:
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        patch_processor(processor, config, tokenizer, model_args)
    except Exception:
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}

    
def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": getattr(model_args, "token", None),
    }

def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
  
def patch_tokenizer(tokenizer: "PreTrainedTokenizer") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)
        
def patch_processor(
    processor: "ProcessorMixin",
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    setattr(processor, "tokenizer", tokenizer)