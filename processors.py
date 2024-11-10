from config import config
from transformers import MT5TokenizerFast, T5TokenizerFast, MT5ForConditionalGeneration
from transformers import WhisperProcessor

#TODO i don't like this should be automatic (just as in fusion_model.py)  
if config.model.using=="base":
    model_config=config.model.base
elif config.model.using=="large":
    model_config=config.model.large
elif config.model.using=="small":
    model_config=config.model.small
else:
    raise NotImplementedError("no model config found")

text_tokenizer:T5TokenizerFast
text_tokenizer=MT5TokenizerFast.from_pretrained(
    model_config.mt5_ckpt, cache_dir=config.utils.models_cache_dir)

audio_processor=WhisperProcessor.from_pretrained(
    model_config.whisper_ckpt, cache_dir=config.utils.models_cache_dir
)

    