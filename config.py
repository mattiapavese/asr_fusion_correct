import yaml
from box import Box

with open("config.yml","r") as f:
    _config=Box(yaml.safe_load(f))

class _ModelBaseConfig:
    embed_dim:                  int     = _config.model.base.embed_dim
    num_layers:                 int     = _config.model.base.num_layers
    num_heads:                  int     = _config.model.base.num_heads
    ffw_dim:                    int     = _config.model.base.ffw_dim
    dict_dim_latent_attention:  int     = _config.model.base.dict_dim_latent_attention
    dropout:                    float   = _config.model.base.dropout
    whisper_ckpt:               str     = _config.model.base.whisper_ckpt
    mt5_ckpt:                   str     = _config.model.base.mt5_ckpt

class _ModelLargeConfig:
    embed_dim:                  int     = _config.model.large.embed_dim
    num_layers:                 int     = _config.model.large.num_layers
    num_heads:                  int     = _config.model.large.num_heads
    ffw_dim:                    int     = _config.model.large.ffw_dim
    dict_dim_latent_attention:  int     = _config.model.large.dict_dim_latent_attention
    dropout:                    float   = _config.model.large.dropout
    whisper_ckpt:               str     = _config.model.large.whisper_ckpt
    mt5_ckpt:                   str     = _config.model.large.mt5_ckpt

class _ModelConfig:
    base        = _ModelBaseConfig()
    large       = _ModelLargeConfig()
    using: str  = _config.model.using

class _OptimizerConfig:
    lr: float   = _config.train.optimizer.lr

class _LoraConfig():
    r:                              int         = _config.train.lora.r
    alpha:                          int         = _config.train.lora.alpha
    dropout:                        float       = _config.train.lora.dropout
    target_modules_audio_encoder:   list[str]   = _config.train.lora.target_modules_audio_encoder
    target_modules_lm:              list[str]   = _config.train.lora.target_modules_lm

class _TrainConfig:
    device:         str = _config.train.device
    num_epochs:     int = _config.train.num_epochs
    batch_size:     int = _config.train.batch_size
    logging_steps:  int = _config.train.logging_steps
    valid_split:    int = _config.train.valid_split
    exp_name:       str = _config.train.exp_name
    optimizer       = _OptimizerConfig()
    lora            = _LoraConfig()

class _UtilsConfig:
    models_cache_dir:       str = _config.utils.models_cache_dir
    remote_machine_ip:      str = _config.utils.remote_machine_ip
    remote_machine_user:    str = _config.utils.remote_machine_user
    remote_machine_port:    str = _config.utils.remote_machine_port
    remote_machine_workdir: str = _config.utils.remote_machine_workdir

class _DataConfig:
    sampling_rate:          int = _config.data.sampling_rate
    audio_samples_folder:   str = _config.data.audio_samples_folder

class _Config:
    model   = _ModelConfig()
    train   = _TrainConfig()
    utils   = _UtilsConfig()
    data    = _DataConfig()

config=_Config()

