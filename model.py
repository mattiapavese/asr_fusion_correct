from transformers import WhisperForAudioClassification, PreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import MT5ForConditionalGeneration, MT5TokenizerFast
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from torch.nn import MultiheadAttention, LayerNorm, Linear, Sequential
from transformers.modeling_outputs import Seq2SeqLMOutput
from config import config
from peft import LoraConfig, get_peft_model

from transformers.modeling_outputs import BaseModelOutput
import torch

# +++ Fusion Info +++
# Name      | MT5 Model     | Whisper Model     | N. layers | Text Embedding Dim    | Audio Embedding Dim
# BASE      | mt5-base      | whisper-small     | 12        | 768                   | 768
# LARGE     | mt5-large     | whisper-medium    | 24        | 1024                  | 1024

# let's start making experiments on BASE version

#TODO combine all pieces in a single model

cache_dir="model"

# fusionModule
class FusionLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, ffw_dim ) -> None:
        super().__init__()

        #self._cross_attention=MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self._mixing_mha=MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self._fusion_mha=MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self._layernorm=LayerNorm(embed_dim)
        self._ffw=Sequential(
            Linear(embed_dim, ffw_dim), 
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(ffw_dim, embed_dim))

    def forward(self, x, text_emb, audio_emb): #x stands for previous fusion layer out, or dictionary
        x=self._mixing_mha(text_emb,x,x)[0] + text_emb
        x=self._layernorm(x)

        x=x+self._fusion_mha(x, audio_emb, audio_emb)[0]
        x=self._layernorm(x)

        x=x+self._ffw(x)
        return self._layernorm(x)
    
#choose num_heads between 8, 12, 16, 24
class FusionStack(torch.nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads=8, dropout=.1, ffw_dim=3072)->None:
        super().__init__()

        self._dictionary=torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.xavier_normal_(self._dictionary)

        self._layers = torch.nn.ModuleList(
            [FusionLayer(embed_dim, num_heads, dropout, ffw_dim) for _ in range(num_layers)]
        )

    def forward(self, text_embeddings:tuple[torch.Tensor], audio_embeddings:tuple[torch.Tensor]):
        if text_embeddings[0].ndim==3:
            x=torch.stack([self._dictionary] * text_embeddings[0].size(0), dim=0)
        else:
            x=self._dictionary
        
        result=[]
        for l,t,a in zip(self._layers, text_embeddings, audio_embeddings):
            x=l(x, t, a)
            result.append(x)
        
        return tuple(result)

class LatentAttentionPooling(torch.nn.Module):
    def __init__(self, embed_dim, latents_dim, num_heads=8, dropout=.1, ffw_dim=3072)->None:
        super().__init__()

        self._dictionary=torch.nn.Parameter(torch.Tensor(latents_dim, embed_dim))
        torch.nn.init.xavier_normal_(self._dictionary)

        self._attention=MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self._ffw=Sequential(
            Linear(embed_dim, ffw_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(ffw_dim, embed_dim))
        self._layernorm=LayerNorm(embed_dim)
    
    def forward(self, x:torch.Tensor):
        if x.ndim==3:
            dictionary=torch.stack([self._dictionary] * x.size(0), dim=0)
        else:
            dictionary=self._dictionary
        x=self._attention(x, dictionary, dictionary)[0]
        x=self._ffw(x)
        mean_pooled=torch.mean(x, -2) #should mean pool on timestamps dim i suppose
        return self._layernorm(mean_pooled)

class ARFusionCorrect(torch.nn.Module):
    def __init__(self, 
            audio_encoder,
            language_model,
            embed_dim_fusion:int, #this has to change according to base/large
            num_layers_fusion:int, #this has to change according to base/large
            num_heads_fusion:int=8,
            ffw_dim_fusion:int=3072,
            dropout:float=.1,
            dict_dim_latent_attention:int=512,
            num_heads_latent_attention:int=8,
            ffw_dim_latent_attention:int=3072)->None:
        super().__init__()
        
        self._audio_encoder= audio_encoder
        self._language_model=language_model
        
        self._fusion=FusionStack(
            num_layers_fusion, 
            embed_dim_fusion, 
            num_heads_fusion,
            dropout,
            ffw_dim_fusion)
        
        self._latent_attention=LatentAttentionPooling(
            embed_dim_fusion, 
            dict_dim_latent_attention, 
            num_heads_latent_attention,
            dropout,
            ffw_dim_latent_attention)
        
        self._classifier=Sequential(
            torch.nn.Linear(embed_dim_fusion,int(embed_dim_fusion/4)),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(int(embed_dim_fusion/4), 1)
        )

        self._classifier_loss_fn=torch.nn.BCEWithLogitsLoss()

        self._classifier_loss:torch.Tensor|None=None
        self._lm_loss:torch.Tensor|None=None
    
    @property
    def loss(self):
        return self._classifier_loss + self._lm_loss
    @property
    def classifier_loss(self):
        return self._classifier_loss
    @property
    def lm_loss(self):
        return self._lm_loss

    def forward(self, 
            text_input_ids:torch.Tensor,
            text_attention_mask:torch.Tensor,
            audio_input_features:torch.Tensor|None=None,
            text_labels:torch.Tensor|None=None,
            classifier_labels:torch.Tensor|None=None
            ):

        compute_loss:bool=False
        if classifier_labels is not None and text_labels is not None:
            compute_loss=True
        elif classifier_labels is None and text_labels is None:
            pass
        else:
            raise Exception("Cannot provide text_decoder_labels or classifier_labels exclusively.")

        audio_enc_out:BaseModelOutput
        audio_embeddings:tuple[torch.Tensor]
        if audio_input_features is not None:
            audio_enc_out=self._audio_encoder(audio_input_features, output_hidden_states=True)
            audio_embeddings=audio_enc_out.hidden_states

        text_enc_out:BaseModelOutput
        text_enc_out=self._language_model.encoder(
            input_ids=text_input_ids, 
            attention_mask=text_attention_mask,
            output_hidden_states=True)
        text_embeddings:tuple[torch.Tensor]
        text_embeddings=text_enc_out.hidden_states
        
        fused:tuple[torch.Tensor]
        # skip fusion if audio input features are not present
        if audio_input_features is not None:
            # oss we slice embeddings because the first dimension is output of word embedding
            # and correspective for audio 
            # TODO check in audio i do not need to grab [0:-1] actually , i am not super sure 🚨🚨🚨🚨
            fused=self._fusion(text_embeddings[1:], audio_embeddings[1:])                                       
            fused=(text_embeddings[0],)+fused          
        else:
            fused=text_embeddings

        latent_pool=self._latent_attention(fused[-1])
        logit=self._classifier(latent_pool)
      
        decoder_out:Seq2SeqLMOutput
        if compute_loss:
            self._classifier_loss=self._classifier_loss_fn(logit, classifier_labels)

            decoder_out=self._language_model(encoder_outputs=fused, labels=text_labels)
            lm_loss=decoder_out.loss

            self._lm_loss=lm_loss
        else:
            raise NotImplementedError(("I still don't know how to do this :) "
                                       "Probably need to exploit `generate` function, "
                                       "and do it in two steps, because i first wanna check "
                                       "the result of classification head."))
        
        return None #still need to understand what to return

        
def get_model_for_generalized_train():

    #TODO i don't like this should be automatic
    if config.model.using=="base":
        model_config=config.model.base
    elif config.model.using=="large":
        model_config=config.model.large
    else:
        raise NotImplementedError("no model config found")
    
    lora_config_audio_enc= LoraConfig(
        r=config.train.lora.r,
        lora_alpha=config.train.lora.alpha,
        target_modules=config.train.lora.target_modules_audio_encoder,
        lora_dropout=config.train.lora.dropout
    )
    lora_config_lm= LoraConfig(
        r=config.train.lora.r,
        lora_alpha=config.train.lora.alpha,
        target_modules=config.train.lora.target_modules_lm,
        lora_dropout=config.train.lora.dropout
    )

    audio_encoder:WhisperEncoder
    audio_encoder=WhisperForAudioClassification.from_pretrained(
        model_config.whisper_ckpt, cache_dir=config.utils.models_cache_dir).encoder
    audio_encoder_lora=get_peft_model(audio_encoder, lora_config_audio_enc)
    audio_encoder_lora.to(config.train.device)
    
    lm_model=MT5ForConditionalGeneration.from_pretrained(
        model_config.mt5_ckpt, cache_dir=config.utils.models_cache_dir)
    lm_model_lora=get_peft_model(lm_model, lora_config_lm)
    lm_model_lora.to(config.train.device)
    
    ar_fusion_model=ARFusionCorrect(
        audio_encoder_lora, 
        lm_model_lora, 
        embed_dim_fusion=model_config.embed_dim,
        num_layers_fusion=model_config.num_layers,
        num_heads_fusion=model_config.num_heads,
        ffw_dim_fusion=model_config.ffw_dim,
        dropout=model_config.dropout,
        dict_dim_latent_attention=model_config.dict_dim_latent_attention,
        num_heads_latent_attention=model_config.num_heads,
        ffw_dim_latent_attention=model_config.ffw_dim
    )

    ar_fusion_model.to(device=config.train.device)

    return ar_fusion_model