from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import os
import librosa
from config import config
from processors import text_tokenizer, audio_processor

def load_multiple_audios(audio_paths:list[str], sr:int=16000):
    audios=[]
    for path in audio_paths:
        audios.append( librosa.load(path, sr=sr) )
    return audios

_dataset=load_dataset("csv", data_files=["dataset.tsv"], sep="\t")
_dataset.map(batched=True, load_from_cache_file=False)

def with_audio_collate_fn(batch:list[dict[str, str|int]]):
    #expects batch as {"original":str, "noisy":str, "audio":str, "correct":int}
    
    audio_samples_folder=config.data.audio_samples_folder
    sampling_rate=config.data.sampling_rate
    device=config.train.device

    # @-> tokenize noisy with padding -> `text_input_ids`, `text_attention_mask`
    noisy_sentences=[el["noisy"] for el in batch]
    text_enc_inputs:dict[str,torch.Tensor]=text_tokenizer(
        noisy_sentences, return_tensors="pt", padding=True, add_special_tokens=True)

    # @-> tokenize original with padding and set pad tokens to -100 where necessary -> `text_labels`
    original_sentences=[el["original"] for el in batch]
    text_labels:torch.Tensor=text_tokenizer(
        original_sentences, return_tensors="pt", padding=True, add_special_tokens=True)["input_ids"]
    text_labels[ text_labels==text_tokenizer.pad_token_id] = - 100

    # @-> using librosa and whisper audio processor load audio files and trasform into tensor -> `audio_input_features`
    audio_names=[ el["audio"] for el in batch ]
    audio_paths = [os.path.join(audio_samples_folder, name) for name in audio_names]
    audios = load_multiple_audios(audio_paths, sr=sampling_rate)
    audios = [ a[0] for a in audios ]

    audio_input_features:torch.Tensor=audio_processor(
        audios,sampling_rate=sampling_rate,return_tensors="pt")["input_features"]
    
    # @-> produce tensor (batch_size, 1) using `correct` -> `classifier_labels`
    classifier_labels=torch.tensor([ el["correct"] for el in batch], dtype=torch.float32).reshape((-1,1))
    
    # @ i already put tensors on device here
    return { 
                "audio_input_features":audio_input_features.to(device),
                "text_input_ids":text_enc_inputs["input_ids"].to(device),
                "text_attention_mask":text_enc_inputs["attention_mask"].to(device),
                "text_labels":text_labels.to(device),
                "classifier_labels":classifier_labels.to(device),
            }   

class DataLoaderWrapper:
    """
    Should wrap with_audio_train_dataloader and only_text_train_dataloader,
    in order to have common access interface to mix data sources in train/validation.

    need methods in, len
    """
    pass

def get_dataloaders()->tuple[DataLoader]: #TODO should return tuple of DataLoaderWrapper
    
    train_val_split=_dataset["train"].train_test_split(test_size=config.train.valid_split)

    with_audio_train_dataloader=DataLoader(
        train_val_split["train"], 
        batch_size=config.train.batch_size, 
        collate_fn=with_audio_collate_fn, 
        shuffle=True)
    
    with_audio_val_dataloader=DataLoader(
        train_val_split["test"], 
        batch_size=config.train.batch_size, 
        collate_fn=with_audio_collate_fn, 
        shuffle=True
    )
    
    return with_audio_train_dataloader, with_audio_val_dataloader



    
            
    

