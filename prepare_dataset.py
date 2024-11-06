import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import matplotlib.pyplot as plt
import numpy as np
import pydub
import polars
import os
import asyncio
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor #take away this shit
import uuid
from pathlib import Path

cache_dir="model"

#TODO speed up by batching:
# 1. concurrent loading of multiple audio files (see ChatGPT quest)
# 2. parallelization of noise injection
# 3. feed batch to whisper model
# 4. optional: span more than one whisper model in different processes
#allow for 50% non modified data

#this is the whisper model used to generate pretraining dataset
DEVICE="mps"
AUXILIARY_MODEL="openai/whisper-small"
whisper=WhisperForConditionalGeneration.from_pretrained(AUXILIARY_MODEL).to(DEVICE)
audio_processor=WhisperProcessor\
    .from_pretrained(AUXILIARY_MODEL, cache_dir=cache_dir)
########################################################################################

def write(audio_arr:np.ndarray, outfile:str, sr:int=16000, normalized:bool=True):
    """numpy array to MP3"""
    channels = 2 if (audio_arr.ndim == 2 and audio_arr.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(audio_arr * 2 ** 15)
    else:
        y = np.int16(audio_arr)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(outfile, format="mp3", bitrate="320k")

def load_audio(audio_path, sr:int=16000)->tuple[np.ndarray, int]:
    return librosa.load(audio_path, sr=sr) #returns (audio np.ndarray, sr int)

# async def load_multiple_audios_concurrently(audio_paths, sr:int=16000):
#     loop=asyncio.get_running_loop()
#     with ProcessPoolExecutor() as executor:
#         tasks = [loop.run_in_executor(executor, load_audio, path, sr) for path in audio_paths]
#         return await asyncio.gather(*tasks)

def clip(audio_arr: np.ndarray, threshold:float=0.1):
    #bigger threshold -> less distortion
    assert audio_arr.ndim==2
    min_vals=audio_arr.min(axis=1)*threshold
    max_vals=audio_arr.max(axis=1)*threshold
    return np.minimum( np.maximum(audio_arr, min_vals[:, None]), max_vals[:, None] )

def reduce_chunk_volume(audio_arr:np.ndarray, reduce_factor:float=0.90):
    assert 0<reduce_factor and reduce_factor<=1
    assert audio_arr.ndim==2 #audio_arr should be (batch_size, timestamps)
    start=np.random.randint(int(audio_arr.shape[1]*0.25), int(audio_arr.shape[1]*0.50))
    len_mask=np.random.uniform(.1,.15)

    mask=np.ones((1, audio_arr.shape[1])) #(1, timestamps)
    mask[:, start:start+int(len_mask*audio_arr.shape[1])]=1-reduce_factor
    return audio_arr*mask

def frequency_modulation(audio_arr:np.ndarray, mod_freq:float=150.0, mod_amplitude:float=2.0, sr:int=16000):
    assert audio_arr.ndim==2
    max_vals:np.ndarray=np.abs(audio_arr).max(axis=1)
    max_vals=max_vals.reshape((-1, 1))

    t = np.stack([np.arange(audio_arr.shape[1]) / sr] * audio_arr.shape[0], axis=0)
    modulation_signal = mod_amplitude* ( max_vals* np.sin(2 * np.pi * mod_freq * t) )
    return audio_arr * (1 + modulation_signal)

def add_noise(audio_arr:np.ndarray, noise_factor:float=0.55):
    assert audio_arr.ndim==2
    noise=np.random.normal(0, noise_factor, size=audio_arr.shape)
    
    #(max + |min|)/2
    max_vals:np.ndarray=( audio_arr.max(axis=1) + np.abs(audio_arr.min(axis=1)) ) /2
    return audio_arr+ (noise*max_vals.reshape(-1, 1) )

def shift_pitch(audio_arr:np.ndarray, steps:int=-3, sr:int=16000):
    return librosa.effects.pitch_shift(audio_arr, sr=sr, n_steps=steps)

def convert_audios_to_input_features(audios:list[np.ndarray],sr:int=16000)->torch.Tensor:
    inputs=audio_processor(
        audios,sampling_rate=sr,return_tensors="pt")["input_features"]
    return inputs

def transcribe_batch(audios:np.ndarray, language:str="italian", sr:int=16000)->list[str]:
    assert audios.ndim==2
    inputs=convert_audios_to_input_features(audios, sr).to(DEVICE)
    forced_decoder_ids = audio_processor.get_decoder_prompt_ids(
            language=language, task="transcribe") #force transcription task in italian
    outs=whisper.generate(inputs, forced_decoder_ids=forced_decoder_ids)
    return audio_processor.batch_decode(outs, skip_special_tokens=True)

def collate_with_zero_pad(audios:list[np.ndarray])->np.ndarray:
    def resize(row, size):
        new = np.array(row)
        new.resize(size)
        return new
    row_length = max(audios, key=len).__len__()
    return np.array( [resize(row, row_length) for row in audios] )

def evenutally_inject_noise(audios:list[np.ndarray],sr:int=16000)->tuple[np.ndarray,int]:
    
    noise_strategy=np.random.randint(0,6)
    noisy_audios:np.ndarray

    if noise_strategy==0: #no extra noise injected, original audio passed 
        noisy_audios= collate_with_zero_pad(audios)
    if noise_strategy==1:
        noisy_audios=add_noise(collate_with_zero_pad(audios), np.random.uniform(0.06, 0.12)) 
    if noise_strategy==2:
        noisy_audios=shift_pitch(collate_with_zero_pad(audios), np.random.randint(-2,3), sr=sr)
    if noise_strategy==3:
        noisy_audios= clip(collate_with_zero_pad(audios), np.random.uniform(0.2, 0.4)) 
    if noise_strategy==4:
        noisy_audios= frequency_modulation(collate_with_zero_pad(audios),np.random.randint(120,150), sr=sr)
    if noise_strategy==5:
        noisy_audios=reduce_chunk_volume(collate_with_zero_pad(audios), np.random.uniform(0.70,0.90)) 

    return noisy_audios, noise_strategy

def read_df_slice(df:polars.DataFrame, batch_size:int, offset:int)->tuple[list[str]]:
    slice=df.slice(offset, batch_size)
    return slice["path"].to_list(), slice["sentence"].to_list()

def write_data_points(
        tsv_file:str, 
        original_sentences:list[str],
        noisy_sentences:list[str],
        audio_names:list[str],
        correct:list[int]|None=None):
    
    if correct is None:
        correct = [int(original.strip()==noisy.strip()) for \
                   original, noisy in zip(original_sentences, noisy_sentences)]
    f=open(tsv_file,"a")
    for o,n,a,c in zip(original_sentences, noisy_sentences, audio_names, correct):
        o=o.strip()
        n=n.strip()
        f.write(f"{o}\t{n}\t{a}\t{c}\n")
    f.close()


def _test_noise_injection(batch_size:int=4, head:int=160):
    audio_samples_index="cv-corpus-19.0-2024-09-13/it/validated.tsv"
    audio_samples_folder="cv-corpus-19.0-2024-09-13/it/clips"
    sampling_rate=16000 #whisper uses this sampling rate

    
    strategy_to_dir_map={0:"none",1:"noise",2:"pitch",3:"clip",4:"freqmod",5:"volume"}
    out_dir="noisy_clips_test"
    for _,v in strategy_to_dir_map.items():
        Path(os.path.join(out_dir, v)).mkdir(parents=True, exist_ok=False)

    df=polars.read_csv(audio_samples_index, separator="\t", quote_char=None).head(head)

    offset=0
    while True:
        audio_names, _ =read_df_slice(df, batch_size, offset)
    
        if not audio_names:
            break

        audio_paths = [os.path.join(audio_samples_folder, name) for name in audio_names]
        audios = asyncio.run( load_multiple_audios_concurrently(audio_paths, sr=sampling_rate) )
        audios = [ a[0] for a in audios ]

        noisy_audios, strategy =evenutally_inject_noise(audios, sr=sampling_rate)

        for a in noisy_audios:
            write(a, os.path.join(out_dir, strategy_to_dir_map[strategy], 
                                  str(uuid.uuid4())[:10].replace("-","")+".mp3"))
        
        offset+=batch_size


    
def construct_tsv_dataset(tsv_file:str, batch_size:int=16, recap:bool=True):

    #index file and samples folder
    audio_samples_index="cv-corpus-19.0-2024-09-13/it/validated.tsv"
    audio_samples_folder="cv-corpus-19.0-2024-09-13/it/clips"
    #######

    logs_file_path="construct_tsv_dataset.logs"

    sampling_rate=16000 #whisper uses this sampling rate

    df=polars.read_csv(audio_samples_index, separator="\t", quote_char=None)

    init_slice:int
    if recap:
        with open("dataset.tsv","r") as f:
            init_slice=len(f.readlines())-1
            df=df.slice(init_slice, None)
    else:
        init_slice=0
    
    offset=0

    
    if not recap:
        #clean tsv dataset and logs
        with open(tsv_file,"w") as f:
            f.write("original\tnoisy\taudio\tcorrect\n")

        with open("construct_tsv_dataset.logs","w") as f:
            pass
    ####

    

    correct:int|None
    while True:

        audio_names, original_sentences=read_df_slice(df, batch_size, offset)

        if not audio_names:
            break

        with open(logs_file_path,"a") as logs_file:
            msg=f"read df slice, line {init_slice+offset} up to {init_slice+offset+batch_size}\n"
            logs_file.write(msg)
            print(msg.strip())

        audio_paths = [os.path.join(audio_samples_folder,name) for name in audio_names]
        audios = asyncio.run( load_multiple_audios_concurrently(audio_paths, sr=sampling_rate) )
        audios = [ a[0] for a in audios ]

        with open(logs_file_path,"a") as logs_file:
            msg=f"read audio files successfully, line {init_slice+offset} up to {init_slice+offset+batch_size}\n"
            logs_file.write(msg)
            print(msg.strip())
        
        correct=None
        if np.random.uniform()>.5:
            noisy_audios,_=evenutally_inject_noise(audios, sr=sampling_rate)
            noisy_sentences=transcribe_batch(noisy_audios,language="italian", sr=sampling_rate)
        else: #50% of data requires correct sentence to train classifier
            correct=[1 for _ in original_sentences]
            noisy_sentences=original_sentences

        write_data_points(tsv_file, original_sentences, noisy_sentences, audio_names, correct)
        
        with open(logs_file_path,"a") as logs_file:
            msg=f"written data points, line {init_slice+offset} up to {init_slice+offset+batch_size}\n"
            logs_file.write(msg)
            print(msg.strip())

        offset+=batch_size

#_test_noise_injection()   
#construct_tsv_dataset("dataset.tsv",recap=True)