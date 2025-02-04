import json
from pydub import AudioSegment
import librosa
import os
import torch
import torchaudio
import numpy as np
from spk_encoder.speaker_encoder.ecapa_tdnn import ECAPA_TDNN_SMALL
from spk_encoder.util import HParams, fix_len_compatibility, process_unit, generate_path, sequence_mask
from scipy.spatial.distance import cdist
from torchaudio.transforms import Resample

class SpeakerEncoder:
    def __init__(self, speaker_encoder_path, config_path):
        # Load configuration
        with open(config_path, "r") as f:
            data = f.read()
        config = json.loads(data)
        self.hps = HParams(**config)

        print('Initializing Speaker Encoder...')
        self.spk_embedder = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state_dict = torch.load(speaker_encoder_path, map_location=lambda storage, loc: storage)
        self.spk_embedder.load_state_dict(state_dict['model'], strict=False)
        self.spk_embedder = self.spk_embedder.cuda().eval()

        
    def get_spk_embd(self, audio, sr):
        # Assuming audio is a Torch tensor and already in the correct format
        # Resample wav to 16000 Hz if needed
        if sr != 16000:
            resample_fn = torchaudio.transforms.Resample(sr, 16000).cuda()
            audio = resample_fn(audio.to("cuda"))
            sr = 16000
            
        
        # Compute speaker embedding
        spk_emb = self.spk_embedder(audio)
        spk_emb = spk_emb / spk_emb.norm()

        return spk_emb

def similarity_matrix(spk_emds):
    spk_emds = np.stack([emb.squeeze(0).cpu().detach().numpy() for emb in spk_emds])
    cosine_distances = cdist(spk_emds, spk_emds, metric='cosine')

    cosine_similarities = 1 - cosine_distances
    
    return cosine_similarities

def process_audio(file_path, target_sample_rate=16000, max_duration=60):
    
    waveform, sample_rate = torchaudio.load(file_path)

    max_samples = max_duration * sample_rate
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    if sample_rate != target_sample_rate:
        resample = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample(waveform)
        sample_rate = target_sample_rate

    return waveform, sample_rate

speaker_encoder_path = "/workspace/Speech/speaker_identification/spk_encoder/speaker_encoder.pt"  
config_path = "/workspace/Speech/speaker_identification/spk_encoder/finetune.json"                        
encoder = SpeakerEncoder(speaker_encoder_path, config_path)

audio_path1 = "/workspace/Speech/speaker_identification/sample_audio/v2-abhay.wav"
audio_path2 = "/workspace/Speech/speaker_identification/sample_audio/v2-pranav.wav" 
 
                       
audio1, sr1 = process_audio(audio_path1)
audio2, sr2 = process_audio(audio_path2)

spk_emb1 = encoder.get_spk_embd(audio1.cuda(), sr1)
spk_emb2 = encoder.get_spk_embd(audio2.cuda(), sr2)
print(similarity_matrix([spk_emb1, spk_emb2]))
