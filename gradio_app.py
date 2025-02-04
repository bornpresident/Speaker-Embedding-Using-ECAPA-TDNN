import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.spatial.distance import cdist
import gradio as gr

class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, f"{channels} % {scale} != 0"
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.width)
            for _ in range(self.nums)
        ])

    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)
        return out

class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))

class SE_Connect(nn.Module):
    def __init__(self, channels, se_channels=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_channels)
        self.linear2 = nn.Linear(se_channels, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)

class ECAPA_TDNN(nn.Module):
    def __init__(self, in_channels=80, channels=512, emb_dim=192):
        super().__init__()
        
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=3, padding=2, dilation=2),
            SE_Connect(channels)
        )
        self.layer3 = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=3, padding=3, dilation=3),
            SE_Connect(channels)
        )
        self.layer4 = nn.Sequential(
            Conv1dReluBn(channels, channels, kernel_size=3, padding=4, dilation=4),
            SE_Connect(channels)
        )
        
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(cat_channels)
        self.linear = nn.Linear(cat_channels, emb_dim)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.pooling(out).squeeze(-1)
        out = self.bn(out)
        out = self.linear(out)
        return F.normalize(out, p=2, dim=1)

def process_audio(audio_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Extract mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80
    )
    
    mel_spec = mel_transform(waveform)
    mel_spec = (mel_spec + torch.finfo(torch.float).eps).log()
    return mel_spec

class SpeakerVerification:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ECAPA_TDNN().to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def compute_similarity(self, audio_path1, audio_path2):
        with torch.no_grad():
            mel1 = process_audio(audio_path1).to(self.device)
            mel2 = process_audio(audio_path2).to(self.device)
            
            emb1 = self.model(mel1)
            emb2 = self.model(mel2)
            
            similarity = F.cosine_similarity(emb1, emb2)
            return similarity.item()

def compare_speakers(audio1, audio2):
    verifier = SpeakerVerification()
    similarity = verifier.compute_similarity(audio1, audio2)
    return f"Similarity Score: {similarity:.4f}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="text-align: center; max-height: 300px;">
            <img src="https://d1fssb5kuizyrz.cloudfront.net/images/BharatGen Logo.png" 
                 style="max-height: 150px; margin: auto;">
            <h1 style="margin-top: 1rem;">Speaker Identification System</h1>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input1 = gr.Audio(label="Upload First Audio", type="filepath")
        with gr.Column():
            audio_input2 = gr.Audio(label="Upload Second Audio", type="filepath")
            
    with gr.Row():
        submit_btn = gr.Button("Compare Speakers")
        
    output_text = gr.Textbox(label="Results")
    
    submit_btn.click(
        fn=compare_speakers,
        inputs=[audio_input1, audio_input2],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch()
