# FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates

[![Demo Page](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://flexicodec.github.io/)
[![ArXiv](https://img.shields.io/badge/arXiv-PDF-green?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2510.00981)


## Abstract
Neural audio codecs are foundational to speech language models. It is expected to have a low frame rate and decoupled semantic and acoustic information. A lower frame rate codec can reduce the computational cost of speech language models by shortening the sequence length. Recent studies have developed 12.5Hz low-frame-rate audio codecs, but even lower frame rate codecs remain underexplored. We find that a major challenge for very low frame rate tokens is missing semantic information. This paper introduces FlexiCodec to address this limitation. FlexiCodec improves semantic preservation with a dynamic frame rate approach and introduces a novel architecture featuring an ASR feature-assisted dual stream encoding and Transformer bottlenecks. With dynamic frame rates, it uses less frames at information-sparse regions through adaptively merging semantically similar frames. A dynamic frame rate also allows FlexiCodec to support inference-time controllable frame rates between 3Hz and 12.5Hz. Experiments on 6.25Hz, 8.3Hz and 12.5Hz average frame rates confirm that FlexiCodec excels over baseline systems in semantic information preservation and delivers a high audio reconstruction quality. We also validate the effectiveness of FlexiCodec in language model-based TTS.
![](.github/flexicodec.png)

## Installation
```bash
git clone https://github.com/amphionspace/FlexiCodec.git
cd FlexiCodec
pip install -r requirements.txt
```
<!-- # pip install -e . -->

## FlexiCodec
Code is available under [`flexicodec/modeling_flexicodec.py`](flexicodec/modeling_flexicodec.py). 

To run inference (automatically downloads checkpoint from huggingface):
```python
import torch
import torchaudio
from flexicodec.infer import prepare_model, encode_flexicodec

model_dict = prepare_model()
  
# Load a real audio file
audio_path = "YOUR_WAV.wav"
audio, sample_rate = torchaudio.load(audio_path)
with torch.no_grad():
    encoded_output = encode_flexicodec(audio, model_dict, sample_rate, num_quantizers=8, merging_threshold=0.91)
    
    reconstructed_audio = model_dict['model'].decode_from_codes(
        semantic_codes=encoded_output['semantic_codes'],
        acoustic_codes=encoded_output['acoustic_codes'],
        token_lengths=encoded_output['token_lengths'],
    )

duration = audio.shape[-1] / sample_rate
output_path = 'decoded_audio.wav'
torchaudio.save(output_path, reconstructed_audio.cpu().squeeze(1), 16000)

print(f"Saved decoded audio to {output_path}")
print(f"This sample avg frame rate: {encoded_output['token_lengths'].shape[-1] / duration:.4f} frames/sec")
```

Notes:
- You may tune the `num_quantizers=xxx` (maximum 24), `merging_threshold=xxx` (maximum 1.0) parameters. If you set `merging_threshold=1.0`, it will be a standard 12.5Hz neural audio codec. All of its `token_lengths` items will be 1. 

- For mainland China users, you might need to execute `export HF_ENDPOINT=https://hf-mirror.com` in terminal, before running the code. If you don't want to automatically download from huggingface, you can manually specify your downloaded checkpoint paths [![Huggingface](https://img.shields.io/badge/huggingface-yellow?logo=huggingface&style=flat-square)](https://huggingface.co/jiaqili3/flexicodec/tree/main) in `prepare_model`. 


- Batched input is supported. You can directly pass audios shaped [B,T] to the script above, but the audio length information will be unavailable.
To resolve this, you can additionally pass an `audio_lens` parameter to `encode_flexicodec`, and you can crop the output for each audio in `encoded_output[speech_token_len]`. 

- If you want to use the above code elsewhere, you might want to add `sys.path.append('/path/to/FlexiCodec')` to find the code.

- To extract continuous features from the semantic tokens, use:
  ```python
  feat = model_dict['model'].get_semantic_feature(encoded_output['semantic_codes'])
  ```

## FlexiCodec-TTS
First, install additional dependencies:
```bash
pip install cached_path
```

Our code for Flexicodec-based AR TTS is available at [`flexicodec/ar_tts/modeling_artts.py`](flexicodec/ar_tts/modeling_artts.py). The training step is inside `training_forward` method. It receives a `dl_output` dictionary containing `x` (the [`feature_extractor`](flexicodec/infer.py#L50) output), `x_lens` (length of each x before padding), `audio` (the 16khz audio tensor). The inference is at the `inference` method in the same file.

Our code for Flow matching-based NAR TTS is based on the voicebox-based implementation [here](https://github.com/jiaqili3/DualCodec/tree/main/dualcodec/model_tts/voicebox).

### FlexiCodec-based Voicebox NAR TTS Inference
The VoiceBox NAR system can decode FlexiCodec's RVQ-1 tokens into speech. It is used as the second stage in FlexiCodec-TTS, but can also be used standalone.
To run NAR TTS inference using FlexiCodec-Voicebox:

```python
import torch
import torchaudio
from flexicodec.nar_tts.inference_voicebox import (
    prepare_voicebox_model, 
    infer_voicebox_tts
)
import cached_path
# Prepare model (loads model and vocoder)
checkpoint_path = cached_path('hf://jiaqili3/flexicodec/nartts.safetensors')
model_dict = prepare_voicebox_model(checkpoint_path)

# Option 1: Inference with audio file paths
gt_audio_path = "audio_examples/61-70968-0000_gt.wav"  # Target content. Example GT audio
ref_audio_path = "audio_examples/61-70968-0000_ref.wav"     # Reference voice/style. 

output_audio, output_sr = infer_voicebox_tts(
    model_dict=model_dict,
    gt_audio_path=gt_audio_path,
    ref_audio_path=ref_audio_path,
    n_timesteps=15,          # Number of diffusion steps (default: 15)
    cfg=2.0,                 # Classifier-free guidance scale (default: 2.0)
    rescale_cfg=0.75,        # CFG rescaling factor (default: 0.75)
    merging_threshold=1.0    # Merging threshold for frame rate control (default: 1.0, max: 1.0)
)

# Save output
torchaudio.save("output.wav", output_audio.unsqueeze(0) if output_audio.dim() == 1 else output_audio, output_sr)

# Option 2: Inference with audio tensors
gt_audio, gt_sr = torchaudio.load("path/to/ground_truth.wav")
ref_audio, ref_sr = torchaudio.load("path/to/reference.wav")

output_audio, output_sr = infer_voicebox_tts(
    model_dict=model_dict,
    gt_audio=gt_audio,
    ref_audio=ref_audio,
    gt_sample_rate=gt_sr,
    ref_sample_rate=ref_sr,
    n_timesteps=15,
    cfg=2.0,
    rescale_cfg=0.75,
    merging_threshold=1.0
)
```

**Notes:**
- The model automatically detects and uses CUDA, MPS (Apple Silicon), or CPU devices
- Ground truth audio (`gt_audio`) determines the semantic content of the output
- Reference audio (`ref_audio`) determines the voice/style characteristics
- Output sample rate is typically 16000 Hz or 24000 Hz depending on the model configuration
- You can reuse `model_dict` for multiple inference calls to avoid reloading the model
- `merging_threshold` controls FlexiCodec's dynamic frame rate: lower values (e.g., 0.87, 0.91) enable merging for lower average frame rates, while 1.0 disables merging (standard 12.5Hz)


## Acknowledgements & Citation
- Our codebase setup is based on [DualCodec](https://github.com/jiaqili3/DualCodec)
- We thank the [Mimi Codec](https://github.com/kyutai-labs/moshi) for transformer implementations

If you find our works useful, please consider citing as:
```biblatex
@article{li2025flexicodec,
  title={FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates},
  author={Li, Jiaqi and Qian, Yao and Hu, Yuxuan and Zhang, Leying and Wang, Xiaofei and Lu, Heng and Thakker, Manthan and Li, Jinyu and Zhao, Shang and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2510.00981},
  year={2025}
}

@article{li2025dualcodec,
  title={Dualcodec: A low-frame-rate, semantically-enhanced neural audio codec for speech generation},
  author={Li, Jiaqi and Lin, Xiaolong and Li, Zhekai and Huang, Shixi and Wang, Yuancheng and Wang, Chaoren and Zhan, Zhenpeng and Wu, Zhizheng},
  journal={Interspeech 2025},
  year={2025}
}
```