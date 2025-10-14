# FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates

[![Demo Page](https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=Github&style=flat-square)](https://flexicodec.github.io/)
[![ArXiv](https://img.shields.io/badge/arxiv-PDF-green?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2510.00981)

## Abstract
Neural audio codecs are foundational to speech language models. It is expected to have a low frame rate and decoupled semantic and acoustic information. A lower frame rate codec can reduce the computational cost of speech language models by shortening the sequence length. Recent studies have developed 12.5Hz low-frame-rate audio codecs, but even lower frame rate codecs remain underexplored. We find that a major challenge for very low frame rate tokens is missing semantic information. This paper introduces FlexiCodec to address this limitation. FlexiCodec improves semantic preservation with a dynamic frame rate approach and introduces a novel architecture featuring an ASR feature-assisted dual stream encoding and Transformer bottlenecks. With dynamic frame rates, it uses less frames at information-sparse regions through adaptively merging semantically similar frames. A dynamic frame rate also allows FlexiCodec to support inference-time controllable frame rates between 3Hz and 12.5Hz. Experiments on 6.25Hz, 8.3Hz and 12.5Hz average frame rates confirm that FlexiCodec excels over baseline systems in semantic information preservation and delivers a high audio reconstruction quality. We also validate the effectiveness of FlexiCodec in language model-based TTS.


## Installation
```bash
git clone https://github.com/amphionspace/FlexiCodec.git
pip install -r requirements.txt
```
<!-- # pip install -e . -->

## FlexiCodec
Code is available under [`flexicodec/modeling_flexicodec.py`](flexicodec/modeling_flexicodec.py). Inference example:
```python
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

duration = audio.shape[-1] / 16000

output_path = 'decoded_audio.wav'
torchaudio.save(output_path, reconstructed_audio.cpu().squeeze(1), 16000)

print(f"Saved decoded audio to {output_path}")
print(f"This sample avg frame rate: {encoded_output['token_lengths'].shape[-1] / duration:.4f} frames/sec")
```
Batched input is supported.

## FlexiCodec-TTS
Our code for Flexicodec-based AR TTS is available at [`flexicodec/ar_tts/modeling_artts.py`](flexicodec/ar_tts/modeling_artts.py).
Our code for Flow matching-based NAR TTS is based on the voicebox-based implementation [here](https://github.com/jiaqili3/DualCodec/tree/main/dualcodec/model_tts/voicebox).

## Acknowledgements & Citation
- Our codebase setup is based on [DualCodec](https://github.com/jiaqili3/DualCodec)
- We thank the [Mimi Codec](https://github.com/kyutai-labs/moshi) for transformer implementations

If you find our work useful, please consider citing as:
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