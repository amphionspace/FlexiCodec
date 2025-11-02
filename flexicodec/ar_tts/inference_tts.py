#!/usr/bin/env python3
# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import torchaudio
import soundfile as sf
from typing import Dict, Optional, Tuple

# Add paths for imports
import sys
sys.path.append(f'{os.path.dirname(__file__)}/../..')

from flexicodec.ar_tts.modeling_artts import TransformerLMWrapper
from flexicodec.ar_tts.utils.whisper_tokenize import text2idx
from flexicodec.feature_extractors import FBankGen

USE_G2P = True  # Set to True if you want to use G2P phonemization for English text
PARTIAL_G2P = False  # Set to True if you want to use partial G2P phonemization

# Global feature extractor for dualcodec
feature_extractor_for_dualcodec = FBankGen(sr=16000)


def prepare_text_tokens(prompt_text: str, target_text: str, language: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare text tokens for inference using Whisper tokenizer."""
    combined_text = prompt_text + ", " + target_text
    print(f"Preparing text tokens for: {combined_text}")
    if USE_G2P:
        tokens = text2idx(combined_text, language=language, g2p_prob=1.0)
    else:
        tokens = text2idx(combined_text, language=language, g2p_prob=0.0)
    text_tokens = torch.tensor([tokens], dtype=torch.long)
    text_lengths = torch.tensor([len(tokens)])
    return text_tokens, text_lengths

def extract_reference_features(
    model: TransformerLMWrapper, 
    ref_audio_path: str, 
    device: str = 'cuda'
) -> Dict[str, torch.Tensor]:
    """Extract reference audio features using dualcodec"""
    ref_audio, sr = torchaudio.load(ref_audio_path)
    
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        ref_audio = resampler(ref_audio)
    
    if ref_audio.shape[0] > 1:
        ref_audio = ref_audio.mean(dim=0, keepdim=True)
    
    ref_audio = ref_audio.to(device)
    
    # Extract mel features using FBankExtractor
    mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(ref_audio.cpu())
    mel_features = mel_features.to(device)
    
    # Calculate x_lens 
    x_lens = torch.tensor([mel_features.shape[1]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        dualcodec_output = model._extract_dualcodec_features(
            ref_audio, mel=mel_features, x_lens=x_lens, sample_rate=16000
        )
    
    return {
        'semantic_codes': dualcodec_output['semantic_codes'],
        'speech_token_len': dualcodec_output['speech_token_len'],
        'token_lengths': dualcodec_output['token_lengths']
    }

def tts_synthesize(
    ar_model_dict: Dict,
    nar_model_dict: Optional[Dict] = None,
    text: str = "",
    language: str = "en",
    ref_audio_path: Optional[str] = None,
    ref_text: str = "",
    merging_threshold: float = 1.0,
    beam_size: int = 1,
    top_k: int = 25,
    temperature: float = 1.0,
    max_token_text_ratio: float = 20.0,
    min_token_text_ratio: float = 0.0,
    predict_duration: bool = True,
    duration_top_k: int = 1,
    duration_temperature: float = 1.0,
    n_timesteps: int = 30,
    cfg: float = 2.0,
    rescale_cfg: float = 0.75,
    use_nar: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Perform AR TTS inference with optional NAR (Voicebox) decoding.
    
    Args:
        ar_model_dict: Dictionary returned from prepare_artts_model()
        nar_model_dict: Optional dictionary returned from prepare_voicebox_model() for NAR decoding
        text: Target text to synthesize
        language: Language code ('en', 'zh', etc.)
        ref_audio_path: Path to reference audio file
        ref_text: Reference text (optional)
        merging_threshold: Merging threshold/frame rate for inference (must be in flex_framerate_options)
            Same parameter used for both AR and NAR inference - controls FlexiCodec frame rate merging
        beam_size: Beam size for AR decoding
        top_k: Top-k sampling for AR decoding
        temperature: Temperature for AR decoding
        max_token_text_ratio: Maximum token-to-text ratio
        min_token_text_ratio: Minimum token-to-text ratio
        predict_duration: Whether to predict duration classes
        duration_top_k: Top-k for duration sampling
        duration_temperature: Temperature for duration sampling
        n_timesteps: Number of diffusion timesteps for NAR (if use_nar=True)
        cfg: Classifier-free guidance scale for NAR
        rescale_cfg: CFG rescaling factor for NAR
        use_nar: Whether to use NAR decoding (Voicebox) or AR-only (FlexiCodec direct)
    
    Returns:
        tuple: (output_audio_tensor, sample_rate)
    """
    from flexicodec.ar_tts.modeling_artts import infer_artts
    from flexicodec.nar_tts.inference_voicebox import infer_voicebox_tts
    
    ar_model = ar_model_dict['model']
    device = ar_model_dict['device']
    
    # Use infer_artts to generate speech tokens
    speech_tokens, duration_classes, prompt_speech_token = infer_artts(
        ar_model_dict=ar_model_dict,
        text=text,
        language=language,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        merging_threshold=merging_threshold,
        beam_size=beam_size,
        top_k=top_k,
        temperature=temperature,
        max_token_text_ratio=max_token_text_ratio,
        min_token_text_ratio=min_token_text_ratio,
        predict_duration=predict_duration,
        duration_top_k=duration_top_k,
    )
    
    # Extract prompt token lengths for NAR decoding
    if ref_audio_path is None:
        raise ValueError("ref_audio_path is required")
    
    ref_features = extract_reference_features(ar_model, ref_audio_path, device)
    prompt_token_lengths = ref_features.get('token_lengths')
    
    # Decode to audio
    if use_nar and nar_model_dict is not None:
        # Use NAR (Voicebox) for decoding
        # Decode with FlexiCodec to get audio for Voicebox
        if hasattr(ar_model, 'flexicodec_model') and hasattr(ar_model.flexicodec_model, 'decode_from_codes'):
            # Decode to get prompt audio for Voicebox
            prompt_audio = ar_model.flexicodec_model.decode_from_codes(
                semantic_codes=prompt_speech_token.unsqueeze(1),
                acoustic_codes=None,
                token_lengths=prompt_token_lengths,
            )
            
            # Decode generated tokens to get GT audio
            gt_audio = ar_model.flexicodec_model.decode_from_codes(
                semantic_codes=speech_tokens.unsqueeze(1),
                acoustic_codes=None,
                token_lengths=duration_classes,
            )
            
            # Use Voicebox inference
            output_audio, output_sr = infer_voicebox_tts(
                model_dict=nar_model_dict,
                gt_audio=gt_audio.squeeze(1),
                ref_audio=prompt_audio.squeeze(1),
                gt_sample_rate=16000,
                ref_sample_rate=16000,
                n_timesteps=n_timesteps,
                cfg=cfg,
                rescale_cfg=rescale_cfg,
                merging_threshold=merging_threshold,
            )
            return output_audio, output_sr
        else:
            raise ValueError("AR model does not have FlexiCodec decoder for Voicebox integration")
    
    else:
        # AR-only decoding with FlexiCodec
        if hasattr(ar_model, 'flexicodec_model') and hasattr(ar_model.flexicodec_model, 'decode_from_codes'):
            decoded_audio = ar_model.flexicodec_model.decode_from_codes(
                semantic_codes=speech_tokens.unsqueeze(1),
                acoustic_codes=None,
                token_lengths=duration_classes,
            )
            return decoded_audio.cpu().squeeze(), 16000
        else:
            raise ValueError("AR model does not have decode_from_codes method")

if __name__ == "__main__":
    # Test example
    from flexicodec.ar_tts.modeling_artts import prepare_artts_model
    from flexicodec.nar_tts.inference_voicebox import prepare_voicebox_model
    
    # Example usage
    ar_checkpoint = "/Users/jiaqi/github/FlexiCodec/artts.safetensors"
    nar_checkpoint = "/Users/jiaqi/github/FlexiCodec/nartts.safetensors"
    
    # Prepare models
    ar_model_dict = prepare_artts_model(ar_checkpoint)
    nar_model_dict = prepare_voicebox_model(nar_checkpoint)
    
    # Run inference
    audio, sr = tts_synthesize(
        ar_model_dict=ar_model_dict,
        nar_model_dict=nar_model_dict,
        text="Hello, this is a test.",
        language="en",
        ref_audio_path="/Users/jiaqi/github/FlexiCodec/audio_examples/61-70968-0000_ref.wav",
        ref_text="",
        merging_threshold=0.91,
        use_nar=True,
    )
    
    # Save output
    torchaudio.save("output.wav", audio.reshape(1, -1), sr)
    print(f"Saved output to output.wav")