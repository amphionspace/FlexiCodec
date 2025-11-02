import json
import os
import shutil
import torch
import time
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from cosyvoice.utils.amphion.base_trainer import BaseTrainer
import safetensors
import numpy as np
from einops import rearrange
import math
import torch.nn.functional as F
import torchaudio

from cosyvoice.utils.mask import make_pad_mask

class Trainer(BaseTrainer):
    """Trainer class for LLM Duration models with DualCodec feature extraction"""

    def __init__(self, args=None, cfg=None, **kwargs):
        """
        Initializes the trainer with DualCodec model for feature extraction.

        Args:
            args (argparse.Namespace, optional): Arguments to be passed on to the model. Defaults to None.
            cfg (dict, optional): Configuration dictionary containing parameters for the model. Defaults to None.
        """
        super().__init__(args, cfg)

        # Initialize DualCodec model for feature extraction
        print("Preparing DualCodec model for feature extraction...")
        self.dualcodec_model = prepare_model()
        print(f"DualCodec model loaded, type: {self.dualcodec_model.get('type', 'unknown')}")
        
    def _accelerator_prepare(self):
        """
        Prepares the model and optimizer for distributed training.
        
        Returns: None
        """
        (
            self.model,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
        )

    def _build_scheduler(self):
        """
        Builds the learning rate scheduler.
        
        Returns: None
        """
        return None

    def _build_model(self):
        """
        Builds the LLM Duration model from configuration.
        
        Returns: The constructed model
        """
        from d2codec_tts.llm_duration import create_transformer_lm_from_config
        self.model = create_transformer_lm_from_config()

        return self.cfg.model

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def _extract_dualcodec_features(self, speech, mel=None, sample_rate=16000):
        """
        Extracts features using DualCodec model with batch inference.
        
        Args:
            speech (torch.Tensor): Speech audio [B, T]
            mel (torch.Tensor, optional): Mel spectrogram features [B, T, D]
            sample_rate (int): Sample rate of the audio
            
        Returns:
            dict: Dictionary containing extracted features and codes
        """
        device = speech.device
        
        # Check if DualCodec model is available
        if not hasattr(self, 'dualcodec_model') or self.dualcodec_model is None:
            raise RuntimeError("DualCodec model not initialized. Please ensure prepare_model() was called successfully.")
        
        codec_model = self.dualcodec_model['model']
        feature_extractor = self.dualcodec_model['feature_extractor']
        
        # Ensure audio is on the correct device
        speech = speech.to(device)
        
        # 1. Resample audio: 16kHz for semantic features
        resampler_16k = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
        audio_16k = resampler_16k(speech)  # [B, T_16k]
        
        # 2. Extract features based on model type
        try:
            if self.dualcodec_model.get('type') == 'sensevoice':
                # Use SenseVoice's FBankGen (on CPU)
                # Note: FBankGen might not support batch processing, so we process one by one
                features_list = []
                for i in range(audio_16k.shape[0]):
                    features, _ = feature_extractor.extract_fbank(audio_16k[i:i+1].cpu())
                    features_list.append(features)
                audio_features = torch.cat(features_list, dim=0).to(device)
            else:
                # Use SeamlessM4TFeatureExtractor with batch processing
                try:
                    # Try batch processing first
                    features = feature_extractor(audio_16k.cpu(), return_tensors="pt", sampling_rate=16000)
                    audio_features = features.input_features.to(device)
                except Exception as batch_error:
                    if self.debug_dualcodec:
                        print(f"Batch feature extraction failed, falling back to individual processing: {batch_error}")
                    # Fallback to individual processing if batch fails
                    features_list = []
                    for i in range(audio_16k.shape[0]):
                        features = feature_extractor(audio_16k[i:i+1].cpu(), return_tensors="pt", sampling_rate=16000)
                        features_list.append(features.input_features)
                    audio_features = torch.cat(features_list, dim=0).to(device)
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            print(f"Audio shape: {audio_16k.shape}, device: {device}")
            raise
        
        # 3. Prepare input for DualCodec model
        # Validate mel features if provided
        if mel is not None:
            if mel.shape[0] != speech.shape[0]:
                raise ValueError(f"Mel batch size mismatch: mel {mel.shape[0]} vs speech {speech.shape[0]}")
            if self.debug_dualcodec:
                print(f"Mel features provided: {mel.shape}")
        
        dl_output = {
            "audio": audio_16k,
            "x": audio_features,
            "num_quantizers": 8,
            "mel": mel,  # Pass mel features if provided
        }
        
        # 4. Encode the audio to get semantic and acoustic codes
        try:
            encoded_output = codec_model(
                dl_output,
                encode_only=True,
            )
        except Exception as e:
            print(f"Error during DualCodec encoding: {e}")
            print(f"Input shapes - audio: {audio_16k.shape}, features: {audio_features.shape}")
            if mel is not None:
                print(f"Mel shape: {mel.shape}")
            raise
        
        # 5. Extract the codes and token lengths
        semantic_codes = encoded_output['semantic_codes']  # [B, T]
        acoustic_codes = encoded_output.get('acoustic_codes', None)
        token_lengths = encoded_output.get('token_lengths', None)  # [B, G] or None
        semantic_features = encoded_output.get('semantic_features', None)
        
        # Validate output shapes
        if semantic_codes.shape[0] != speech.shape[0]:
            raise ValueError(f"Batch size mismatch: semantic_codes {semantic_codes.shape[0]} vs speech {speech.shape[0]}")
        
        # Debug print (can be disabled in production)
        if hasattr(self, 'debug_dualcodec') and self.debug_dualcodec:
            print(f"DualCodec output - semantic_codes: {semantic_codes.shape}, token_lengths: {token_lengths.shape if token_lengths is not None else None}")
        
        # 6. Handle token_lengths (duration) - convert to per-token duration if needed
        if token_lengths is not None:
            # If token_lengths is [B, G], we need to expand it to match semantic_codes [B, T]
            if len(token_lengths.shape) == 2:
                # This means we have per-group durations, need to expand to per-token
                # Use alignment matrix if available to properly expand durations
                alignment_matrix = encoded_output.get('alignment_matrix', None)
                if alignment_matrix is not None:
                    # Use alignment matrix to expand token_lengths to per-token durations
                    expanded_token_lengths = []
                    for i in range(token_lengths.shape[0]):
                        # For each batch item, expand group durations to token durations
                        item_token_lengths = []
                        for j in range(semantic_codes.shape[1]):  # For each token
                            # Find which group this token belongs to
                            group_idx = alignment_matrix[i, :, j].argmax().item()
                            # Use the duration of that group
                            group_duration = token_lengths[i, group_idx] if group_idx < token_lengths.shape[1] else 1
                            item_token_lengths.append(group_duration)
                        expanded_token_lengths.append(torch.tensor(item_token_lengths, device=device, dtype=token_lengths.dtype))
                    token_lengths = torch.stack(expanded_token_lengths, dim=0)  # [B, T]
                else:
                    # Fallback: use the first group's duration for all tokens
                    expanded_token_lengths = []
                    for i in range(token_lengths.shape[0]):
                        group_duration = token_lengths[i, 0] if token_lengths[i].numel() > 0 else 1
                        expanded_token_lengths.append(torch.full((semantic_codes.shape[1],), group_duration, 
                                                               device=device, dtype=token_lengths.dtype))
                    token_lengths = torch.stack(expanded_token_lengths, dim=0)  # [B, T]
            else:
                # Already in the right format [B, T]
                pass
        else:
            # If no token_lengths available, use semantic code sequence length
            token_lengths = torch.tensor([semantic_codes.shape[1]] * semantic_codes.shape[0], 
                                       device=device, dtype=torch.long)
        
        return {
            'semantic_codes': semantic_codes,  # [B, T] - speech tokens
            'token_lengths': token_lengths,    # [B, T] - duration info for each speech token
            'semantic_features': semantic_features,  # [B, T, D] if available
            'acoustic_codes': acoustic_codes,  # [B, n_q, T] if available
        }



    def _train_step(self, batch):
        """
        Performs a single training step with a batch of data.
        
        Args:
            batch (dict): Dictionary containing batch data
            
        Returns:
            tuple: (total_loss, train_losses, train_stats)
        """
        device = self.accelerator.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        # Get input data from batch
        speech = batch["speech"]  # [B, T]
        text_ids = batch["text_token"]  # [B, L]
        text_len = batch["text_token_len"]  # [B]
        mel = batch.get("mel", None)  # [B, T, D] - mel spectrogram features if available
        
        # Extract features using DualCodec
        dualcodec_output = self._extract_dualcodec_features(speech, mel=mel)
        
        # Get semantic codes (speech tokens) and duration info
        speech_tokens = dualcodec_output['semantic_codes']  # [B, T]
        token_lengths = dualcodec_output['token_lengths']   # [B, T] - duration for each speech token
        
        # Get actual speech token lengths (sequence lengths)
        speech_token_len = torch.tensor([speech_tokens.shape[-1]] * speech_tokens.shape[0], 
                                       device=device, dtype=torch.long)
        
        # Create duration targets - for simplicity, use uniform duration classes
        # In practice, you might want to quantize the token_lengths into duration classes
        if token_lengths is not None:
            # token_lengths is now [B, T] - per-token durations
            # Quantize continuous token lengths into discrete duration classes (0-9)
            max_duration = token_lengths.max().item() if token_lengths.numel() > 0 else 1
            duration_classes = (token_lengths * 9 / max_duration).clamp(0, 9).long()
            # duration_classes is already [B, T], no need to expand
            duration = duration_classes
        else:
            # If no token_lengths available, use dummy duration classes
            duration = torch.randint(0, 10, (speech_tokens.shape[0], speech_tokens.shape[-1]), 
                                   device=device, dtype=torch.long)
        
        # Prepare batch for LLM model
        model_batch = {
            "text_token": text_ids,              # [B, L]
            "text_token_len": text_len,          # [B]
            "speech_token": speech_tokens,       # [B, T]
            "speech_token_len": speech_token_len, # [B]
            "duration": duration,                # [B, T] - duration classes for each speech token
            "embedding": batch.get("embedding", None)  # [B, embed_dim] if available
        }
        
        # Forward pass through LLM Duration model
        result = self.model(model_batch, device)
        total_loss = result["loss"]  # Combined speech token + duration loss
        acc = result["acc"]          # Speech token accuracy
        
        metrics = {"accuracy": acc}
        
        return total_loss, metrics

    def _test_step(self, batch):
        """
        Performs a test step with a batch of data.
        
        Args:
            batch (dict): Dictionary containing batch data
        """
        raise NotImplementedError

    @torch.inference_mode()
    def _valid_epoch(self):
        """
        Validation epoch function.
        
        Returns:
            float: Average validation loss
        """
        epoch_sum_loss = 0.0
        return epoch_sum_loss

    def _inference(self):
        """
        Performs inference with the model.
        """
        pass

    def test_loop(self):
        """
        Test loop for the model.
        """
        return
        self.model.eval()
        for batch in self.train_dataloader:
            self._test_step(batch)
