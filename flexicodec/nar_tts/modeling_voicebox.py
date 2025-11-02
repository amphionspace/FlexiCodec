import random
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torchaudio
import yaml
import math
from .voicebox_models import VoiceBox, voicebox_300M, extract_normalized_mel_spec_50hz

from flexicodec.infer import prepare_model

class VoiceboxWrapper(nn.Module):
    """
    Wrapper class that integrates VoiceBox model with FlexiCodec feature extraction
    """
    
    def __init__(
        self,
        voicebox_config: Optional[Dict] = None,
        mel_mean: float = -4.92,
        mel_var: float = 8.14,
        sigma: float = 1e-5,
        use_repa_loss: bool = True,
        repa_loss_weight: float = 1.0,
        repa_loss_type: str = "cosine",
        repa_layer_idx: int = 9,
        repa_projection_dim: int = 512,
        ssl_feature_dim: int = 512,
        use_decoder_latent: bool = False,
        decoder_latent_pass_transformer: bool = True,
        use_decoder_latent_before_agg: bool = False,
        infer_using_dynamic_threshold=False,
        flex_framerate: bool = False,
        flex_framerate_options: list = [0.87,0.91,1.0],
        add_framerate_embedding: bool = False,
        concat_speaker_embedding: bool = False,
        **kwargs
    ):
        """
        Initialize VoiceboxWrapper
        
        Args:
            voicebox_config: Optional config dict for VoiceBox model
            mel_mean: Mean for mel feature normalization
            mel_var: Variance for mel feature normalization  
            sigma: Sigma parameter for flow matching
            use_repa_loss (bool): Whether to use REPA loss. Defaults to False.
            repa_loss_weight (float): Weight for REPA loss. Defaults to 1.0.
            repa_loss_type (str): Type of REPA loss ('cosine', 'l1', 'l2'). Defaults to "cosine".
            repa_layer_idx (int): Layer index from VoiceBox for REPA loss. Defaults to 9.
            repa_projection_dim (int): Dimension of the REPA projection. Defaults to 512.
            voicebox_hidden_dim (int): Hidden dimension of Voicebox model. Defaults to 1024.
            ssl_feature_dim (int): Dimention of SSL features. Defaults to 512.
            concat_speaker_embedding (bool): Whether to concatenate speaker embedding to condition features. Defaults to False.
        """
        super().__init__()
        
        print("Preparing FlexiCodec model for feature extraction...")
        self.flexicodec_model = prepare_model()['model']
        
        # Freeze FlexiCodec model parameters
        for param in self.flexicodec_model.parameters():
            param.requires_grad = False
        self.flexicodec_model.eval()
        # Initialize VoiceBox model
        if voicebox_config is not None:
            self.voicebox_model = VoiceBox(**voicebox_config)
        else:
            # Use default 300M model
            self.voicebox_model, self.mel_spec_model = voicebox_300M()
        
        self.cond_scale_factor = voicebox_config['cond_scale_factor']

        # Store normalization parameters
        self.mel_mean = mel_mean
        self.mel_var = mel_var
        self.sigma = float(sigma)
        self.use_decoder_latent = use_decoder_latent
        self.use_decoder_latent_before_agg = use_decoder_latent_before_agg
        self.decoder_latent_pass_transformer = decoder_latent_pass_transformer
        self.infer_using_dynamic_threshold = infer_using_dynamic_threshold
        self.flex_framerate = flex_framerate
        self.flex_framerate_options = flex_framerate_options
        self.add_framerate_embedding = add_framerate_embedding
        self.concat_speaker_embedding = concat_speaker_embedding
        
        if self.add_framerate_embedding:
            assert self.flex_framerate, "add_framerate_embedding requires flex_framerate to be True"
            self.framerate_embedding = nn.Embedding(
                len(self.flex_framerate_options),
                voicebox_config['hidden_size']
            )
        
        # Initialize speaker embedding model if enabled
        if self.concat_speaker_embedding:
            print("Loading WeSpeaker model for speaker embedding extraction...")
            self.speaker_model = wespeaker.load_model("english")
            self.speaker_model.set_device("cuda" if torch.cuda.is_available() else "cpu")
            # Freeze speaker model parameters
            for param in self.speaker_model.model.parameters():
                param.requires_grad = False
            self.speaker_model.model.eval()

            self.spk_linear = nn.Linear(voicebox_config['hidden_size']+256, voicebox_config['hidden_size'])
        
        self.use_repa_loss = use_repa_loss
        if self.use_repa_loss:
            self.repa_loss_weight = repa_loss_weight
            self.repa_loss_type = repa_loss_type
            self.repa_layer_idx = repa_layer_idx
            self.repa_projection = nn.Sequential(
                nn.Linear(voicebox_config['hidden_size'], ssl_feature_dim * 4),
                nn.SiLU(),
                nn.Linear(ssl_feature_dim * 4, repa_projection_dim),
            )
        
        self.trainer_callbacks = []
        self.step = 0
    
    @property
    def dualcodec_model(self):
        """Alias for flexicodec_model for backward compatibility"""
        return self.flexicodec_model
    
    def compute_repa_loss(self, hidden_features_downsampled, target_repr, mask=None):
        """
        Compute REPA (Representation Alignment) loss between VoiceBox hidden features and SenseVoice representations.
        
        Args:
            hidden_features_downsampled: torch.Tensor, shape (B, D, T) - Hidden features from VoiceBox (3x downsampled)
            target_repr: torch.Tensor, shape (B, D, T) - Target SenseVoice representation
            mask: torch.Tensor, shape (B, T) - Optional mask for valid tokens (original length)
            
        Returns:
            torch.Tensor: REPA loss value
        """
        # Ensure both tensors have the same shape
        if hidden_features_downsampled.shape[-1] != target_repr.shape[-1]:
            # Assert shape difference is at most 2
            assert abs(hidden_features_downsampled.shape[-1] - target_repr.shape[-1]) <= 2
            min_len = min(hidden_features_downsampled.shape[-1], target_repr.shape[-1])
            hidden_features_downsampled = hidden_features_downsampled[..., :min_len]
            target_repr = target_repr[..., :min_len]
        
        # Compute REPA loss based on the specified type
        if self.repa_loss_type == "cosine":
            # Cosine similarity loss (maximize similarity)
            # Normalize along the feature dimension (dim=1 for B, D, T format)
            hidden_norm = F.normalize(hidden_features_downsampled, dim=1)
            target_norm = F.normalize(target_repr, dim=1)
            cosine_sim = F.cosine_similarity(hidden_norm, target_norm, dim=1)  # (B, T)
            
            if mask is not None:
                # Downsample mask to match the downsampled features
                if mask.shape[-1] != hidden_features_downsampled.shape[-1]:
                    mask_downsampled = F.interpolate(
                        mask.unsqueeze(1).float(), 
                        size=hidden_features_downsampled.shape[-1], 
                        mode='linear', 
                        align_corners=False
                    ).squeeze(1).bool()
                else:
                    mask_downsampled = mask
                
                repa_loss = (1 - cosine_sim) * mask_downsampled.float()
                repa_loss = repa_loss.sum() / mask_downsampled.sum().clamp(min=1)
            else:
                repa_loss = (1 - cosine_sim).mean()
        elif self.repa_loss_type == "l2":
            # L2 distance loss
            repa_loss = F.mse_loss(hidden_features_downsampled, target_repr)
        elif self.repa_loss_type == "l1":
            # L1 distance loss
            repa_loss = F.l1_loss(hidden_features_downsampled, target_repr)
        else:
            raise ValueError(f"Unknown REPA loss type: {self.repa_loss_type}")
        
        return repa_loss * self.repa_loss_weight

    @torch.no_grad()
    @torch.autocast('cuda', enabled=False)
    def _extract_dualcodec_features(self, speech, mel=None, x_lens=None, sample_rate=16000, infer_using_dynamic_threshold=False, manual_threshold=None):
        """
        Extract features using DualCodec model with batch inference
        
        Args:
            speech (torch.Tensor): Speech audio [B, T]
            mel (torch.Tensor, optional): Mel spectrogram features [B, T, D]
            x_lens (torch.Tensor, optional): Sequence lengths [B]
            sample_rate (int): Sample rate of the audio
            
        Returns:
            dict: Dictionary containing extracted features and codes
        """
        dl_output = {
            "audio": speech,
            "x": mel,
            "num_quantizers": 24,
            "x_lens": x_lens,
        }
        if manual_threshold is not None:
            dl_output["manual_threshold"] = manual_threshold
            encoded_output = self.dualcodec_model(dl_output, encode_only=True)
        else:
            encoded_output = self.dualcodec_model(dl_output, encode_only=True, infer_using_dynamic_threshold=infer_using_dynamic_threshold)
        semantic_codes = encoded_output['semantic_codes_deaggregated']
        token_lengths = encoded_output['token_lengths']
        speech_token_len = encoded_output['speech_token_len']
        
        return {
            'semantic_codes': semantic_codes,  # [B, 1，T] - speech tokens
            'semantic_codes_aggregated': encoded_output['semantic_codes'],  # [B, T] - aggregated semantic codes
            'token_lengths': token_lengths,    # [B, T] - duration info for each speech token
            'speech_token_len': speech_token_len, # [B] - speech token length
            'semantic_repr_ret': encoded_output['semantic_repr_ret'],
            'decoder_latent': encoded_output.get('decoder_latent', None),  # [B, T, D] - decoder latent if available
            'decoder_latent_before_agg': encoded_output.get('decoder_latent_before_agg', None),  # [B, T, D] - decoder latent if available
        }
    
    def _extract_mel_features(self, speech):
        """
        Extract and normalize mel features from speech
        
        Args:
            speech (torch.Tensor): Speech audio [B, T]
            
        Returns:
            torch.Tensor: Normalized mel features [B, T, mel_dim]
        """
        # Extract mel features using the cached model
        # Resample speech from 16kHz to 24kHz
        if speech.shape[-1] > 0:  # Only resample if speech has content
            speech = torchaudio.functional.resample(speech, 16000, 24000)
        mel_feat = extract_normalized_mel_spec_50hz(speech, device=speech.device)
        # mel_feat is [B, mel_dim, T], transpose to [B, T, mel_dim]
        mel_feat = mel_feat.transpose(1, 2)
        
        return mel_feat
    
    @torch.no_grad()
    def _extract_speaker_embedding(self, speech, sample_rate=16000):
        """
        Extract speaker embedding using WeSpeaker
        
        Args:
            speech (torch.Tensor): Speech audio [B, T]
            sample_rate (int): Sample rate of the audio
            
        Returns:
            torch.Tensor: Speaker embedding [B, 256]
        """
        if not self.concat_speaker_embedding:
            return None
            
        batch_size = speech.shape[0]
        speaker_embeddings = []
        
        for i in range(batch_size):
            # Extract embedding for each sample in the batch
            audio_sample = speech[i:i+1]  # Keep batch dimension [1, T]
            embedding = self.speaker_model.extract_embedding_from_pcm(audio_sample, sample_rate)
            if embedding is None:
                # If no speech detected, use zero embedding
                embedding = torch.zeros(256, device=speech.device)
            else:
                embedding = embedding.to(speech.device)
            speaker_embeddings.append(embedding)
        
        # Stack embeddings to get [B, 256]
        speaker_embeddings = torch.stack(speaker_embeddings, dim=0)
        return speaker_embeddings.float()
    
    def training_forward(self, dl_output) -> Dict[str, Optional[torch.Tensor]]:
        """
        Training forward pass
        
        Args:
            dl_output: Dictionary containing training data
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Extract inputs
        x = dl_output.get("x", None)  # Mel features if provided
        x_lens = dl_output.get("x_lens", None)
        audio = dl_output.get("audio", None).float()
        audio_lens = dl_output.get("audio_lens", None)
        
        device = audio.device if audio is not None else x.device
        
        selected_framerate = None
        if self.flex_framerate and self.training:
            selected_framerate = random.choice(self.flex_framerate_options)

        # Extract features using DualCodec with optional manual_threshold
        if selected_framerate is not None:
            dualcodec_output = self._extract_dualcodec_features(audio, mel=x, x_lens=x_lens, manual_threshold=selected_framerate)
        else:
            # Extract DualCodec features
            dualcodec_output = self._extract_dualcodec_features(audio, mel=x, x_lens=x_lens, infer_using_dynamic_threshold=self.use_decoder_latent or self.use_decoder_latent_before_agg or self.infer_using_dynamic_threshold)
        
        # Get semantic codes for conditioning
        if self.use_decoder_latent:
            semantic_codes = dualcodec_output['semantic_codes_aggregated'].squeeze(1)  # [B, T]
        else:
            semantic_codes = dualcodec_output['semantic_codes'].squeeze(1)  # [B, T]
        speech_token_len = dualcodec_output['speech_token_len']  # [B]
        
        # if using decoder latent, should set interpolate to 1
        if self.use_decoder_latent:
            mel_feat = dualcodec_output.get('decoder_latent', None)
            mel_feat = mel_feat.transpose(1, 2)  # after: [B, T, D]
            x_mask = torch.ones(mel_feat.shape[0], mel_feat.shape[1], device=device)  # [B, T]
        elif self.use_decoder_latent_before_agg:
            mel_feat = dualcodec_output.get('decoder_latent_before_agg', None)
            if self.decoder_latent_pass_transformer:
                mel_feat = self.dualcodec_model.bottleneck_transformer(mel_feat)
            mel_feat = mel_feat.transpose(1, 2)  # [B, T, D]
            x_mask = torch.zeros(mel_feat.shape[0], mel_feat.shape[1], device=device)  # [B, T]
            for i, length in enumerate(speech_token_len):
                x_mask[i, :length] = 1.0
        else:

            # Extract mel features if not provided
            mel_feat = self._extract_mel_features(audio)
            
            # Convert audio lengths to mel frame lengths (assuming 50Hz mel)
            mel_lens = (audio_lens.float() / 24000 * 50).long()  # 24kHz to 50Hz
            
            # Create mask: 1 for valid frames, 0 for padding
            batch_size, max_len = mel_feat.shape[0], mel_feat.shape[1]
            x_mask = torch.zeros(batch_size, max_len, device=device)
            for i, length in enumerate(mel_lens):
                x_mask[i, :length] = 1.0
            
        # Ensure sequence lengths match
        # seq_len = min(semantic_codes.shape[-1], mel_feat.shape[1])
        # semantic_codes = semantic_codes[:, :seq_len]
        # mel_feat = mel_feat[:, :seq_len, :]
        # x_mask = x_mask[:, :seq_len]
        
        # Extract speaker embedding if enabled
        speaker_embedding = None
        if self.concat_speaker_embedding:
            speaker_embedding = self._extract_speaker_embedding(audio, sample_rate=16000)  # [B, 256]

        # Forward pass through VoiceBox model
        if not self.add_framerate_embedding and not self.concat_speaker_embedding:
            # Original path without any additional embeddings
            noise, x, flow_pred, final_mask, prompt_len, hidden_features = self.voicebox_model.forward_with_hidden_features(
                x=mel_feat, 
                x_mask=x_mask, 
                cond_code=semantic_codes
            )
        else:
            # Path with additional embeddings (framerate and/or speaker)
            cond_feature = self.voicebox_model.cond_emb(semantic_codes)  # [B, T, hidden_size]
            
            # Add framerate embedding if enabled
            if self.add_framerate_embedding and selected_framerate is not None:
                framerate_idx = self.flex_framerate_options.index(selected_framerate)
                framerate_emb = self.framerate_embedding(torch.tensor(framerate_idx, device=device))
                cond_feature = cond_feature + framerate_emb.unsqueeze(0).unsqueeze(0)
            
            # Concatenate speaker embedding if enabled
            if self.concat_speaker_embedding and speaker_embedding is not None:
                # Expand speaker embedding to match sequence length: [B, 256] -> [B, T, 256]
                B, T, D = cond_feature.shape
                speaker_emb_expanded = speaker_embedding.unsqueeze(1).expand(B, T, 256).to(cond_feature.device)  # [B, T, 256]
                # Concatenate along feature dimension: [B, T, hidden_size + 256]
                cond_feature = self.spk_linear(torch.cat([cond_feature, speaker_emb_expanded], dim=-1))
            
            noise, x, flow_pred, final_mask, prompt_len, hidden_features = self.voicebox_model.forward_with_hidden_features(
                x=mel_feat, 
                x_mask=x_mask, 
                cond_feature=cond_feature,
            )



        # Compute L1 flow matching loss (from voicebox_trainer.py)
        final_mask = final_mask.squeeze(-1)  # [B, T]
        flow_gt = x - (1 - self.sigma) * noise
        
        # L1 loss with mask
        diff_loss = F.l1_loss(
            flow_pred, flow_gt, reduction="none"
        ).float() * final_mask.unsqueeze(-1)
        diff_loss = torch.mean(diff_loss, dim=2).sum() / final_mask.sum()
        
        # Compute REPA loss if enabled
        if self.use_repa_loss:
            target_repr = dualcodec_output['semantic_repr_ret']
            projected_features = hidden_features
            
            # Apply self.cond_scale_factor downsampling to projected features to match SenseVoice frame rate
            B, T, D = projected_features.shape
            target_length = T // self.cond_scale_factor  # self.cond_scale_factor downsampling
            
            # Use average pooling with stride=self.cond_scale_factor to downsample
            projected_features_downsampled = projected_features.transpose(1, 2)  # [B, D, T]
            projected_features_downsampled = F.avg_pool1d(
                projected_features_downsampled,
                kernel_size=self.cond_scale_factor,
                stride=self.cond_scale_factor
            )  # [B, D, T//self.cond_scale_factor]
            assert abs(projected_features_downsampled.shape[-1] - target_repr.shape[-1]) <= 2, f"projected_features_downsampled.shape[-1]: {projected_features_downsampled.shape[-1]}, target_repr.shape[-1]: {target_repr.shape[-1]}"
            # trim to the same length
            min_len = min(projected_features_downsampled.shape[-1], target_repr.shape[-1])
            projected_features_downsampled = projected_features_downsampled[..., :min_len]
            target_repr = target_repr[..., :min_len]
            projected_features_downsampled = self.repa_projection(projected_features_downsampled.transpose(1, 2)).transpose(1, 2)
            
            repa_loss = self.compute_repa_loss(projected_features_downsampled, target_repr, mask=x_mask)
            total_loss = diff_loss + repa_loss
        else:
            repa_loss = torch.tensor(0.0)
            total_loss = diff_loss

        self.step += 1
        
        # Prepare metrics
        metrics = {
            "loss": total_loss.cpu().detach(),
            "diff_loss": diff_loss.cpu().detach(),
            "repa_loss": repa_loss.cpu().detach(),
            "prompt_len_mean": prompt_len.float().mean().cpu().detach() if len(prompt_len) > 0 else torch.tensor(0.0),
            "step": self.step,
        }
        
        return total_loss, metrics
    
    @torch.inference_mode()
    def inference(
        self,
        speech: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        n_timesteps: int = 10,
        cfg: float = 1.0,
        rescale_cfg: float = 0.75,
    ) -> torch.Tensor:
        """
        Inference with VoiceBox model
        
        Args:
            speech: Input speech audio [B, T]
            prompt: Optional prompt mel features [B, T_prompt, mel_dim]
            x_mask: Optional target mask [B, T_target]
            prompt_mask: Optional prompt mask [B, T_prompt]
            n_timesteps: Number of diffusion timesteps
            cfg: Classifier-free guidance scale
            rescale_cfg: Rescaling factor for CFG
            
        Returns:
            torch.Tensor: Generated mel features [B, T_target, mel_dim]
        """
        device = speech.device
        
        # Extract DualCodec features for conditioning
        dualcodec_output = self._extract_dualcodec_features(speech)
        cond_codes = dualcodec_output['semantic_codes'].squeeze(1)  # [B, T]
        
        # Handle additional conditioning features
        if self.concat_speaker_embedding:
            # Extract speaker embedding
            speaker_embedding = self._extract_speaker_embedding(speech, sample_rate=16000)  # [B, 256]
            
            # Create conditioning feature with speaker embedding
            cond_feature = self.voicebox_model.cond_emb(cond_codes)  # [B, T, hidden_size]
            
            if speaker_embedding is not None:
                B, T, D = cond_feature.shape
                speaker_emb_expanded = speaker_embedding.unsqueeze(1).expand(B, T, 256)  # [B, T, 256]
                cond_feature = torch.cat([cond_feature, speaker_emb_expanded], dim=-1)
            
            # Use VoiceBox reverse diffusion with cond_feature
            generated_mel = self.voicebox_model.reverse_diffusion(
                cond_feature=cond_feature,
                prompt=prompt,
                x_mask=x_mask,
                prompt_mask=prompt_mask,
                n_timesteps=n_timesteps,
                cfg=cfg,
                rescale_cfg=rescale_cfg,
            )
        else:
            # Use VoiceBox reverse diffusion with cond_codes
            generated_mel = self.voicebox_model.reverse_diffusion(
                cond=cond_codes,
                prompt=prompt,
                x_mask=x_mask,
                prompt_mask=prompt_mask,
                n_timesteps=n_timesteps,
                cfg=cfg,
                rescale_cfg=rescale_cfg,
            )
        
        return generated_mel


def create_voicebox_wrapper_from_config(
    dualcodec_config_path: str,
    dualcodec_ckpt: str,
    voicebox_config: Optional[Dict] = None,
    **kwargs
) -> VoiceboxWrapper:
    """
    Factory function to create VoiceboxWrapper from configuration
    
    Args:
        dualcodec_config_path: Path to DualCodec config file
        dualcodec_ckpt: Path to DualCodec checkpoint
        voicebox_config: Optional VoiceBox model configuration
        **kwargs: Additional arguments for VoiceboxWrapper
        
    Returns:
        VoiceboxWrapper: Initialized wrapper model
    """
    return VoiceboxWrapper(
        dualcodec_config_path=dualcodec_config_path,
        dualcodec_ckpt=dualcodec_ckpt,
        voicebox_config=voicebox_config,
        **kwargs
    ) 