from typing import Dict, Optional, Union, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import torchaudio
import os
import random

from flexicodec.ar_tts.utils.common import IGNORE_ID
from flexicodec.ar_tts.utils.label_smoothing_loss import LabelSmoothingLoss
from flexicodec.ar_tts.utils.common import th_accuracy
params = lambda model: sum(p.numel() for p in model.parameters())

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=IGNORE_ID)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            # Only average over non-ignored elements
            mask = (targets != IGNORE_ID)
            return focal_loss[mask].mean() if mask.any() else torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance
    """
    def __init__(self, class_weights=None, reduction='mean'):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.class_weights is not None:
            # Apply class weights
            weighted_inputs = inputs * self.class_weights.unsqueeze(0)
            return F.cross_entropy(weighted_inputs, targets, reduction=self.reduction, ignore_index=IGNORE_ID)
        else:
            return F.cross_entropy(inputs, targets, reduction=self.reduction, ignore_index=IGNORE_ID)


class TransformerLM(torch.nn.Module):
    """
    TransformerLM Module
    """

    def __init__(
        self,
        text_encoder_input_size: int,
        llm_input_size: int,
        llm_output_size: int,
        text_token_size: int,
        speech_token_size: int,
        text_encoder: torch.nn.Module,
        llm: torch.nn.Module,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        spk_embed_dim: int = 192,
        duration_classes: int = 10,
        duration_loss_type: str = "focal",  # "focal", "weighted", "ce", "label_smoothing"
        duration_class_weights: Optional[torch.Tensor] = None,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        duration_lsm_weight: float = 0.1,
        use_duration_conditioning: bool = True,
        use_dialog_span: bool = False,
        flex_framerate: bool = False,
        flex_framerate_options: list = [0.87,0.91,1.0]
    ):
        """
        :param text_encoder_input_size:
        :param llm_input_size:
        :param llm_output_size:
        :param text_token_size:
        :param speech_token_size:
        :param text_encoder:
        :param llm:
        :param length_normalized_loss:
        :param lsm_weight:
        :param spk_embed_dim:
        :param duration_classes: Number of duration classes for classification
        :param duration_loss_type: Type of loss for duration prediction ("focal", "weighted", "ce", "label_smoothing")
        :param duration_class_weights: Class weights for weighted loss (duration_classes,)
        :param focal_alpha: Alpha parameter for focal loss
        :param focal_gamma: Gamma parameter for focal loss
        :param duration_lsm_weight: Label smoothing weight for duration prediction (0.0 to disable)
        :param use_duration_conditioning: Whether to use duration tokens as conditioning (default: False)
        :param use_dialog_span: Whether to use dialog span for speaker change signaling
        """
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        self.duration_classes = duration_classes
        self.duration_loss_type = duration_loss_type
        self.use_duration_conditioning = use_duration_conditioning
        self.use_dialog_span = use_dialog_span
        self.flex_framerate = flex_framerate
        self.flex_framerate_options = flex_framerate_options
        
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(
            text_token_size, text_encoder_input_size
        )
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(), llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)
        
        # 3.1. [Optional] build frame rate embedding for flex_framerate
        if self.flex_framerate:
            self.framerate_embed_affine_layer = torch.nn.Linear(len(flex_framerate_options), llm_input_size)

        self.step = 0

        # 4. Duration prediction head with improved loss handling
        self.duration_decoder = nn.Linear(llm_output_size, duration_classes)
        
        # 5. Duration conditioning embeddings (if enabled)
        if self.use_duration_conditioning:
            self.duration_embedding = torch.nn.Embedding(duration_classes + 1, llm_input_size)
        
        # Initialize duration loss based on type
        if duration_loss_type == "focal":
            self.duration_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif duration_loss_type == "weighted":
            if duration_class_weights is not None:
                self.register_buffer('duration_class_weights', duration_class_weights)
            else:
                # Default inverse frequency weights (will be updated during training)
                self.register_buffer('duration_class_weights', torch.ones(duration_classes))
            self.duration_criterion = WeightedCrossEntropyLoss(
                class_weights=self.duration_class_weights if duration_class_weights is not None else None
            )
        elif duration_loss_type == "label_smoothing":
            self.duration_criterion = LabelSmoothingLoss(
                size=duration_classes,
                padding_idx=IGNORE_ID,
                smoothing=duration_lsm_weight,
                normalize_length=False,  # Usually not needed for duration prediction
            )
        else:  # "ce"
            self.duration_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
    def encode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        """
        :param text:
        :param text_lengths:
        :return:
        """
        encoder_out, encoder_mask = self.text_encoder(
            text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def prepare_duration_targets(
        self,
        speech_token_len: torch.Tensor,
        duration: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Prepare duration targets for each speech token position
        :param speech_token_len: (B,) - lengths of speech tokens
        :param duration: (B, T) - duration classes for each speech token
        :param device: device to place tensors on
        :return: duration_targets (B, max_len) - padded duration targets
        """
        B = speech_token_len.shape[0]
        max_len = speech_token_len.max().item()
        
        # Create duration targets with padding
        duration_targets = torch.full(
            (B, max_len), IGNORE_ID, dtype=torch.long, device=device
        )
        
        for i in range(B):
            seq_len = speech_token_len[i].item()
            duration_targets[i, :seq_len] = duration[i, :seq_len]
        
        return duration_targets

    def pad_unpad_sequence(
        self,
        sos_eos_emb,
        embedding,
        text_token,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_len,
    ):
        """
        :param sos_eos_emb:
        :param embedding:
        :param text_token:
        :param text_token_len:
        :param task_id_emb:
        :param speech_token:
        :param speech_token_len:
        :return:
        """
        B = text_token.shape[0]
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True
        )
        if embedding is None:
            embedding = torch.zeros(
                B, 1, self.llm_input_size, device=sos_eos_emb.device
            )
        lm_input = [
            torch.concat(
                [
                    sos_eos_emb.squeeze(dim=0),
                    embedding[i],
                    text_token[i],
                    task_id_emb.squeeze(dim=0),
                    speech_token[i],
                ],
                dim=0,
            )
            for i in range(len(text_token))
        ]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text_token: (B, L)
            text_token_lengths: (B,)
            speech_token: (B, T)
            speech_token_lengths: (B,)
            duration: (B, T) - duration classes for each speech token
            embedding: (B,)
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)
        
        # Handle duration targets
        if "duration" in batch and batch["duration"] is not None:
            duration = batch["duration"].to(device)
            duration_targets = self.prepare_duration_targets(
                speech_token_len, duration, device
            )
        else:
            duration_targets = None

        if batch["embedding"] is not None:
            embedding = batch["embedding"].to(device)
        else:
            embedding = None

        # Get selected framerate if provided (for flex_framerate)
        selected_framerate = batch.get("selected_framerate", None)

        # 1. prepare llm_target
        lm_target = [
            torch.tensor(
                [IGNORE_ID] * (2 + text_token_len[i])
                + speech_token[i, : speech_token_len[i]].tolist()
                + [self.speech_token_size]
            )
            for i in range(text_token.size(0))
        ]
        lm_target = pad_sequence(
            lm_target, batch_first=True, padding_value=IGNORE_ID
        ).to(device)

        # 2. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 3. embedding projection
        if self.flex_framerate and selected_framerate is not None:
            # Use the selected frame rate for embedding (passed from training_forward)
            # Create frame rate embedding (one-hot encoded)
            framerate_embedding = torch.zeros(len(self.flex_framerate_options), device=device)
            framerate_idx = self.flex_framerate_options.index(selected_framerate)
            framerate_embedding[framerate_idx] = 1.0
            
            # Expand to batch size and apply affine transformation
            batch_size = text_token.size(0)
            framerate_embedding = framerate_embedding.unsqueeze(0).expand(batch_size, -1)
            embedding = self.framerate_embed_affine_layer(framerate_embedding)
            embedding = embedding.unsqueeze(1) # [b, 1, h]
        elif embedding is not None:
            assert False
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(1)

        # 4. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 5. encode speech_token
        speech_token = self.speech_embedding(speech_token)
        
        # 6. Add duration conditioning if enabled
        if self.use_duration_conditioning and duration_targets is not None:
            # Create shifted duration tokens (right-shift)
            # For speech tokens [s1, s2, s3] with durations [d1, d2, d3]
            # Duration conditioning should be [0, d1, d2] (0 is special start token)
            duration_conditioning = torch.zeros_like(speech_token_len.unsqueeze(1).expand(-1, speech_token.size(1)), dtype=torch.long, device=device)
            
            for i in range(speech_token.size(0)):
                seq_len = speech_token_len[i].item()
                if seq_len > 1:
                    # Shift right: first position gets 0 (start token), rest get previous durations
                    duration_conditioning[i, 1:seq_len] = duration[i, :seq_len-1]
                # First position remains 0 (start token)
            
            # Convert to embeddings and add to speech tokens
            assert (duration_conditioning <= self.duration_classes).all()
            duration_emb = self.duration_embedding(duration_conditioning)
            speech_token = speech_token + duration_emb

        # 7. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_eos_emb,
            embedding,
            text_token,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_len,
        )

        # 8. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(
            logits.view(-1, self.speech_token_size + 1),
            lm_target,
            ignore_label=IGNORE_ID,
        )
        
        # 9. Duration prediction
        duration_loss = torch.tensor(0.0, device=device)
        total_loss = loss
        if duration_targets is not None:
            # Extract speech token positions from lm_output for duration prediction
            duration_logits = []
            duration_targets_flat = []
            
            for i in range(text_token.size(0)):
                text_len = text_token_len[i].item()
                speech_len = speech_token_len[i].item()
                
                # Speech tokens start after: sos_eos + embedding + text_tokens + task_id
                speech_start_pos = 2 + text_len + 1
                speech_positions = range(speech_start_pos, speech_start_pos + speech_len)
                
                # Extract LM output for speech token positions
                duration_logits.append(lm_output[i, speech_positions])
                duration_targets_flat.append(duration_targets[i, :speech_len])
            
            if duration_logits:
                duration_logits = torch.cat(duration_logits, dim=0)
                duration_targets_flat = torch.cat(duration_targets_flat, dim=0)
                
                duration_logits = self.duration_decoder(duration_logits)
                duration_loss = self.duration_criterion(duration_logits, duration_targets_flat)
                
                # Add duration loss to main loss
                total_loss = loss + duration_loss
        
        self.step += 1
        metrics = {
            "loss": loss.cpu().detach(),
            "acc": acc.cpu().detach(),
            "duration_loss": duration_loss.cpu().detach(),
            "step": self.step,
        }
        return total_loss, metrics

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        sampling: Union[bool, int, float] = True,
        beam_size: int = 1,
        ignore_eos: bool = True,
    ):
        """
        :param weighted_scores:
        :param sampling:
        :param beam_size:
        :param ignore_eos:
        :return:
        """
        while True:
            prob, indices = weighted_scores.softmax(dim=-1).topk(sampling)
            top_ids = prob.multinomial(beam_size, replacement=True)
            top_ids = indices[top_ids]
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        prompt_token_lengths: Optional[torch.Tensor] = None,
        beam_size: int = 1,
        top_k: int = 25,
        temperature: float = 1.0,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        predict_duration: bool = True,
        duration_temperature: float = 1.0,
        duration_top_k: int = 5,
        duration_use_class_weights: bool = True,
        inference_framerate: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        :param text:
        :param text_len:
        :param prompt_text:
        :param prompt_text_len:
        :param prompt_speech_token:
        :param prompt_speech_token_len:
        :param embedding:
        :param prompt_token_lengths: Duration classes for prompt speech tokens
        :param beam_size:
        :param top_k:
        :param temperature:
        :param max_token_text_ratio:
        :param min_token_text_ratio:
        :param predict_duration: Whether to predict duration for generated tokens
        :param duration_temperature: Temperature for duration sampling
        :param duration_top_k: Top-k sampling for duration
        :param duration_use_class_weights: Whether to use class weights for duration sampling
        :param inference_framerate: Frame rate to use for inference (overrides speaker embedding if provided)
        :return: Dictionary containing 'speech_tokens' and optionally 'duration_classes'
        """
        print(inference_framerate, duration_temperature, self.flex_framerate_options)
        if inference_framerate is not None:
            assert float(inference_framerate) in self.flex_framerate_options
        device = text.device
        if prompt_text is not None:
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len
        if prompt_text_len is None:
            prompt_text_len = 0
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if self.flex_framerate and inference_framerate is not None:
            # Use frame rate embedding instead of speaker embedding
            inference_framerate = float(inference_framerate)
            if inference_framerate in self.flex_framerate_options:
                framerate_embedding = torch.zeros(len(self.flex_framerate_options), device=device)
                framerate_idx = self.flex_framerate_options.index(inference_framerate)
                framerate_embedding[framerate_idx] = 1.0
                
                # Apply affine transformation and reshape for inference
                embedding = self.framerate_embed_affine_layer(framerate_embedding.unsqueeze(0))
                embedding = embedding.unsqueeze(dim=1)
            else:
                raise ValueError(f"inference_framerate {inference_framerate} not in flex_framerate_options {self.flex_framerate_options}")
        elif embedding is not None and embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
            if self.use_duration_conditioning and predict_duration:
                assert prompt_token_lengths is not None, "prompt_token_lengths is required when use_duration_conditioning is True"
                prompt_len = prompt_speech_token.shape[1]
                # Create shifted duration tokens for prompt
                # first position gets 0 (start token), rest get previous durations
                prompt_duration_conditioning = torch.zeros(1, prompt_len, dtype=torch.long, device=device)
                if prompt_len > 1:
                    prompt_duration_conditioning[0, 1:] = prompt_token_lengths[0, :prompt_len - 1]
                
                duration_emb = self.duration_embedding(prompt_duration_conditioning)
                prompt_speech_token_emb = prompt_speech_token_emb + duration_emb
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size).to(device)
        lm_input = torch.concat(
            [sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        duration_classes = []
        previous_duration = 0  # Start with special start token for duration conditioning
        if self.use_duration_conditioning and prompt_token_lengths is not None and prompt_speech_token.shape[1] > 0:
            last_prompt_duration_class = prompt_token_lengths[0, -1].item()
            previous_duration = last_prompt_duration_class

        offset = 0
        att_cache, cnn_cache = torch.zeros(
            (0, 0, 0, 0), device=lm_input.device
        ), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(
                lm_input,
                offset=0,
                required_cache_size=-1,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=torch.tril(
                    torch.ones(
                        (1, lm_input.shape[1], lm_input.shape[1]),
                        device=lm_input.device,
                    )
                ).to(torch.bool),
            )
            
            # Predict duration for the previous token (skip first iteration)
            # At this point, the context includes the previous token
            if predict_duration and i > 0:
                duration_logits = self.duration_decoder(y_pred[:, -1])  # [1, num_classes]

                # tune down the logits for length 1
                # duration_logits[:,1] = duration_logits[:,1] / 2
                
                # Apply temperature scaling
                if duration_temperature != 1.0:
                    duration_logits = duration_logits / duration_temperature
                
                # Apply class weights for sampling if requested
                if duration_use_class_weights and hasattr(self, 'duration_class_weights'):
                    duration_logits = duration_logits + torch.log(self.duration_class_weights.unsqueeze(0))
                
                # Sample duration class
                if duration_top_k > 0:
                    # Top-k sampling
                    top_logits, top_indices = torch.topk(duration_logits, min(duration_top_k, duration_logits.size(-1)), dim=-1)
                    probs = F.softmax(top_logits, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1).squeeze(-1)
                    duration_class = top_indices.gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)
                else:
                    # Standard sampling
                    probs = F.softmax(duration_logits, dim=-1)
                    duration_class = torch.multinomial(probs, 1).squeeze(-1)
                
                duration_classes.append(duration_class.item())
                # Update previous duration for next iteration
                previous_duration = duration_class.item()
            
            # Predict speech token
            logits = self.llm_decoder(y_pred[:, -1])
            if temperature > 0:
                logits = logits / temperature
            
            top_ids = self.sampling_ids(
                logits.squeeze(dim=0),
                top_k,
                beam_size,
                ignore_eos=True if i < min_len else False,
            ).item()
            if top_ids == self.speech_token_size:
                break
            out_tokens.append(top_ids)
            
            offset += lm_input.size(1)
            
            # Get speech token embedding
            speech_token_emb = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)
            
            # Add duration conditioning if enabled
            if self.use_duration_conditioning:
                duration_emb = self.duration_embedding.weight[previous_duration].reshape(1, 1, -1)
                speech_token_emb = speech_token_emb + duration_emb
            
            lm_input = speech_token_emb

        speech_tokens = torch.tensor([out_tokens], dtype=torch.int64, device=device)
        result = {"speech_tokens": speech_tokens}
        
        # 6. Add duration predictions if they were generated
        if predict_duration and duration_classes:
            result["duration_classes"] = torch.tensor([duration_classes], dtype=torch.int64, device=device)
        print(result['duration_classes'])
        return result
import yaml


class TransformerLMWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Preparing FlexiCodec model for feature extraction...")
        # Note: flexicodec_model is not moved to any device per trainer implementation
        # The following line is commented out as `prepare_model` is not available in this context.
        from flexicodec.infer import prepare_model
        self.flexicodec_model = prepare_model()['model'] 
        
        # Freeze FlexiCodec model parameters
        for param in self.flexicodec_model.parameters():
            param.requires_grad = False
        
        # Store flex_framerate parameters for coordination
        self.flex_framerate = kwargs.get('flex_framerate', False)
        self.flex_framerate_options = kwargs.get('flex_framerate_options', [0.87, 0.91, 1.0])
        
        self.transformer_lm = create_transformer_lm_from_config(**kwargs)
        self.trainer_callbacks = []
    
    @property
    def dualcodec_model(self):
        """Alias for flexicodec_model for backward compatibility"""
        return self.flexicodec_model
    
    @torch.no_grad()
    @torch.autocast('cuda', enabled=False)
    def _extract_flexicodec_features(self, speech, mel, x_lens=None, sample_rate=16000, manual_threshold=None):
        """
        Extracts features using FlexiCodec model with batch inference.
        
        Args:
            speech (torch.Tensor): Speech audio [B, T]
            mel (torch.Tensor, optional): Mel spectrogram features [B, T, D]
            x_lens (torch.Tensor, optional): Lengths of the mel features
            sample_rate (int): Sample rate of the audio
            manual_threshold (float, optional): Manual threshold for frame rate control
            
        Returns:
            dict: Dictionary containing extracted features and codes
        """
        assert mel is not None, "Mel spectrogram is required"
        dl_output = {
            "audio": speech,
            "x": mel,
            "num_quantizers": 1,
            "x_lens": x_lens,
        }
        
        # Add manual_threshold to dl_output if provided
        if manual_threshold is not None:
            dl_output["manual_threshold"] = manual_threshold
            
        encoded_output = self.flexicodec_model(dl_output, encode_only=True)
        semantic_codes = encoded_output['semantic_codes']
        token_lengths = encoded_output['token_lengths']
        speech_token_len = encoded_output['speech_token_len']
        return {
            'semantic_codes': semantic_codes,  # [B, T] - speech tokens
            'token_lengths': token_lengths,    # [B, T] - duration info for each speech token
            'speech_token_len': speech_token_len, # [B] - speech token length
        }
    
    def _extract_dualcodec_features(self, speech, mel, x_lens=None, sample_rate=16000, manual_threshold=None):
        """Alias for _extract_flexicodec_features for backward compatibility"""
        return self._extract_flexicodec_features(speech, mel, x_lens, sample_rate, manual_threshold)
    
    def training_forward(self, dl_output) -> Dict[str, Optional[torch.Tensor]]:
        x = dl_output.get("x", None)
        x_lens = dl_output.get("x_lens", None)
        text_ids = dl_output.get("text_ids", None)
        text_ids_lens = dl_output.get("text_ids_lens", None)
        audio = dl_output.get("audio", None)
        audio_lens = dl_output.get("audio_lens", None)
        # Handle flex_framerate: randomly select frame rate during training
        selected_framerate = None
        if self.flex_framerate and self.training:
            selected_framerate = random.choice(self.flex_framerate_options)

        # Extract features using FlexiCodec with optional manual_threshold
        if selected_framerate is not None:
            flexicodec_output = self._extract_flexicodec_features(audio, mel=x, x_lens=x_lens, manual_threshold=selected_framerate)
        else:
            flexicodec_output = self._extract_flexicodec_features(audio, mel=x, x_lens=x_lens)
        
        # Get semantic codes (speech tokens) and duration info
        speech_tokens = flexicodec_output['semantic_codes'].squeeze(1)  # [B, T]
        token_lengths = flexicodec_output['token_lengths']   # [B, T] - duration for each speech token
        speech_token_len = flexicodec_output['speech_token_len'] # [B] - speech token length
        device = speech_tokens.device
        
        # Prepare batch for LLM model
        model_batch = {
            "text_token": text_ids,              # [B, L]
            "text_token_len": text_ids_lens,          # [B]
            "speech_token": speech_tokens,       # [B, T]
            "speech_token_len": speech_token_len, # [B]
            "duration": token_lengths,                # [B, T] - duration classes for each speech token
            "embedding": None,  # [B, embed_dim] if available
            "selected_framerate": selected_framerate  # Pass selected frame rate to transformer
        }
        
        return self.transformer_lm(model_batch, device)
    
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        prompt_token_lengths: Optional[torch.Tensor] = None,
        beam_size: int = 1,
        top_k: int = 25,
        temperature: float = 1.0,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
        predict_duration: bool = True,
        duration_temperature: float = 0.9,
        duration_top_k: int = 5,
        duration_use_class_weights: bool = True,
        inference_framerate: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Wrapper for transformer inference with flex_framerate support.
        
        :param inference_framerate: Frame rate to use for inference (overrides speaker embedding if provided)
        :return: Dictionary containing 'speech_tokens' and optionally 'duration_classes'
        """
        return self.transformer_lm.inference(
            text=text,
            text_len=text_len,
            prompt_text=prompt_text,
            prompt_text_len=prompt_text_len,
            prompt_speech_token=prompt_speech_token,
            prompt_speech_token_len=prompt_speech_token_len,
            embedding=embedding,
            prompt_token_lengths=prompt_token_lengths,
            beam_size=beam_size,
            top_k=top_k,
            temperature=temperature,
            max_token_text_ratio=max_token_text_ratio,
            min_token_text_ratio=min_token_text_ratio,
            predict_duration=predict_duration,
            duration_temperature=duration_temperature,
            duration_top_k=duration_top_k,
            duration_use_class_weights=duration_use_class_weights,
            inference_framerate=inference_framerate,
        )


def create_transformer_lm_from_config(
    text_encoder_input_size: int = 1024,
    llm_input_size: int = 1536,
    llm_output_size: int = 1536,
    text_token_size: int = 51866,
    speech_token_size: int = 32768,
    spk_embed_dim: int = 192,
    duration_classes: int = 10,
    duration_loss_type: str = "focal",
    duration_class_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    duration_lsm_weight: float = 0.0,
    use_duration_conditioning: bool = True,
    use_dialog_span: bool = False,
    flex_framerate: bool = False,
    flex_framerate_options: list = [0.87, 0.91, 1.0],
    # Text encoder configurable parameters
    text_encoder_output_size: int = 1024,
    text_encoder_attention_heads: int = 8,
    text_encoder_linear_units: int = 3584,
    text_encoder_num_blocks: int = 4,
    text_encoder_dropout_rate: float = 0.1,
    text_encoder_positional_dropout_rate: float = 0.1,
    text_encoder_attention_dropout_rate: float = 0.0,
    text_encoder_normalize_before: bool = True,
    text_encoder_input_layer: str = 'identity',
    text_encoder_pos_enc_layer_type: str = 'rel_pos_espnet',
    text_encoder_selfattention_layer_type: str = 'rel_selfattn',
    text_encoder_use_cnn_module: bool = False,
    text_encoder_macaron_style: bool = False,
    text_encoder_use_dynamic_chunk: bool = False,
    text_encoder_use_dynamic_left_chunk: bool = False,
    text_encoder_static_chunk_size: int = 1,
    # Language model configurable parameters
    llm_attention_heads: int = 12,
    llm_linear_units: int = 5376,
    llm_num_blocks: int = 12,
    llm_dropout_rate: float = 0.1,
    llm_positional_dropout_rate: float = 0.1,
    llm_attention_dropout_rate: float = 0.0,
    llm_input_layer: str = 'identity',
    llm_pos_enc_layer_type: str = 'rel_pos_espnet',
    llm_selfattention_layer_type: str = 'rel_selfattn',
    llm_static_chunk_size: int = 1,
    # General model parameters
    length_normalized_loss: bool = True,
    lsm_weight: float = 0.0,
    **kwargs
) -> TransformerLM:
    """
    Factory function to create a TransformerLM instance with default configuration
    
    Args:
        text_encoder_input_size: Input size for text encoder
        llm_input_size: Input size for language model
        llm_output_size: Output size for language model
        text_token_size: Size of text vocabulary
        speech_token_size: Size of speech token vocabulary
        spk_embed_dim: Speaker embedding dimension
        duration_classes: Number of duration classes for classification
        duration_loss_type: Type of loss for duration prediction ("focal", "weighted", "ce", "label_smoothing")
        duration_class_weights: Class weights for weighted loss (duration_classes,)
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        duration_lsm_weight: Label smoothing weight for duration prediction (0.0 to disable)
        use_duration_conditioning: Whether to use duration tokens as conditioning (default: False)
        use_dialog_span: Whether to use dialog span for speaker change signaling
        
        # Text encoder configurable parameters
        text_encoder_output_size: Output size for text encoder
        text_encoder_attention_heads: Number of attention heads in text encoder
        text_encoder_linear_units: Number of linear units in text encoder
        text_encoder_num_blocks: Number of transformer blocks in text encoder
        text_encoder_dropout_rate: Dropout rate for text encoder
        text_encoder_positional_dropout_rate: Positional dropout rate for text encoder
        text_encoder_attention_dropout_rate: Attention dropout rate for text encoder
        text_encoder_normalize_before: Whether to normalize before attention/ffn
        text_encoder_input_layer: Type of input layer for text encoder
        text_encoder_pos_enc_layer_type: Type of positional encoding for text encoder
        text_encoder_selfattention_layer_type: Type of self-attention for text encoder
        text_encoder_use_cnn_module: Whether to use CNN module in text encoder
        text_encoder_macaron_style: Whether to use macaron style in text encoder
        text_encoder_use_dynamic_chunk: Whether to use dynamic chunking in text encoder
        text_encoder_use_dynamic_left_chunk: Whether to use dynamic left chunking in text encoder
        text_encoder_static_chunk_size: Static chunk size for text encoder
        
        # Language model configurable parameters
        llm_attention_heads: Number of attention heads in language model
        llm_linear_units: Number of linear units in language model
        llm_num_blocks: Number of transformer blocks in language model
        llm_dropout_rate: Dropout rate for language model
        llm_positional_dropout_rate: Positional dropout rate for language model
        llm_attention_dropout_rate: Attention dropout rate for language model
        llm_input_layer: Type of input layer for language model
        llm_pos_enc_layer_type: Type of positional encoding for language model
        llm_selfattention_layer_type: Type of self-attention for language model
        llm_static_chunk_size: Static chunk size for language model
        
        # General model parameters
        length_normalized_loss: Whether to normalize loss by sequence length
        lsm_weight: Label smoothing weight
        
        **kwargs: Additional arguments to pass to text_encoder and llm constructors
        
    Returns:
        TransformerLM: Initialized transformer language model
    """
    from flexicodec.ar_tts.utils.transformer.encoder import ConformerEncoder, TransformerEncoder
    
    # Create text encoder
    text_encoder = ConformerEncoder(
        input_size=text_encoder_input_size,
        output_size=text_encoder_output_size,
        attention_heads=text_encoder_attention_heads,
        linear_units=text_encoder_linear_units,
        num_blocks=text_encoder_num_blocks,
        dropout_rate=text_encoder_dropout_rate,
        positional_dropout_rate=text_encoder_positional_dropout_rate,
        attention_dropout_rate=text_encoder_attention_dropout_rate,
        normalize_before=text_encoder_normalize_before,
        input_layer=text_encoder_input_layer,
        pos_enc_layer_type=text_encoder_pos_enc_layer_type,
        selfattention_layer_type=text_encoder_selfattention_layer_type,
        use_cnn_module=text_encoder_use_cnn_module,
        macaron_style=text_encoder_macaron_style,
        use_dynamic_chunk=text_encoder_use_dynamic_chunk,
        use_dynamic_left_chunk=text_encoder_use_dynamic_left_chunk,
        static_chunk_size=text_encoder_static_chunk_size,
        **kwargs
    )
    
    # Create language model
    llm = TransformerEncoder(
        input_size=llm_input_size,
        output_size=llm_output_size,
        attention_heads=llm_attention_heads,
        linear_units=llm_linear_units,
        num_blocks=llm_num_blocks,
        dropout_rate=llm_dropout_rate,
        positional_dropout_rate=llm_positional_dropout_rate,
        attention_dropout_rate=llm_attention_dropout_rate,
        input_layer=llm_input_layer,
        pos_enc_layer_type=llm_pos_enc_layer_type,
        selfattention_layer_type=llm_selfattention_layer_type,
        static_chunk_size=llm_static_chunk_size,
        **kwargs
    )
    
    # Create TransformerLM instance
    model = TransformerLM(
        text_encoder_input_size=text_encoder_input_size,
        llm_input_size=llm_input_size,
        llm_output_size=llm_output_size,
        text_token_size=text_token_size,
        speech_token_size=speech_token_size,
        text_encoder=text_encoder,
        llm=llm,
        length_normalized_loss=length_normalized_loss,
        lsm_weight=lsm_weight,
        spk_embed_dim=spk_embed_dim,
        duration_classes=duration_classes,
        duration_loss_type=duration_loss_type,
        duration_class_weights=duration_class_weights,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        duration_lsm_weight=duration_lsm_weight,
        use_duration_conditioning=use_duration_conditioning,
        use_dialog_span=use_dialog_span,
        flex_framerate=flex_framerate,
        flex_framerate_options=flex_framerate_options,
    )
    
    return model


def prepare_artts_model(
    checkpoint_path: str,
    dualcodec_config_path: Optional[str] = None,
    dualcodec_ckpt: Optional[str] = None,
    device: Optional[str] = None,
    **model_kwargs
) -> Dict:
    """
    Prepare and load the AR TTS (TransformerLMWrapper) model for inference.
    
    Args:
        checkpoint_path: Path to the AR TTS model checkpoint (.pt or .safetensors)
        dualcodec_config_path: Path to FlexiCodec config YAML file (optional, for loading FlexiCodec model)
        dualcodec_ckpt: Path to FlexiCodec checkpoint (optional, for loading FlexiCodec model)
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detection)
        **model_kwargs: Additional model configuration parameters
        
    Returns:
        dict: Dictionary containing 'model' and 'device' keys
    """
    # Default model configuration
    default_config = {
        'text_encoder_input_size': 1024,
        'text_encoder_output_size': 1024,
        'text_encoder_num_blocks': 4,
        'llm_input_size': 1024,
        'llm_output_size': 1024,
        'llm_attention_heads': 8,
        'llm_num_blocks': 16,
        'llm_dropout_rate': 0.1,
        'llm_positional_dropout_rate': 0.1,
        'llm_attention_dropout_rate': 0.0,
        'text_token_size': 51866,
        'speech_token_size': 32768,
        'spk_embed_dim': 192,
        'duration_classes': 10,
        'use_duration_conditioning': True,
        'flex_framerate': True,
        'flex_framerate_options': [0.86, 0.91],
    }
    
    # Merge default config with provided kwargs
    config = {**default_config, **model_kwargs}
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Creating TransformerLMWrapper model...")
    
    # Note: TransformerLMWrapper.__init__ will load FlexiCodec model internally
    # If you need to customize FlexiCodec loading, you can pass the paths via config
    # but the current implementation loads it automatically in __init__
    
    # Create model (FlexiCodec will be loaded inside TransformerLMWrapper.__init__)
    model = TransformerLMWrapper(**config)
    
    # Load AR TTS checkpoint
    print(f"Loading AR TTS checkpoint from: {checkpoint_path}")
    if checkpoint_path.endswith('.safetensors'):
        import safetensors.torch
        state_dict = safetensors.torch.load_file(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # if missing_keys:
    #     print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
    # if unexpected_keys:
    #     print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
    
    model.eval()
    model = model.to(device)
    
    return {
        'model': model,
        'device': device
    }


def infer_artts(
    ar_model_dict: Dict,
    text: str,
    language: str = "en",
    ref_audio_path: Optional[str] = None,
    ref_text: str = "",
    merging_threshold: Optional[float] = None,
    beam_size: int = 1,
    top_k: int = 25,
    temperature: float = 1.0,
    max_token_text_ratio: float = 20.0,
    min_token_text_ratio: float = 0.0,
    predict_duration: bool = True,
    duration_top_k: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Perform AR TTS inference to generate speech tokens.
    
    Args:
        ar_model_dict: Dictionary returned from prepare_artts_model()
        text: Target text to synthesize
        language: Language code ('en', 'zh', etc.)
        ref_audio_path: Path to reference audio file
        ref_text: Reference text (optional)
        merging_threshold: Merging threshold/frame rate for inference (must be in flex_framerate_options)
            Same as inference_framerate - controls FlexiCodec frame rate merging
        beam_size: Beam size for AR decoding
        top_k: Top-k sampling for AR decoding
        temperature: Temperature for AR decoding
        max_token_text_ratio: Maximum token-to-text ratio
        min_token_text_ratio: Minimum token-to-text ratio
        predict_duration: Whether to predict duration classes
        duration_top_k: Top-k for duration sampling
    
    Returns:
        tuple: (speech_tokens, duration_classes, prompt_speech_token)
            - speech_tokens: Generated speech tokens [1, T]
            - duration_classes: Duration classes for each token [1, T] or None
            - prompt_speech_token: Prompt speech tokens [1, T]
    """
    from flexicodec.ar_tts.utils.whisper_tokenize import text2idx
    from flexicodec.feature_extractors import FBankGen
    
    ar_model = ar_model_dict['model']
    device = ar_model_dict['device']
    
    # Prepare text tokens
    prompt_text = ref_text if ref_text else ""
    combined_text = prompt_text + ", " + text if prompt_text else text
    tokens = text2idx(combined_text, language=language, g2p_prob=1.0)
    text_tokens = torch.tensor([tokens], dtype=torch.long)
    text_lengths = torch.tensor([len(tokens)])
    
    text_tokens = text_tokens.to(device)
    text_lengths = text_lengths.to(device)
    
    # Extract reference features
    if ref_audio_path is None:
        raise ValueError("ref_audio_path is required")
    
    ref_audio, sr = torchaudio.load(ref_audio_path)
    if sr != 16000:
        ref_audio = torchaudio.transforms.Resample(sr, 16000)(ref_audio)
    if ref_audio.shape[0] > 1:
        ref_audio = ref_audio.mean(dim=0, keepdim=True)
    ref_audio = ref_audio.to(device)
    
    # Extract mel features
    feature_extractor = FBankGen(sr=16000)
    mel_features, _ = feature_extractor.extract_fbank(ref_audio.cpu())
    mel_features = mel_features.to(device)
    x_lens = torch.tensor([mel_features.shape[1]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        flexicodec_output = ar_model._extract_dualcodec_features(
            ref_audio, mel=mel_features, x_lens=x_lens, sample_rate=16000
        )
    
    prompt_speech_token = flexicodec_output['semantic_codes'].squeeze(1)
    prompt_speech_token_len = flexicodec_output['speech_token_len']
    prompt_token_lengths = flexicodec_output.get('token_lengths')
    
    if prompt_token_lengths is None:
        predict_duration = False
        print("Warning: No token lengths provided for prompt, duration prediction will be disabled.")
    
    # Generate speech tokens using AR model
    result = ar_model.inference(
        text=text_tokens,
        text_len=text_lengths,
        prompt_text=None,
        prompt_text_len=None,
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=prompt_speech_token_len,
        prompt_token_lengths=prompt_token_lengths,
        embedding=None,
        beam_size=beam_size,
        top_k=top_k,
        temperature=temperature,
        max_token_text_ratio=max_token_text_ratio,
        min_token_text_ratio=min_token_text_ratio,
        predict_duration=predict_duration,
        duration_top_k=duration_top_k,
        inference_framerate=merging_threshold,  # merging_threshold is the same as inference_framerate
    )
    
    speech_tokens = result['speech_tokens']
    duration_classes = result.get('duration_classes', None)
    
    return speech_tokens, duration_classes, prompt_speech_token

