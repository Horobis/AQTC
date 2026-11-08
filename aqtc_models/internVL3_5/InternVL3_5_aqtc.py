# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from ..novelty import NoveltyEstimator
from ..scoring import ImportanceFusion
from ..llava.model.aqtc_visual_compress import compress_pooled_video_tokens

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "InternVisionModel",
        "Qwen3DecoderLayer",
    ]

    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(
            self,
            config: InternVLChatConfig,
            vision_model=None,
            language_model=None,
            use_flash_attn=True,
            aqtc_enabled: bool = False,
            keep_ratio: float = 0.8,
            memory_capacity: int = 256,
            query_weight: float = 0.4,
            saliency_weight: float = 0.3,
            novelty_frames_per_segment: int = 4,
            dynamic_segment_budget: bool = False,
            dynamic_budget_strength: float = 0.5,
            budget_ceiling: float = 0.45,
            temporal_aware: bool = False,
            temporal_weight: float = 0.2,
            temporal_protect_ratio: float = 0.15,
            vision_chunk_frames: int = 4,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            architecture: str = config.llm_config.architectures[0]
            if architecture == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif architecture == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            elif architecture == 'Qwen3MoeForCausalLM':
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            elif architecture == 'Qwen3ForCausalLM':
                self.language_model = Qwen3ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{architecture} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        self.aqtc_enabled = bool(aqtc_enabled)
        self.keep_ratio = float(keep_ratio)
        self.novelty_frames_per_segment = int(novelty_frames_per_segment)
        self.dynamic_segment_budget = bool(dynamic_segment_budget)
        self.dynamic_budget_strength = float(dynamic_budget_strength)
        self.budget_ceiling = float(budget_ceiling)
        self.temporal_aware = bool(temporal_aware)
        self.temporal_weight = float(temporal_weight)
        self.temporal_protect_ratio = float(temporal_protect_ratio)
        self.vision_chunk_frames = max(1, int(vision_chunk_frames))
        self.novelty_estimator = NoveltyEstimator(llm_hidden_size, int(memory_capacity))
        self.importance_fusion = ImportanceFusion(query_weight, saliency_weight)
        self.total_original_visual_tokens = 0
        self.total_compressed_visual_tokens = 0
        self.total_original_prefill_tokens = 0
        self.total_compressed_prefill_tokens = 0
        self.total_generated_tokens = 0
        self.last_importance_scores = None

        # Transformers 5.x builds all_tied_weights_keys in post_init().
        # InternVL's original implementation omits this call, which causes
        # device-mapped checkpoint loading to fail before weights are loaded.
        self.post_init()

    def reset_novelty_memory(self) -> None:
        self.novelty_estimator.reset()

    def reset_token_stats(self) -> None:
        self.total_original_visual_tokens = 0
        self.total_compressed_visual_tokens = 0
        self.total_original_prefill_tokens = 0
        self.total_compressed_prefill_tokens = 0
        self.total_generated_tokens = 0

    def get_token_stats(self) -> dict[str, int]:
        return {
            'original_visual': int(self.total_original_visual_tokens),
            'compressed_visual': int(self.total_compressed_visual_tokens),
            'original_prefill': int(self.total_original_prefill_tokens),
            'compressed_prefill': int(self.total_compressed_prefill_tokens),
            'generated': int(self.total_generated_tokens),
        }

    def _aqtc_query_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        text_mask = input_ids != self.img_context_token_id
        text_ids = input_ids[text_mask]
        if text_ids.numel() == 0:
            return self.language_model.get_input_embeddings()(input_ids[:0])
        return self.language_model.get_input_embeddings()(text_ids)

    @torch.no_grad()
    def _compress_video_features(
            self,
            vit_embeds: torch.Tensor,
            num_patches_list: List[int],
            query_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Compress InternVL features while preserving one output group per frame.

        InternVL emits one 256-token square grid per image tile. Equal tile counts
        per frame are compressed as parallel temporal tile streams; irregular tile
        counts fall back to independent tile compression.
        """
        if not num_patches_list or len(num_patches_list) <= 1:
            return vit_embeds, [int(vit_embeds.shape[0] * vit_embeds.shape[1])]

        chunks = list(torch.split(vit_embeds, num_patches_list, dim=0))
        tile_counts = {int(x) for x in num_patches_list}
        frame_outputs = [[] for _ in chunks]

        def compress_stream(stream: torch.Tensor) -> List[torch.Tensor]:
            spatial_tokens = int(stream.shape[1])
            saliency = stream.float().norm(dim=-1).to(dtype=stream.dtype)
            self.novelty_estimator.reset()
            compressed, stats, _ = compress_pooled_video_tokens(
                stream,
                query_embeds,
                novelty_estimator=self.novelty_estimator,
                importance_fusion=self.importance_fusion,
                keep_ratio=self.keep_ratio,
                saliency_scores=saliency,
                novelty_frames_per_segment=self.novelty_frames_per_segment,
                dynamic_segment_budget=self.dynamic_segment_budget,
                dynamic_budget_strength=self.dynamic_budget_strength,
                budget_ceiling=self.budget_ceiling,
                temporal_aware=self.temporal_aware,
                temporal_weight=self.temporal_weight,
                temporal_protect_ratio=self.temporal_protect_ratio,
            )
            self.total_original_visual_tokens += int(stats['original_visual'])
            self.total_compressed_visual_tokens += int(stats['compressed_visual'])
            self.last_importance_scores = {
                'query': stats['query'],
                'saliency': stats['saliency'],
                'importance': stats['importance'],
                'novelty': stats['novelty'],
            }
            return compressed

        if len(tile_counts) == 1:
            tile_count = next(iter(tile_counts))
            tile_streams = []
            for tile_idx in range(tile_count):
                tile_streams.append(torch.stack([frame[tile_idx] for frame in chunks], dim=0))
            for stream in tile_streams:
                compressed = compress_stream(stream)
                for frame_idx, frame_tokens in enumerate(compressed):
                    frame_outputs[frame_idx].append(frame_tokens)
        else:
            # Dynamic preprocessing can produce different tile counts for frames.
            # Keep the layout valid by compressing each tile independently.
            for frame_idx, frame in enumerate(chunks):
                for tile in frame:
                    compressed = compress_stream(tile.unsqueeze(0))
                    frame_outputs[frame_idx].append(compressed[0])

        merged = []
        counts = []
        for parts in frame_outputs:
            frame_tokens = torch.cat(parts, dim=0) if parts else vit_embeds.new_empty((0, vit_embeds.shape[-1]))
            merged.append(frame_tokens)
            counts.append(int(frame_tokens.shape[0]))
        return torch.cat(merged, dim=0), counts

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.size(0))
            input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def _extract_feature_impl(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    @torch.no_grad()
    def extract_feature(
            self,
            pixel_values,
            num_patches_list: Optional[List[int]] = None,
            vision_chunk_frames: Optional[int] = None,
    ):
        """Extract visual features with bounded video-frame batches."""
        if (
                num_patches_list is None
                or len(num_patches_list) <= 1
                or vision_chunk_frames is None
                or int(vision_chunk_frames) <= 0
        ):
            return self._extract_feature_impl(pixel_values)

        frame_tiles = list(torch.split(pixel_values, num_patches_list, dim=0))
        chunk_size = max(1, int(vision_chunk_frames))
        feature_chunks = []
        for start in range(0, len(frame_tiles), chunk_size):
            frame_chunk = torch.cat(frame_tiles[start:start + chunk_size], dim=0)
            feature_chunks.append(self._extract_feature_impl(frame_chunk))
            if pixel_values.is_cuda:
                torch.cuda.empty_cache()
        return torch.cat(feature_chunks, dim=0)

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False, is_video=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        is_video = bool(is_video or question.lstrip().startswith('Frame1:'))
        use_aqtc = self.aqtc_enabled and is_video
        visual_features = None
        visual_token_counts = [self.num_image_token * num_patches for num_patches in num_patches_list]
        if use_aqtc and pixel_values is not None:
            base_inputs = tokenizer(query, return_tensors='pt')
            base_input_ids = base_inputs['input_ids'][0].to(self.device)
            query_embeds = self._aqtc_query_embeds(base_input_ids)
            raw_features = self.extract_feature(
                pixel_values,
                num_patches_list=num_patches_list,
                vision_chunk_frames=self.vision_chunk_frames,
            )
            visual_features, visual_token_counts = self._compress_video_features(
                raw_features, num_patches_list, query_embeds
            )

        for visual_token_count in visual_token_counts:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * visual_token_count + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=None if visual_features is not None else pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=visual_features,
            **generation_config
        )
        if use_aqtc:
            text_token_count = int((input_ids != img_context_token_id).sum().item())
            original_visual_tokens = self.num_image_token * sum(num_patches_list)
            compressed_visual_tokens = sum(visual_token_counts)
            self.total_original_prefill_tokens += text_token_count + original_visual_tokens
            self.total_compressed_prefill_tokens += text_token_count + compressed_visual_tokens
            self.total_generated_tokens += int(generation_output.numel())

        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None or visual_features is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.language_model.set_output_embeddings(value)
