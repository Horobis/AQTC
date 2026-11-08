#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union, Dict
import time
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        # Do not set config.rope_scaling = None under transformers>=5:
        # it clears rope_parameters and breaks Qwen2RotaryEmbedding.
        if getattr(config, "rope_parameters", None) is None:
            theta = getattr(config, "rope_theta", None) or 1000000.0
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": theta,
            }

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        # AQTC modules are attached by setup_aqtc (builder / demo); defaults keep compression off
        # until explicitly configured so accidental loads stay baseline-compatible.
        self.aqtc_enabled = False
        self.keep_ratio = 1.0
        self.novelty_frames_per_segment = 4
        self.novelty_estimator = None
        self.importance_fusion = None
        self.last_importance_scores = None
        self.total_original_visual_tokens = 0
        self.total_compressed_visual_tokens = 0
        self.temporal_aware = False
        self.temporal_weight = 0.2
        self.temporal_protect_ratio = 0.15

    def setup_aqtc(
        self,
        keep_ratio: float = 0.8,
        memory_capacity: int = 256,
        query_weight: float = 0.7,
        novelty_frames_per_segment: int = 4,
        dynamic_segment_budget: bool = False,
        dynamic_budget_strength: float = 0.5,
        budget_ceiling: float = 0.45,
        temporal_aware: bool = False,
        temporal_weight: float = 0.2,
        temporal_protect_ratio: float = 0.15,
        enabled: bool = True,
    ):
        """Attach AQTC compression modules (call after from_pretrained)."""
        from llava.model.aqtc_visual_compress import NoveltyEstimator, ImportanceFusion

        if novelty_frames_per_segment < 1:
            raise ValueError("novelty_frames_per_segment must be at least 1")
        self.keep_ratio = float(keep_ratio)
        self.novelty_frames_per_segment = int(novelty_frames_per_segment)
        query_weight = float(query_weight)
        if not 0.0 <= query_weight <= 1.0:
            raise ValueError("query_weight must be between 0 and 1")
        self.novelty_estimator = NoveltyEstimator(self.config.hidden_size, memory_capacity)
        self.importance_fusion = ImportanceFusion(query_weight, 1.0 - query_weight)
        # Dynamic budget (simplified)
        self.dynamic_segment_budget = bool(int(dynamic_segment_budget))
        self.dynamic_budget_strength = float(dynamic_budget_strength)
        self.budget_ceiling = float(budget_ceiling)
        self.temporal_aware = bool(int(temporal_aware))
        self.temporal_weight = float(temporal_weight)
        self.temporal_protect_ratio = float(temporal_protect_ratio)
        self.aqtc_enabled = bool(int(enabled))
        self.last_importance_scores = None
        self.reset_token_stats()
        return self

    def reset_novelty_memory(self) -> None:
        if self.novelty_estimator is not None:
            self.novelty_estimator.reset()

    def reset_token_stats(self) -> None:
        self.total_original_visual_tokens = 0
        self.total_compressed_visual_tokens = 0

    def get_token_stats(self) -> Dict[str, int]:
        return {
            "original_visual": int(self.total_original_visual_tokens),
            "compressed_visual": int(self.total_compressed_visual_tokens),
        }

    def get_last_importance_scores(self):
        return self.last_importance_scores

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        self._aqtc_reset_stage_times()
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_llm = time.perf_counter()
        outputs = super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._aqtc_add_stage_time("llm_generate_s", time.perf_counter() - _t_llm)
        self._aqtc_print_stage_times()
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
