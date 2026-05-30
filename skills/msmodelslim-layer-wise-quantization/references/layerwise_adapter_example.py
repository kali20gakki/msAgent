import os.path
from collections import defaultdict
from functools import lru_cache
from unittest.mock import patch

from safetensors import safe_open
from torch import nn

from msmodelslim.utils.security import json_safe_load


class LayerwiseMixin:
    @lru_cache(maxsize=1)
    def get_weight_map(self):
        index_path = os.path.join(self.model_path, "model.safetensors.index.json")
        model_index = json_safe_load(index_path)
        return model_index["weight_map"]

    def get_state_dict(self, module: nn.Module, prefix: str = ""):
        weight_map = self.get_weight_map()
        file_to_names = defaultdict(list)
        for name, _ in module.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name in weight_map:
                file_to_names[weight_map[full_name]].append(name)

        state_dict = {}
        for file_name, names in file_to_names.items():
            file_path = os.path.join(self.model_path, file_name)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in names:
                    full_name = f"{prefix}.{name}" if prefix else name
                    state_dict[name] = f.get_tensor(full_name)
        return state_dict

    def load_decoder_if_not_exist(self, model: nn.Module, name: str, idx: int):
        try:
            return model.get_submodule(name)
        except AttributeError:
            with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
                module_list: nn.ModuleList = model.model.layers
                template_module = module_list[0]
                decoder = template_module.__class__(config=self.config, layer_idx=idx)
                state_dict = self.get_state_dict(decoder, prefix=name)
                decoder.load_state_dict(state_dict)
                decoder.eval()
                module_list.append(decoder)
                return decoder

    def generate_decoder_layer(self, model: nn.Module):
        for idx in range(self.config.num_hidden_layers):
            name = f"model.layers.{idx}"
            yield name, self.load_decoder_if_not_exist(model, name=name, idx=idx)
