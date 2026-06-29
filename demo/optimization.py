"""
"""

from typing import Any
from typing import Callable
from typing import ParamSpec

import spaces
import torch
from torch.utils._pytree import tree_map_only

from optimization_utils import capture_component_call
from optimization_utils import aoti_compile


P = ParamSpec('P')


TRANSFORMER_HIDDEN_DIM = torch.export.Dim('hidden', min=4096, max=8212)

TRANSFORMER_DYNAMIC_SHAPES = {
    'hidden_states': {1: TRANSFORMER_HIDDEN_DIM},
    'img_ids': {0: TRANSFORMER_HIDDEN_DIM},
}

INDUCTOR_CONFIGS = {
    'conv_1x1_as_mm': True,
    'epilogue_fusion': False,
    'coordinate_descent_tuning': True,
    'coordinate_descent_check_all_directions': True,
    'max_autotune': True,
    'triton.cudagraphs': True,
}


def optimize_pipeline_(pipeline: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):

    @spaces.GPU(duration=1500)
    def compile_transformer():

        with capture_component_call(pipeline, 'transformer') as call:
            pipeline(*args, **kwargs)

        dynamic_shapes = tree_map_only((torch.Tensor, bool), lambda t: None, call.kwargs)
        dynamic_shapes |= TRANSFORMER_DYNAMIC_SHAPES

        pipeline.transformer.fuse_qkv_projections()

        exported = torch.export.export(
            mod=pipeline.transformer,
            args=call.args,
            kwargs=call.kwargs,
            dynamic_shapes=dynamic_shapes,
        )

        return aoti_compile(exported, INDUCTOR_CONFIGS)

    transformer_config = pipeline.transformer.config
    pipeline.transformer = compile_transformer()
    pipeline.transformer.config = transformer_config # pyright: ignore[reportAttributeAccessIssue]
