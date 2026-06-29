"""
"""
import contextlib
from contextvars import ContextVar
from io import BytesIO
from typing import Any
from typing import cast
from unittest.mock import patch

import torch
from torch._inductor.package.package import package_aoti
from torch.export.pt2_archive._package import AOTICompiledModel
from torch.export.pt2_archive._package_weights import TensorProperties
from torch.export.pt2_archive._package_weights import Weights


INDUCTOR_CONFIGS_OVERRIDES = {
    'aot_inductor.package_constants_in_so': False,
    'aot_inductor.package_constants_on_disk': True,
    'aot_inductor.package': True,
}


class ZeroGPUCompiledModel:
    def __init__(self, archive_file: torch.types.FileLike, weights: Weights, cuda: bool = False):
        self.archive_file = archive_file
        self.weights = weights
        if cuda:
            self.weights_to_cuda_()
        self.compiled_model: ContextVar[AOTICompiledModel | None] = ContextVar('compiled_model', default=None)
    def weights_to_cuda_(self):
        for name in self.weights:
            tensor, properties = self.weights.get_weight(name)
            self.weights[name] = (tensor.to('cuda'), properties)
    def __call__(self, *args, **kwargs):
        if (compiled_model := self.compiled_model.get()) is None:
            constants_map = {name: value[0] for name, value in self.weights.items()}
            compiled_model = cast(AOTICompiledModel, torch._inductor.aoti_load_package(self.archive_file))
            compiled_model.load_constants(constants_map, check_full_update=True, user_managed=True)
            self.compiled_model.set(compiled_model)
        return compiled_model(*args, **kwargs)
    def __reduce__(self):
        weight_dict: dict[str, tuple[torch.Tensor, TensorProperties]] = {}
        for name in self.weights:
            tensor, properties = self.weights.get_weight(name)
            tensor_ = torch.empty_like(tensor, device='cpu').pin_memory()
            weight_dict[name] = (tensor_.copy_(tensor).detach().share_memory_(), properties)
        return ZeroGPUCompiledModel, (self.archive_file, Weights(weight_dict), True)


def aoti_compile(
    exported_program: torch.export.ExportedProgram,
    inductor_configs: dict[str, Any] | None = None,
):
    inductor_configs = (inductor_configs or {}) | INDUCTOR_CONFIGS_OVERRIDES
    gm = cast(torch.fx.GraphModule, exported_program.module())
    assert exported_program.example_inputs is not None
    args, kwargs = exported_program.example_inputs
    artifacts = torch._inductor.aot_compile(gm, args, kwargs, options=inductor_configs)
    archive_file = BytesIO()
    files: list[str | Weights] = [file for file in artifacts if isinstance(file, str)]
    package_aoti(archive_file, files)
    weights, = (artifact for artifact in artifacts if isinstance(artifact, Weights))
    return ZeroGPUCompiledModel(archive_file, weights)


@contextlib.contextmanager
def capture_component_call(
    pipeline: Any,
    component_name: str,
    component_method='forward',
):

    class CapturedCallException(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    class CapturedCall:
        def __init__(self):
            self.args: tuple[Any, ...] = ()
            self.kwargs: dict[str, Any] = {}

    component = getattr(pipeline, component_name)
    captured_call = CapturedCall()

    def capture_call(*args, **kwargs):
        raise CapturedCallException(*args, **kwargs)

    with patch.object(component, component_method, new=capture_call):
        try:
            yield captured_call
        except CapturedCallException as e:
            captured_call.args = e.args
            captured_call.kwargs = e.kwargs
