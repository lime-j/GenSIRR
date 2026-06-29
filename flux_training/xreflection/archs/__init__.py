import importlib
from os import path as osp

from xreflection.utils import scandir
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from xreflection.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'xreflection.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = opt.copy()
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    rank_zero_info(f'Network [{net.__class__.__name__}] is created.')
    return net