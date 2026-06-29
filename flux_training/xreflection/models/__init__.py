import importlib
from copy import deepcopy
from os import path as osp

from xreflection.utils import scandir
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from xreflection.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'xreflection.models.{file_name}') for file_name in model_filenames]


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    rank_zero_info(f'Model [{model.__class__.__name__}] is created.')
    return model

def build_model_from_checkpoint(opt, checkpoint_path):
    model = MODEL_REGISTRY.get(opt['model_type'])
    return model.load_from_checkpoint(checkpoint_path)