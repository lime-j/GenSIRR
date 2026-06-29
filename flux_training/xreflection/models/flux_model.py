import lightning as L
import torch
import os
from os import path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from xreflection.utils.registry import MODEL_REGISTRY
from xreflection.models.base_model import BaseModel
from xreflection.utils import imwrite, tensor2img
from lightning.pytorch.utilities import rank_zero_only, rank_zero_info, rank_zero_warn
from torchmetrics import MetricCollection
from deepspeed.ops.adam import FusedAdam as FusedAdamW
from xreflection.metrics import calculate_metric


@MODEL_REGISTRY.register()
class FluxModel(BaseModel):
    """
    """

    def __init__(self, opt):
        """Initialize the ClsModel.
        
        Args:
            opt (dict): Configuration options.
        """
        super().__init__(opt)

        self.cri_kontext = None
        

        
    def setup_losses(self):
        """Setup loss functions"""
        from xreflection.losses import build_loss
        # kontext flow matching loss
        self.cri_kontext = build_loss(self.opt['train']['kontext_opt'])


    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch (dict): Input batch containing 'input', 'target_t', 'target_r'.
            batch_idx (int): Batch index.
            
        Returns:
            torch.Tensor: Total loss.
        """
        # Get inputs
        inp = batch['input']
        target_t = batch['target_t']
        target_r = batch['target_r']


        loss = self.net_g.training_step(inp, target_t, cached_prompt=True)

        # Log losses
        self.log(f'train/loss', loss, prog_bar=True, sync_dist=True)

        return loss
    
    def testing(self, inp):
        with torch.no_grad():
            output_clean = self.net_g(inp)
        return output_clean

    def configure_optimizer_params(self):
        """Configure optimizer parameters.
        
        Returns:
            list: List of parameter groups.
        """
        train_opt = self.opt['train']

        # Setup different parameter groups with their learning rates
        params_lr = [
            {'params': self.net_g.get_dit_params(), 'lr': train_opt['optim_g']['dit_lr']},
        ]

        # Get optimizer configuration without modifying original config
        optim_type = train_opt['optim_g']['type']
        optim_config = {k: v for k, v in train_opt['optim_g'].items()
                        if k not in ['type', 'dit_lr']}

        return {
            'optim_type': optim_type,
            'params': params_lr,
            **optim_config,
        }

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Optimizer and scheduler configuration.
        """
        train_opt = self.opt['train']
        optimizer = FusedAdamW(params=self.net_g.get_dit_params(), 
                            lr=float(train_opt['optim_g']['dit_lr']), 
                            weight_decay=float(train_opt['optim_g']['weight_decay']),
                            betas=(0.9, 0.999),
                            eps=1e-8)

        # # Setup learning rate scheduler without modifying original config
        # scheduler_type = train_opt['scheduler']['type']
        # scheduler_config = {k: v for k, v in train_opt['scheduler'].items()
        #                     if k != 'type'}

        # if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        # elif scheduler_type == 'CosineAnnealingRestartLR':
        #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        # else:
        #     raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

        # Get the monitor metric from checkpoint config if available
        monitor_metric = self.opt.get('checkpoint', {}).get('monitor', 'val/psnr')

        return {
            "optimizer": optimizer,
        }


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """验证步骤。
        
        Args:
            batch (dict): 输入批次。
            batch_idx (int): 批次索引。
            dataloader_idx (int, optional): 数据加载器索引，用于多个验证集。
            
        Returns:
            dict: 包含清晰图像和反射图像的输出字典。
        """
        # 获取当前验证数据集的名称
        dataset_name = self.val_dataset_names[dataloader_idx]
        
        # 验证批次是否包含所需字段
        required_keys = ['input']
        for key in required_keys:
            if key not in batch:
                rank_zero_warn(f"Required key '{key}' missing from batch during validation")
                return {'error': f"Missing required key: {key}"}

        # 保存输入图像信息
        inp = batch['input']
        output = self.testing(inp)

        # 优雅地处理缺失的inp_path
        if 'inp_path' in batch and len(batch['inp_path']) > 0:
            img_name = osp.splitext(osp.basename(batch['inp_path'][0]))[0]
        else:
            # 如果缺少inp_path，生成一个后备名称
            img_name = f"sample_{batch_idx}"
            rank_zero_warn(f"'inp_path' key missing in batch, using fallback name: {img_name}")

        # 处理图像用于指标计算和可视化
        clean_img = tensor2img([output])
        target_t_img = tensor2img([batch['target_t']])

        metric_data = {'img': clean_img, 'img2': target_t_img}


        # 保存验证图像
        if self.opt['val'].get('save_img', False):
            self._save_images(clean_img, img_name, dataset_name)

        # 计算指标
        # input(f'{metric_data.keys()}, {self.opt["val"].get("metrics")}')
        if 'img2' in metric_data and self.opt['val'].get('metrics') is not None:
            for name, opt_ in self.opt['val']['metrics'].items():
                print(f'{name}, {opt_}')
                try:
                    metric_value = calculate_metric(metric_data, opt_)
                    print('metric_value', metric_value)
                    # 存储以供后续聚合
                    if dataset_name not in self.current_val_metrics:
                        self.current_val_metrics[dataset_name] = {}
                    if name not in self.current_val_metrics[dataset_name]:
                        self.current_val_metrics[dataset_name][name] = []
                    self.current_val_metrics[dataset_name][name].append(metric_value)
                except Exception as e:
                    rank_zero_warn(f"Error calculating metric '{name}': {str(e)}") 

        return {
            'output_clean': clean_img,
            'output_reflection': clean_img,
            'img_name': img_name,
            'dataset_name': dataset_name
        }

    def _save_images(self, clean_img , img_name, dataset_name):
        try:
            save_dir = osp.join(self.opt['path']['visualization'], dataset_name, img_name)
            os.makedirs(save_dir, exist_ok=True)
            if self.opt['val'].get('suffix'):
                save_clean_img_path = osp.join(save_dir, f'{img_name}_clean_{self.opt["val"]["suffix"]}_epoch_{self.current_epoch}.png')
            else:
                save_clean_img_path = osp.join(save_dir, f'{img_name}_clean_{self.opt["name"]}_epoch_{self.current_epoch}.png')

            # 保存图像
            imwrite(clean_img, save_clean_img_path)
        except Exception as e:
            rank_zero_warn(f"Error saving validation images: {str(e)}")


    def get_optimizer(self, optim_type, params, **kwargs):
        """Get optimizer based on type.
        
        Args:
            optim_type (str): Optimizer type.
            params (list): Parameter groups.
            **kwargs: Additional optimizer arguments.
            
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, **kwargs)
        elif optim_type == 'FusedAdamW':
            optimizer = FusedAdamW(params, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer


    # def on_validation_epoch_start(self):
    #     """Setup metrics collection at the start of validation epoch."""
    #     self.current_val_metrics = {}
        
    #     # 获取验证数据集名称，用于后续记录和显示
    #     if hasattr(self, 'trainer') and hasattr(self.trainer, 'val_dataloaders'):
    #         if isinstance(self.trainer.val_dataloaders, list):
    #             for idx, loader in enumerate(self.trainer.val_dataloaders):
    #                 if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
    #                     dataset_name = loader.dataset.opt['name']
    #                 else:
    #                     dataset_name = f"val_{idx}"
    #                 self.val_dataset_names[idx] = dataset_name
    #         else:
    #             # 单个验证集情况
    #             loader = self.trainer.val_dataloaders
    #             if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
    #                 dataset_name = loader.dataset.opt['name']
    #             else:
    #                 dataset_name = "val"
    #             self.val_dataset_names[0] = dataset_name

    # def on_validation_epoch_end(self):
    #     """Operations at the end of validation epoch."""
    #     # Calculate and log average metrics across all validation samples
    #     if self.current_val_metrics:
    #         total_average_metrics = {}
    #         for dataset_name, metrics in self.current_val_metrics.items():
    #             log_str = f'\n Validation [{dataset_name}] Epoch {self.current_epoch}\n'

    #             for metric_name, values in metrics.items():
                    
    #                 avg_value = sum(values) / len(values)
    #                 log_str += f'\t # {metric_name}: {avg_value:.4f}'

    #                 self.log(f'metrics/{dataset_name}/{metric_name}', avg_value, on_epoch=True, on_step=False, sync_dist=True)
    #                 if metric_name not in total_average_metrics.keys():
    #                     total_average_metrics[metric_name] = {
    #                         'val': sum(values),
    #                         'counts' : len(values)
    #                     }
    #                 else:
    #                     total_average_metrics[metric_name]['val'] += sum(values)
    #                     total_average_metrics[metric_name]['counts'] += len(values)
    #             # Log to console
    #             rank_zero_info(log_str)
    #         total_average_metrics = {
    #             k: v['val'] / v['counts'] for k, v in total_average_metrics.items()
    #         }
            
    #         log_str = f'\n Validation Epoch {self.current_epoch} Average Metrics:\n'
    #         for metric_name, metric_value in total_average_metrics.items():
    #             self.log(f'metrics/average/{metric_name}', metric_value, on_epoch=True, on_step=False, sync_dist=True)
    #             log_str += f'\t # {metric_name}: {metric_value:.4f}'
            
    #         rank_zero_info(log_str)
            
    #         self.top_psnr_epochs.append((total_average_metrics['psnr'], self.current_epoch))
    #         self.top_psnr_epochs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    #         self.top_psnr_epochs = self.top_psnr_epochs[:self.opt['val'].get('save_img_top_n', 5)]
    #         rank_zero_info(f'\t # The Best Average PSNR: {self.top_psnr_epochs[0][0]:.4f} at Epoch {self.top_psnr_epochs[0][1]}')
            
    #         self.top_ssim_epochs.append((total_average_metrics['ssim'], self.current_epoch))
    #         self.top_ssim_epochs.sort(key=lambda x: (x[0], x[1]), reverse=True)
    #         self.top_ssim_epochs = self.top_ssim_epochs[:1]
    #         rank_zero_info(f'\t # The Best Average SSIM: {self.top_ssim_epochs[0][0]:.4f} at Epoch {self.top_ssim_epochs[0][1]}\n')
            
    #         self._delete_images_not_in_top_psnr()

    # def on_test_epoch_start(self):
    #     """Operations at the start of test epoch."""
    #     """Setup metrics collection at the start of test epoch."""
    #     self.current_val_metrics = {}
    #     if hasattr(self, 'trainer') and hasattr(self.trainer, 'test_dataloaders'):
    #         if isinstance(self.trainer.test_dataloaders, list):
    #             for idx, loader in enumerate(self.trainer.test_dataloaders):
    #                 if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
    #                     dataset_name = loader.dataset.opt['name']
    #                 else:
    #                     dataset_name = f"test_{idx}"
    #                 self.val_dataset_names[idx] = dataset_name
    #         else:
    #             # 单个验证集情况
    #             loader = self.trainer.test_dataloaders
    #             if hasattr(loader.dataset, 'opt') and 'name' in loader.dataset.opt:
    #                 dataset_name = loader.dataset.opt['name']
    #             else:
    #                 dataset_name = "test"
    #             self.val_dataset_names[0] = dataset_name
    
    # def on_test_epoch_end(self):
    #     """Operations at the end of test epoch."""
    #     return self.on_validation_epoch_end()