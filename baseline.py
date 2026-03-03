import os
import time
from datetime import datetime

import torch
import logging
import numpy as np
from torch import nn
from torch.utils.data import random_split, DataLoader, ConcatDataset

from models.model_utils import *
from utils.dataset import *
from utils.logger import setlogger
from utils.tools import EarlyStopping
from utils.visualization import EngineVisualizer

class RULprediction:
    def __init__(self, config):
        self.config = config

    def setup(self):
        config = self.config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载数据集
        source_train_datasets_list = []
        source_val_datasets_list = []
        
        for domain_index in config['source_domain']:
            train_dataset = NCMAPSS(
                domain_index = domain_index,
                data_path = os.path.join(config['data_folder'], config['data_filenames'][domain_index]),
                sparse_idx = config['sparse_idx'],
                seq_length = config['seq_length'],
                stride = config['stride'],
                mode = 'train',
                used_sensors = config['used_sensors'],
                used_degradations = config['train_used_degradations']
            )

            # 数据集划分
            train_size = int(len(train_dataset) * config['train_ratio'])
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size],
                generator = torch.Generator().manual_seed(config['manual_seed'])
            )

            source_train_datasets_list.append(train_dataset)
            source_val_datasets_list.append(val_dataset)
        
        source_train_datasets = ConcatDataset(source_train_datasets_list)
        source_val_datasets = ConcatDataset(source_val_datasets_list)

        source_train_sampler = BalancedBatchSampler(source_train_datasets_list, 
                                                    len(config['source_domain']), 
                                                    config['batch_size'])
        source_val_sampler = BalancedBatchSampler(source_val_datasets_list, 
                                                  len(config['source_domain']), 
                                                  config['batch_size'])

        # 数据加载器
        self.train_loader = DataLoader(source_train_datasets, 
                                       batch_sampler = source_train_sampler,
                                       pin_memory = (self.device.type == 'cuda'))
        self.val_loader = DataLoader(source_val_datasets, 
                                     batch_sampler = source_val_sampler,
                                     pin_memory = (self.device.type == 'cuda'))

        # 初始化模型
        if config['baseline_model'] + '_params' in config:
            model_params = config[config['baseline_model'] + '_params']
            model_params['input_channels'], model_params['output_channels'] = len(config['used_sensors']), 1
        else:
            raise ValueError(f"Model type {config['baseline_model']} not found in config parameters.")

        self.model = get_model(config['baseline_model'], model_params)       
        self.model.to(self.device)

        # 损失函数
        if config['RUL_loss'] == "MSELoss":
            self.RUL_loss = nn.MSELoss()
        elif config['RUL_loss'] == "L1Loss":
            self.RUL_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {config['RUL_loss']}")
        
        # 优化器
        if config['optimizer'] == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr = config['learning_rate'], 
                weight_decay = config['optimizer_params']['weight_decay'],
                amsgrad = config['optimizer_params']['amsgrad']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
        
        # 学习率调度器
        if config['lr_scheduler'] == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode = config['lr_scheduler_params']['mode'], 
                factor = config['lr_scheduler_params']['factor'], 
                patience = config['lr_scheduler_params']['patience']
            )
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {config['lr_scheduler']}")
        
        # 早停机制
        if config['early_stopping']:
            self.early_stopper = EarlyStopping(
                patience = config['early_stopping_params']['patience'],
                min_delta = config['early_stopping_params']['min_delta'],
            )
        else:
            self.early_stopper = None
        
        # 创建保存路径
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])

        logging.info("Setup completed. Starting training...")

    def train(self):
        train_losses = []
        val_losses = []

        # 训练循环
        for epoch in range(self.config['num_epochs']):
            logging.info(f"{'-' * 5}Epoch {epoch + 1}/{self.config['num_epochs']}{'-' * 5}")
            total_loss = 0
            start_time = time.time()

            self.model.train()
            
            for inputs, labels, _ in self.train_loader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.RUL_loss(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # 验证集评估
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels, _ in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    val_loss += self.RUL_loss(outputs, labels).item()
            
            # 记录loss
            train_losses.append(total_loss/len(self.train_loader))
            val_losses.append(val_loss/len(self.val_loader))

            # 更新学习率
            self.lr_scheduler.step(val_loss/len(self.val_loader))

            logging.info(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {total_loss/len(self.train_loader):.4f} | "
                f"Val Loss: {val_loss/len(self.val_loader):.4f} | "
                f"Time: {time.time() - start_time:.2f}s | "
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            if self.early_stopper is not None:
                self.early_stopper(val_loss)
                if self.early_stopper.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

        
        # 保存loss历史
        self.train_losses = train_losses
        self.val_losses = val_losses

        # 保存模型和标准化器
        torch.save({
            'model': self.model.state_dict(),
        }, os.path.join(self.config['save_path'], f'{self.config["baseline_model"]}.pth'))      

    def test(self):
        config = self.config
        self.test_outputs = []
        self.test_labels = []

        for deg in config['test_used_degradations']:
            test_dataset = NCMAPSS(
                domain_index = config['target_domain'],
                data_path = os.path.join(config['data_folder'], config['data_filenames'][config['target_domain']]),
                sparse_idx = config['sparse_idx'],
                seq_length = config['seq_length'],
                stride = config['stride'],
                mode = 'test',
                used_sensors = config['used_sensors'],
                used_degradations = [deg]
            )
            test_loader = DataLoader(test_dataset, batch_size = self.config['batch_size'])
        
            # 测试集评估
            self.model.eval()
            test_loss = 0
            test_outputs = []
            test_labels = []

            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    test_loss += self.RUL_loss(outputs, labels).item()

                    test_outputs.extend(outputs.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())

            self.test_outputs.append(np.array(test_outputs))
            self.test_labels.append(np.array(test_labels))
            
            test_losses = test_loss/len(test_loader)
            logging.info(f"{'-' * 5}Unit{deg} Test Results{'-' * 5}")
            logging.info(f"Unit{deg} Test Loss: {test_losses:.4f}")

    def plot(self):
        # val_dataset = self.val_loader.dataset.dataset
        # self.real_outputs = val_dataset.inverse_transform(self.outputs)
        # self.real_labels = val_dataset.inverse_transform(self.labels)
        visualizer = EngineVisualizer()

        for i in range(len(self.config['test_used_degradations'])):
            deg = self.config['test_used_degradations'][i]
            title = self.config['data_filenames'][self.config['target_domain']][9:-3] + f'_Unit{deg}: '

            # 绘制预测散点图
            visualizer.plot_predictions(
                self.test_outputs[i], 
                self.test_labels[i],
                title,
                save_path = os.path.join(self.config['save_path'], f'prediction_scatter_Unit{deg}.png')
            )

            # 绘制RUL变化曲线
            visualizer.plot_RUL(
                self.test_outputs[i], 
                self.test_labels[i],
                title,
                save_path = os.path.join(self.config['save_path'], f'RUL_curve_Unit{deg}.png')
            )

        # 绘制损失函数变化曲线
        visualizer.plot_loss_curves(
            [self.train_losses, self.val_losses],
            ['Train Loss', 'Val Loss'],
            save_path = os.path.join(self.config['save_path'], 'loss_curves.png')
        )

if __name__ == "__main__":
    from utils.config import CONFIG

    for model_type in ['MLP', 'CNN', 'RNN', 'LSTM']:
        CONFIG['baseline_model'] = f'Baseline{model_type}'

        CONFIG["save_path"] = os.path.join(CONFIG["save_root_path"], 
                                    datetime.strftime(datetime.now(), "%m%d-%H%M%S") + '-' 
                                    + CONFIG["baseline_model"])
        
        if not os.path.exists(CONFIG["save_path"]):
            os.makedirs(CONFIG["save_path"])

        setlogger(os.path.join(CONFIG["save_path"], "train.log"))
        for k, v in CONFIG.items():
            logging.info(f"{k}: {v}")

        trainer = RULprediction(CONFIG)
        trainer.setup()
        trainer.train()
        trainer.test()
        trainer.plot()