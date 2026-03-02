import os
import time
import itertools
from datetime import datetime

import torch
import logging
import numpy as np
from torch import nn
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torch.nn.functional import one_hot

from models.model_utils import get_model
from utils.dataset import *
from utils.logger import setlogger
from utils.loss import L2, MMD, JMMD, CORAL
from utils.tools import *
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
        if config['feature_extractor_type'] + '_params' in config:
            extractor_params = config[config['feature_extractor_type'] + '_params']
            extractor_params['input_channels'] = len(config['used_sensors'])
            extractor_params['output_channels'] = config['feature_dim']
        else:
            raise ValueError(f"Model type {config['feature_extractor_type']} not found in config parameters.")

        self.extractor = get_model(config['feature_extractor_type'], extractor_params)       
        self.extractor.to(self.device)

        if config['RUL_regressor_type'] + '_params' in config:
            regressor_params = config[config['RUL_regressor_type'] + '_params']
            regressor_params['input_channels'] = config['feature_dim']
            regressor_params['output_channels'] = 1
        else:
            raise ValueError(f"Model type {config['RUL_regressor_type']} not found in config parameters.")
        
        self.regressor = get_model(config['RUL_regressor_type'], regressor_params)
        self.regressor.to(self.device)

        if config['domain_discrimination']:
            if config['discrimination_params']['type'] + '_params' in config:
                discrinimator_params = config[config['discrimination_params']['type'] + '_params']
                discrinimator_params['input_channels'] = config['feature_dim']
                discrinimator_params['output_channels'] = len(config['source_domain'])
            else:
                raise ValueError(f"Model type {config['discrimination_params']['type']} not found in config parameters.")
            
            self.discriminator = get_model(config['discrimination_params']['type'], discrinimator_params)
            self.discriminator.to(self.device)

        # 损失函数
        if config['RUL_loss'] == "MSELoss":
            self.RUL_loss = nn.MSELoss()
        elif config['RUL_loss'] == "L1Loss":
            self.RUL_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {config['RUL_loss']}")

        if config['domain_alignment']:
            if config['alignment_params']['loss'] == 'L2':
                self.alignment_loss = L2
            elif config['alignment_params']['loss'] == 'MMD':
                self.alignment_loss = MMD
            elif config['alignment_params']['loss'] == 'JMMD':
                self.alignment_loss = JMMD
            elif config['alignment_params']['loss'] == 'CORAL':
                self.alignment_loss = CORAL
            else:
                raise ValueError(f"Unsupported alignment loss function: {config['alignment_loss']}")
        
        if config['domain_discrimination']:
            if config['discrimination_params']['loss'] == 'CrossEntropy':
                self.discrimination_loss = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unsupported discrimination loss function: {config['discrimination_loss']}")

        # 优化器
        if config['optimizer'] == "AdamW":
            if config['domain_discrimination']:
                self.optimizer = torch.optim.AdamW(
                    itertools.chain(
                        self.extractor.parameters(), 
                        self.regressor.parameters(),
                        self.discriminator.parameters()
                    ),
                    lr = config['learning_rate'], 
                    weight_decay = config['optimizer_params']['weight_decay'],
                    amsgrad = config['optimizer_params']['amsgrad']
                )
            else:
                self.optimizer = torch.optim.AdamW(
                    itertools.chain(
                        self.extractor.parameters(), 
                        self.regressor.parameters()
                    ),
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
        train_RUL_losses = []
        val_losses = []

        if self.config['domain_alignment']:
            train_alignment_losses = []
        if self.config['domain_discrimination']:
            train_discrimination_losses = []

        # 训练循环
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            logging.info(f"{'-' * 5}Epoch {epoch + 1}/{self.config['num_epochs']}{'-' * 5}")

            total_loss = 0
            total_RUL_loss = 0
            if self.config['domain_alignment']:
                total_alignment_loss = 0
            if self.config['domain_discrimination']:
                total_discrimination_loss = 0

            # 设定模型为训练状态
            self.extractor.train()
            self.regressor.train()
            if self.config['domain_discrimination']:
                self.discriminator.train()
            
            for inputs, labels, domain in self.train_loader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()

                # 前向传播
                features = self.extractor(inputs)
                outputs = self.regressor(features)
                RUL_loss = self.RUL_loss(outputs, labels)
                loss = RUL_loss.clone()  # 确保RUL_loss不被alignment_loss修改
                
                # 计算领域泛化的判别损失
                if self.config['domain_discrimination']:
                    features_reversed = GRL.apply(features, self.config['discrimination_params']['tradeoff'])
                    domain_predictions = self.discriminator(features_reversed)  # 判别器不参与特征提取器的反向传播

                    domain_labels = one_hot(domain - 1,
                                            num_classes=len(self.config['source_domain'])).float()
                    domain_labels = domain_labels.to(self.device)

                    discrimination_loss = self.discrimination_loss(domain_predictions, domain_labels)
                    loss += discrimination_loss

                # 计算领域泛化的对齐损失
                if self.config['domain_alignment']:
                    alignment_loss = torch.tensor(0.0, device=self.device)  # 初始化对齐损失为0
                    if domain.dim() == 2:
                        domain = domain.squeeze(1)

                    splitted_features = {}

                    for d in torch.unique(domain):
                        idx = torch.where(domain == d)[0]
                        splitted_features[int(d.item())] = features[idx]
                    
                    values = list(splitted_features.values())
                    stacked = torch.stack(values, dim=0)
                    mean_features = stacked.mean(dim=0)
                    
                    for domain_features in values:       
                        alignment_loss += self.alignment_loss(domain_features, mean_features)

                    alignment_loss /= len(values)
                    loss += alignment_loss * self.config['alignment_params']['tradeoff']

                loss.backward()
                if self.config['domain_discrimination']:
                    torch.nn.utils.clip_grad_norm_(itertools.chain(self.extractor.parameters(), 
                                                                   self.regressor.parameters(), 
                                                                   self.discriminator.parameters()), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(itertools.chain(self.extractor.parameters(), 
                                                                   self.regressor.parameters()), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_RUL_loss += RUL_loss.item()
                if self.config['domain_alignment']:
                    total_alignment_loss += alignment_loss.item() 
                if self.config['domain_discrimination']:
                    total_discrimination_loss += discrimination_loss.item()
            
            # 验证集评估
            self.extractor.eval()
            self.regressor.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels, domain in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    features = self.extractor(inputs)
                    outputs = self.regressor(features)
                    val_loss += self.RUL_loss(outputs, labels).item()
            
            # 记录loss
            train_losses.append(total_loss/len(self.train_loader))
            train_RUL_losses.append(total_RUL_loss/len(self.train_loader))
            val_losses.append(val_loss/len(self.val_loader))

            if self.config['domain_alignment']:
                train_alignment_losses.append(total_alignment_loss/len(self.train_loader))
            if self.config['domain_discrimination']:
                train_discrimination_losses.append(total_discrimination_loss/len(self.train_loader))

            # 更新学习率
            self.lr_scheduler.step(val_loss/len(self.val_loader))

            logging.info(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {total_loss/len(self.train_loader):.4f} | "
                f"Val Loss: {val_loss/len(self.val_loader):.4f} |"
                f"Time: {time.time() - start_time:.2f}s | "
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            loss_info = f"RUL Loss: {total_RUL_loss/len(self.train_loader):.4f}"
            if self.config['domain_alignment']:
                loss_info += f" | Alignment Loss: {total_alignment_loss/len(self.train_loader):.4f}"
            if self.config['domain_discrimination']:
                loss_info += f" | Discrimination Loss: {total_discrimination_loss/len(self.train_loader):.4f}"

            logging.info(loss_info)

            if self.config['early_stopping']:
                self.early_stopper(val_loss)
                if self.early_stopper.early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

        
        # 保存loss历史
        self.train_losses = train_losses
        self.val_losses = val_losses

        # 保存模型和标准化器
        if self.config['domain_discrimination']:
            torch.save({
                'extractor': self.extractor.state_dict(),
                'regressor': self.regressor.state_dict(),
                'discriminator': self.discriminator.state_dict(),
            }, os.path.join(self.config['save_path'], f'Engine_{self.config["feature_extractor_type"]}.pth'))      
        else:
            torch.save({
                'extractor': self.extractor.state_dict(),
                'regressor': self.regressor.state_dict(),
            }, os.path.join(self.config['save_path'], f'Engine_{self.config["feature_extractor_type"]}.pth'))      

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
            self.extractor.eval()
            self.regressor.eval()
            test_loss = 0
            test_outputs = []
            test_labels = []

            with torch.no_grad():
                for inputs, labels, domain in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    features = self.extractor(inputs)
                    outputs = self.regressor(features)
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
            self.train_losses,
            self.val_losses,
            save_path = os.path.join(self.config['save_path'], 'loss_curves.png')
        )

if __name__ == "__main__":
    from utils.config import CONFIG

    CONFIG["save_path"] = os.path.join(CONFIG["save_root_path"], 
                                    datetime.strftime(datetime.now(), "%m%d-%H%M%S"))
    if CONFIG['domain_alignment']:
        CONFIG["save_path"] += '-' + CONFIG["alignment_params"]["loss"]
    if CONFIG['domain_discrimination']:
        CONFIG["save_path"] += '-Discrimination'
        
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