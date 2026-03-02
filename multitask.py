import os
import time
import itertools
from datetime import datetime

import torch
import logging
import numpy as np
from torch import nn
from torch.utils.data import random_split, DataLoader, ConcatDataset

from models.model_utils import get_model
from models.MoE import MoEModel
from utils.dataset import *
from utils.logger import setlogger
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
        self.test_loaders_list = []
        
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
            self.test_loaders_list.append(test_loader)

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

        if config['MoE_params']['discriminator_type'] + '_params' in config:
            discrinimator_params = config[config['MoE_params']['discriminator_type'] + '_params']
            discrinimator_params['input_channels'] = config['feature_dim']
            discrinimator_params['output_channels'] = len(config['source_domain'])
        else:
            raise ValueError(f"Model type {config['MoE_params']['discriminator_type']} not found in config parameters.")
        
        self.discriminator = get_model(config['MoE_params']['discriminator_type'], discrinimator_params)
        self.discriminator.to(self.device)

        if config['MoE_params']['gate_type'] + '_params' in config:
            gate_params = config[config['MoE_params']['gate_type'] + '_params']
            gate_params['input_channels'] = config['feature_dim']
            gate_params['output_channels'] = len(config['source_domain'])
        else:
            raise ValueError(f"Model type {config['MoE_params']['gate_type']} not found in config parameters.")
        
        self.gate_network = get_model(config['MoE_params']['gate_type'], gate_params)
        self.gate_network.to(self.device)

        self.model = MoEModel(len(config['source_domain']), 
                              self.extractor, 
                              self.regressor, 
                              self.gate_network)
        
        # 损失函数
        if config['RUL_loss'] == "MSELoss":
            self.RUL_loss = nn.MSELoss()
        elif config['RUL_loss'] == "L1Loss":
            self.RUL_loss = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {config['RUL_loss']}")
        
        if config['MoE_params']['loss'] == "CrossEntropy":
            self.discrimination_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {config['MoE_params']['loss']}")
        
        # 优化器
        if config['optimizer'] == "AdamW":
            self.optimizer1 = torch.optim.AdamW(
                itertools.chain(
                    self.model.extractor.parameters(), 
                    self.model.regressors.parameters()
                ),
                lr = config['learning_rate'], 
                weight_decay = config['optimizer_params']['weight_decay'],
                amsgrad = config['optimizer_params']['amsgrad']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
        
        # 学习率调度器
        if config['lr_scheduler'] == "ReduceLROnPlateau":
            self.scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer1,
                mode = config['lr_scheduler_params']['mode'], 
                factor = config['lr_scheduler_params']['factor'], 
                patience = config['lr_scheduler_params']['patience']
            )
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {config['lr_scheduler']}")
        
        
        # 创建保存路径
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])

        logging.info("Setup completed. Starting training...")
    
    def train(self):
        train_losses = []
        val_losses = []

        # 第一阶段训练
        for epoch in range(self.config['MoE_params']['epoch1']):
            start_time = time.time()
            logging.info(f"{'-' * 5}Epoch {epoch + 1}/{self.config['MoE_params']['epoch1'] + self.config['MoE_params']['epoch2']}{'-' * 5}")

            total_loss = 0
            self.model.train()
            for inputs, labels, domain in self.train_loader:
                inputs, labels, domain = inputs.to(self.device), labels.to(self.device), domain.to(self.device)
                self.optimizer1.zero_grad()

                # 第一阶段前向传播
                outputs = self.model.forward_hard(inputs, domain)
                loss = self.RUL_loss(outputs, labels)

                # 反向传播
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer1.step()

                # 记录
                total_loss += loss.item()
            
            # 验证集评估
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels, domain in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model.forward_hard(inputs, domain)
                    val_loss += self.RUL_loss(outputs, labels).item()

            train_losses.append(total_loss/len(self.train_loader))
            val_losses.append(val_loss/len(self.val_loader))
            self.scheduler1.step(val_loss/len(self.val_loader))

            logging.info(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {total_loss/len(self.train_loader):.4f} | "
                f"Val Loss: {val_loss/len(self.val_loader):.4f} |"
                f"Time: {time.time() - start_time:.2f}s | "
                f"Learning Rate: {self.optimizer1.param_groups[0]['lr']:.6f}"
            )
        
        # 初始化第二阶段优化器与学习率调度器
        last_lr = self.optimizer1.param_groups[0]['lr']
        if self.config['optimizer'] == "AdamW":
            self.optimizer2 = torch.optim.AdamW(
                self.model.parameters(), 
                lr = last_lr,
                weight_decay = self.config['optimizer_params']['weight_decay'],
                amsgrad = self.config['optimizer_params']['amsgrad']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")   

        if self.config['lr_scheduler'] == "ReduceLROnPlateau":
            self.scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer2,
                mode = self.config['lr_scheduler_params']['mode'], 
                factor = self.config['lr_scheduler_params']['factor'], 
                patience = self.config['lr_scheduler_params']['patience']
            )
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {self.config['lr_scheduler']}")
        
        # 第二阶段训练
        for epoch in range(self.config['MoE_params']['epoch2']):
            start_time = time.time()
            logging.info(f"{'-' * 5}Epoch {epoch + 1 + self.config['MoE_params']['epoch1']}/{self.config['MoE_params']['epoch1'] + self.config['MoE_params']['epoch2']}{'-' * 5}")

            total_loss = 0
            total_RUL_loss = 0
            # total_entropy = 0
            total_discrimination = 0
            self.model.train() 

            for inputs, labels, domain in self.train_loader:
                inputs, labels, domain = inputs.to(self.device), labels.to(self.device), domain.to(self.device)
                self.optimizer2.zero_grad()   
                outputs, weights = self.model.forward_soft(inputs)

                # 领域分类标签
                domain_labels = nn.functional.one_hot(domain - 1,
                                        num_classes=len(self.config['source_domain'])).float()
                domain_labels = domain_labels.to(self.device)

                # 计算损失
                RUL_loss = self.RUL_loss(outputs, labels)
                # entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()                   
                discrimination = self.discrimination_loss(weights, domain_labels)
                loss = RUL_loss + \
                    self.config['MoE_params']['discrimination_tradeoff'] * discrimination

                # 反向传播
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer2.step()

                # 记录损失
                total_loss += loss.item()
                total_RUL_loss += RUL_loss.item()
                # total_entropy += entropy.item()
                total_discrimination += discrimination.item()

            # 验证集评估
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, labels, domain in self.val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, weights = self.model.forward_soft(inputs)
                    val_loss += self.RUL_loss(outputs, labels).item()

            train_losses.append(total_loss/len(self.train_loader))
            val_losses.append(val_loss/len(self.val_loader))
            self.scheduler2.step(val_loss/len(self.val_loader))

            logging.info(f"Epoch {epoch+1:03d} | "
                f"Train Loss: {total_loss/len(self.train_loader):.4f} | "
                f"Val Loss: {val_loss/len(self.val_loader):.4f} | "
                f"Time: {time.time() - start_time:.2f}s | "
                f"Learning Rate: {self.optimizer1.param_groups[0]['lr']:.6f}"
            )

            logging.info(f"RUL Loss: {total_RUL_loss/len(self.train_loader):.4f} | "
                f"Discrimination Loss: {total_discrimination/len(self.train_loader):.4f}"
            )

        # 保存loss历史
        self.train_losses = train_losses
        self.val_losses = val_losses

        # 保存模型
        torch.save(self.model.state_dict(),
                   os.path.join(self.config['save_path'], f'Engine_{self.config["feature_extractor_type"]}.pth'))   

    def test(self):
        config = self.config
        self.test_outputs = []
        self.test_labels = []

        for deg in config['test_used_degradations']:
            test_loader = self.test_loaders_list[config['test_used_degradations'].index(deg)]
        
            # 测试集评估
            self.model.eval()
            test_loss = 0
            test_outputs = []
            test_labels = []
            test_weights = []

            with torch.no_grad():
                for inputs, labels, domain in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, weights = self.model.forward_soft(inputs)
                    test_loss += self.RUL_loss(outputs, labels).item()

                    test_outputs.extend(outputs.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
                    test_weights.extend(weights.cpu().numpy())

            self.test_outputs.append(np.array(test_outputs))
            self.test_labels.append(np.array(test_labels))
            test_mean_weights = np.array(test_weights).mean(axis=0)
            
            test_losses = test_loss/len(test_loader)
            logging.info(f"{'-' * 5}Unit{deg} Test Results{'-' * 5}")
            logging.info(f"Unit{deg} Test Loss: {test_losses:.4f}")
            logging.info(f"Unit{deg} Weights: {test_mean_weights}")

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
                                       datetime.strftime(datetime.now(), "%m%d-%H%M%S")
                                       + '-MoE')
    
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


