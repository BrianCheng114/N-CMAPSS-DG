import os
import time
from datetime import datetime

import torch
import logging
import numpy as np
from torch import nn
from torch.utils.data import random_split, DataLoader

from models.model_utils import *
from utils.logger import setlogger
from utils.dataset import NCMAPSS
from utils.visualization import EngineVisualizer

class ModelLoading:
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path

    def load(self):
        config = self.config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        if config['model_type'] + '_params' in config:
            model_params = config[config['model_type'] + '_params']
            model_params['input_channels'], model_params['output_channels'] = len(config['used_sensors']), 1
            model_params['dropout'] = config['dropout']
        else:
            raise ValueError(f"Model type {config['model_type']} not found in config parameters.")

        self.model = get_model(config['model_type'], model_params)       
        self.model.to(self.device)

        # 损失函数
        if config['criterion'] == "MSELoss":
            self.criterion = nn.MSELoss()
        elif config['criterion'] == "L1Loss":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {config['criterion']}")
        
    def test(self):
        config = self.config
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])

        self.test_outputs = None
        self.test_labels = None

        for deg in config['test_used_degradations']:
            test_dataset = NCMAPSS(
                data_path = config['data_path'],
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
            test_outputs = None
            test_labels = None

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    test_loss += self.criterion(outputs, labels).item()

                    if test_outputs is not None:
                        test_outputs = np.append(test_outputs, outputs.cpu().numpy(), axis=0)
                    else:
                        test_outputs = outputs.cpu().numpy()
                    
                    if test_labels is not None:
                        test_labels = np.append(test_labels, labels.cpu().numpy(), axis=0)
                    else:
                        test_labels = labels.cpu().numpy()

            if self.test_outputs is not None:
                self.test_outputs.append(test_outputs)
                self.test_labels.append([test_labels])
            else:
                self.test_outputs = [test_outputs]
                self.test_labels = [test_labels]
            
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
            title = self.config['data_path'][29:-3] + f'_Unit{deg}: '

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

if __name__ == "__main__":
    from utils.config import CONFIG

    CONFIG["save_path"] = os.path.join(CONFIG["save_root_path"], 
                                    datetime.strftime(datetime.now(), "%m%d-%H%M%S") + '-test')

    if not os.path.exists(CONFIG["save_path"]):
        os.makedirs(CONFIG["save_path"])

    setlogger(os.path.join(CONFIG["save_path"], "train.log"))
    for k, v in CONFIG.items():
        logging.info(f"{k}: {v}")

    CONFIG['model_type'] = 'LSTM'
    model_path = os.path.join(CONFIG["save_root_path"], '0208-225430-LSTM', 'Engine_LSTM.pth')

    trainer = ModelLoading(CONFIG, model_path)
    trainer.load()
    trainer.test()
    trainer.plot()