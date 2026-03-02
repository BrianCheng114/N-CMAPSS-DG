import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, BatchSampler


class NCMAPSS(Dataset):
    def __init__(self,
                 domain_index: int,
                 data_path: str,
                 sparse_idx: int,
                 seq_length: int,
                 stride: int,
                 mode: str,
                 used_sensors: list,
                 used_degradations = 'all'
                 ):
        '''
        NCMAPSS数据集加载器
        Args:
            domain_index: 域索引，整数类型
            data_path: 数据集文件路径
            sparse_idx: 稀疏采样间隔
            seq_length: 滑动窗口长度
            stride: 窗口滑动步长
            mode: 训练/测试模式
            used_sensors: 使用的传感器编号列表
            used_degradations: 使用的退化数据编号(训练集1-6, 测试集7-10)
        '''

        # 加载数据
        self.domain = domain_index

        with h5py.File(data_path, 'r') as hdf:
            if mode == 'train':
                self.x = np.array(hdf.get('X_s_dev'))[::sparse_idx, used_sensors]
                self.y = np.array(hdf.get('Y_dev'))[::sparse_idx, :]
                auxiliary = np.array(hdf.get('A_dev'), dtype = int)[::sparse_idx, 0]
            elif mode == 'test':
                self.x = np.array(hdf.get('X_s_test'))[::sparse_idx, used_sensors]
                self.y = np.array(hdf.get('Y_test'))[::sparse_idx, :]
                auxiliary = np.array(hdf.get('A_test'), dtype = int)[::sparse_idx, 0]
            else:
                raise ValueError("Mode should be 'train' or 'test'")
        
        if used_degradations == 'all':
            used_degradations = np.unique(auxiliary).tolist()

        if np.all(np.isin(used_degradations, auxiliary)):
            # 只使用部分几段退化数据
            self.inputs = []
            self.labels = []
            for degradation in used_degradations:
                idx = np.where(auxiliary == degradation)
                x_tmp = self.x[idx, :].squeeze()
                y_tmp = self.y[idx]
                # print(inputs_tmp.shape, labels_tmp.shape)
                for i in range(0, len(y_tmp) - seq_length + 1, stride):
                    seq = x_tmp[i:i + seq_length, :] # [seq, features = 14]
                    label = y_tmp[i + seq_length - 1] # [1]
                    self.inputs.append(seq)
                    self.labels.append(label)

        else:
            raise ValueError("Some specified degradations are not present in the dataset.")
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # seq, label = self.sequences[idx]
        # 输入形状: [seq, inputs=14] 
        # 输出形状: [1]
        # print(seq.shape, label.shape)
        return (torch.FloatTensor(self.inputs[idx]),
                torch.FloatTensor(self.labels[idx]),
                self.domain)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, datasets, num_domains, batch_size):
        '''
        平衡多域样本的BatchSampler, 确保每个Batch中各域样本数量相等
        Args:
            datasets: ConcatDataset对象, 包含多个域的数据集
            num_domains: 域的数量
            batch_size: 每个Batch的总样本数量
        '''

        self.samples_per_domain = batch_size // num_domains

        # 记录各域在ConcatDataset中的索引范围
        self.domain_indices = []
        start = 0
        for dataset in datasets:
            end = start + len(dataset)
            self.domain_indices.append(np.arange(start, end))
            start = end

    def __iter__(self):
        # 为每个域打乱索引
        shuffled_indices = [np.random.permutation(indices) for indices in self.domain_indices]

        # 计算最小可用batch数
        self.min_batches = min(len(indices) // self.samples_per_domain for indices in shuffled_indices)

        for i in range(self.min_batches):
            batch = []
            for domain_indices in shuffled_indices:
                start = i * self.samples_per_domain
                end = start + self.samples_per_domain
                batch.extend(domain_indices[start:end])
            np.random.shuffle(batch)  # 打乱不同域样本的顺序
            yield batch

    def __len__(self):
        return self.min_batches


if __name__ == '__main__':
    dataset = NCMAPSS(data_path = 'E:/Datasets/NCMAPSS/N-CMAPSS_DS06.h5',
                      sparse_idx = 1,
                      seq_length = 50,
                      stride = 10,
                      mode = 'train',
                      used_sensors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                      used_degradations = [1, 2, 3, 4, 5, 6])
    print('Length of dataset: ' + str(len(dataset)))
    seq, label = dataset[0]
    print('Sequence shape: ' + str(seq.shape))
    print('Label shape: ' + str(label.shape))
    print(torch.FloatTensor([6]).shape)
