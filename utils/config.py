CONFIG = {
    # 数据部分
    'source_domain': [1, 2, 3, 4, 5, 6, 7, 8],
    'target_domain': 9,
    'data_folder': 'E:/Datasets/NCMAPSS/',
    'data_filenames': {
        1: 'N-CMAPSS_DS01-005.h5',
        2: 'N-CMAPSS_DS02-006.h5',
        3: 'N-CMAPSS_DS03-012.h5',
        4: 'N-CMAPSS_DS04.h5',
        5: 'N-CMAPSS_DS05.h5',
        6: 'N-CMAPSS_DS06.h5',
        7: 'N-CMAPSS_DS07.h5',
        8: 'N-CMAPSS_DS08a-009.h5',
        9: 'N-CMAPSS_DS08c-008.h5',
        10: 'N-CMAPSS_DS08d-010.h5'
    },
    'sparse_idx': 10,
    'seq_length': 30,
    'stride': 10,
    'used_sensors': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'train_used_degradations': 'all',
    'test_used_degradations': [7, 8, 9, 10],

    # 模型部分
    'baseline_model': 'BaselineLSTM',
    'feature_extractor_type': 'ExtractorCNN',
    'RUL_regressor_type': 'RegressorMLP',

    # baseline模型参数
    'BaselineMLP_params': {
        'seq_length': 30,
        'hidden_dims': [256, 128, 64],
        'dropout': 0.1,
    },
    'BaselineRNN_params': {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1,
    },
    'BaselineLSTM_params': {
        'hidden_size': 64,  # 隐藏层维数
        'num_layers': 2,  # LSTM层数
        'dropout': 0.1,
    },
    'BaselineCNN_params': {
        'conv_channels': [64, 128, 256, 512],  # 卷积层输出通道数
        'kernel_size': 5,  # 卷积核大小
        'dropout': 0.1,
    },

    # 领域泛化模型参数
    'feature_dim': 320,
    'ExtractorCNN_params': {
        'conv_channels': [64, 128, 192, 256],  # 卷积层输出通道数
        'kernel_size': 5,  # 卷积核大小
        'dropout': 0.1,
    },
    'RegressorMLP_params': {
        'hidden_dims': [128, 32],
        'dropout': 0.1,
    },
    'DiscriminatorMLP_params': {
        'hidden_dims': [16],
        'dropout': 0.1,
    },
    'GateMLP_params': {
        'hidden_dims': [16],
        'dropout': 0.1,
    },

    # 训练部分
    'batch_size': 128,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'train_ratio': 0.8,
    'manual_seed': 42,
    'RUL_loss': "MSELoss",
    'save_root_path': './results',

    'optimizer': "AdamW",  # 优化器
    'optimizer_params': {
        'weight_decay': 5e-4,
        'amsgrad': True
    },
    
    'lr_scheduler': "ReduceLROnPlateau",  # 学习率调度器
    'lr_scheduler_params': {
        'mode': 'min',
        'factor': 0.5,
        'patience': 10
    },

    'early_stopping': False,  # 是否使用早停
    'early_stopping_params': {
        'patience': 10,  # 早停耐心值
        'min_delta': 0.001,  # 最小变化量
    },

    # 领域泛化部分
    'domain_alignment': True,
    'alignment_params': {
        'loss': 'MMD',
        'tradeoff': 5.0, # 不宜设置过大，会导致只优化alignment loss，RUL loss不下降
    },

    'domain_discrimination': False,
    'discrimination_params': {
        'type': 'DiscriminatorMLP',
        'loss': 'CrossEntropy',
        'tradeoff': 40.0,
    },

    'Multi_task_learning': False,
    'MoE_params': {
        'discriminator_type': 'DiscriminatorMLP',
        'gate_type': 'GateMLP',
        'loss': 'CrossEntropy',
        'entropy_tradeoff': 200.0,
        'discrimination_tradeoff': 50.0,
        'epoch1': 10,
        'epoch2': 20,
    },
}