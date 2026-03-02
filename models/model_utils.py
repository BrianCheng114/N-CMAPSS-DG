import importlib

def get_model(model_name: str, model_params: dict):
    """
    动态加载 models/ 目录下的模型
    Args:
        model_name: 模型名称
        model_params: 传入模型的初始化参数
    Returns: 
        model_class: 模型实例
    """
    module = importlib.import_module(f"models.{model_name}")
    class_name = 'Engine' + model_name 
    model_class = getattr(module, class_name)
    return model_class(**model_params)