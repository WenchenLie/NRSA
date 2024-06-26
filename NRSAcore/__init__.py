from pathlib import Path
from NRSAcore.Task import _Task
from NRSAcore.Analysis import _SDOFmodel


class Task:
    """
    生成SDOF分析任务  
    关键实例属性：  
    (1) self.paras = {参数名: (参数值, 参数类型)}  
    参数名: str  
    参数值：int | float | list  
    参数类型: Literal[1, 2, 3], 1-常数型, 2-独立参数, 3-从属参数  
    (2) self.task_info, 用于生成json文件的记录所有SDOF模型信息的字典  
    (3) self.independent_paras = list[str], 所有独立参数的参数名  
    (4) self.dependent_paras = dict[参数名, list[映射函数, *独立参数名]]  
    (5) self.constant_paras = list[str], 所有常数型参数的参数名  
    """
    def __new__(cls, task_name: str, working_directory: str | Path):
        """创建一个分析任务

        Args:
            task_name (str): 任务名称
            working_directory (str | Path): 工作路径文件夹
        """
        obj = _Task(task_name, working_directory)
        return obj


class SDOFmodel:
    def __new__(cls,
            records_file: Path | str, 
            overview_file: Path | str,
            SDOFmodel_file: Path | str,
            output_dir: Path | str
        ):
        """导入地震动文件、模型概览、SDOF模型参数，设置输出文件夹路径

        Args:
            records_file (Path | str): 地震动文件(.pkl)
            overview_file (Path | str): 模型概览文件(.json)
            SDOFmodel_file (Path | str): SDOF模型参数(.csv)
            output_dir (Path | str): 输出文件夹路径
        """
        obj = _SDOFmodel(records_file, overview_file, SDOFmodel_file, output_dir)
        return obj


