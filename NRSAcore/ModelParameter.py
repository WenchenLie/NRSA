import numpy as np


class ModelParameter(np.ndarray):
    """继承ndarray的类，但同时可以记录输入的变量名和值"""

    all_var = dict()  # 已经存在的参数
    var_num = 0  # 变量数量
    combination_num = 0  # 所有组合情况的数量

    def __new__(cls, name, value):
        obj = np.asarray(value).view(cls)
        obj.name = name
        obj.value = value
        if name in cls.all_var.keys():
            print(f'【警告】：变量`{name}`名称已存在！')
        cls.all_var[name] = value
        cls.var_num += 1
        if isinstance(obj.value, int) or isinstance(obj.value, float):
            length = 1
        else:
            length = len(obj.value)
        if cls.combination_num == 0:
            cls.combination_num = length
        else:
            cls.combination_num *= length
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, 'name', None)
        self.value = getattr(obj, 'value', None)

    def info(self):
        return f'{self.name}: {self}'
    
    @classmethod
    def print_var(cls):
        """打印所有已定义的参数"""
        for key, val in cls.all_var.items():
            print(f'{key}: {val}')



if __name__ == "__main__":
    Fy = ModelParameter('Fy', [3, 4, 9])
    Fy1 = ModelParameter('Fy4', [3, 4, 5])
    Fy1 = ModelParameter('Fy5', [1, 1])
    # ModelParameters.print_var()
    print(ModelParameter.combination_num)
    # a = np.array(1)
    # print(len(a))





