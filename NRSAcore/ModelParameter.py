from pathlib import Path
import numpy as np
import sys
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.absolute()))
    from utils.utils import ModelParameter_Error


class ModelParameter(np.ndarray):
    """继承ndarray的类，但同时可以记录输入的变量名和值"""

    all_var = dict()  # 已经存在的参数
    var_num = 0  # 变量数量
    combination_num = 0  # 所有组合情况的数量
    ufunc = [np.add, np.subtract, np.multiply, np.divide, np.power, np.square]

    def __new__(cls, name, value):
        obj = np.asarray(value).view(cls)
        obj.name = name
        obj.value = cls._get_value(value)
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

    def __len__(self) -> int:
        if isinstance(self.value, (float, int)):
            return 1
        else:
            return len(self.value)
        
    def to_list(self) -> list:
        if len(self) == 1:
            return [self.value]
        else:
            return list(self.value)
    
    @classmethod
    def _get_value(cls, value):
        if not isinstance(value, ModelParameter):
            return value
        else:
            return cls._get_value(value.value)

    def info(self):
        return f'{self.name}: {self}'
    
    @classmethod
    def print_var(cls):
        """打印所有已定义的参数"""
        for key, val in cls.all_var.items():
            print(f'{key}: {val}')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # 调用原始的 ufunc 计算结果
        args = [i.view(np.ndarray) if isinstance(i, ModelParameter) else i for i in inputs]
        result = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if result is not None and method == '__call__':
            if isinstance(result, np.ndarray):
                result = result.view(type(self))
                result.name = "result"
                if ufunc in self.ufunc:
                    values = [getattr(i, 'value', i) for i in inputs if isinstance(i, ModelParameter)]
                    result.value = ufunc(*values, **kwargs)
                else:
                    raise ModelParameter_Error(f'不支持运算：{ufunc}')
        return result



if __name__ == "__main__":
    g = ModelParameter('g', 9800)
    m = ModelParameter('m', 1)
    T = ModelParameter('T', np.arange(0.2, 2.2, 0.2))
    Cy = ModelParameter('Cy', [0.4, 0.8, 1.2])
    Fy = ModelParameter('Fy', Cy * m * g)
    print(Fy)



