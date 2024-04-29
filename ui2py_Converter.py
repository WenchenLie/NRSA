from pathlib import Path
import os

pyuic5file = r"D:\Python311\Scripts"  # 大电脑
# pyuic5file = r"D:\Python31011\Scripts"  # 笔记本

path = (Path(__file__).parent/'ui').absolute().as_posix()  # ui文件所在目录
all_files = os.listdir(path)
ui_files = []
for file in all_files:
    if file[-3:] == '.ui':
        ui_files.append(file)
for file in ui_files:
    ui_file = path + '\\' + file
    py_file = ui_file[:-2] + 'py'
    cmd = pyuic5file + f'\\pyuic5 -x {ui_file} -o {py_file}'
    os.system(cmd)
print('转换完成！')