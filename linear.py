
# 使用%matplotlib命令可以将matplotlib的图表直接嵌入到Notebook之中，
# 或者使用指定的界面库显示图表，它有一个参数指定matplotlib图表的显示方式。
# inline表示将图表嵌入到Notebook中。

import random
import torch
import os
import pandas as pd
from deeplearning import  torch
import numpy as np
def f(x):
    return 3 * x ** 2 - 4 * x

# def use_svg_display():  #@save
#     """使用svg格式在Jupyter中显示绘图"""



# 放弃了找来找去发现不能显示为svg
x = np.arange(0, 3, 0.1)
print(f(x))
torch.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'] )



