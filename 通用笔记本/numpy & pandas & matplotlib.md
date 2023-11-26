## Numpy

Numpy 在数值运算上效率优于python内置的list, 所以熟练掌握是必要的。

Numpy主要分为两个核心部分，N维数组对象 Ndarry  和 通用函数对象 Ufunc

当然可以！下面是关于NumPy库的各个常用模块的详细介绍，包括中文名称、英文名称、功能介绍和解决的场景。我会以Markdown表格的形式呈现给你。

| 中文名称       | 英文名称                      | 介绍                                                         | 解决场景                       |
| -------------- | ----------------------------- | ------------------------------------------------------------ | ------------------------------ |
| 数组对象       | ndarray (N-dimensional array) | 多维数组对象，用于存储同类型的元素，支持矢量化操作和广播运算。 | 数值计算、线性代数、统计分析等 |
| 通用函数       | Universal Functions (ufunc)   | 快速的元素级数组函数，对数组中的元素逐个进行操作，支持矢量化运算。 | 数值计算、数学运算、逻辑运算等 |
| 索引和切片     | Indexing and Slicing          | 用于访问和修改数组中的元素，可以通过索引、切片和布尔掩码进行操作。 | 数据访问、数据修改、数据筛选等 |
| 广播           | Broadcasting                  | 对不同形状的数组进行自动的元素级运算，使得不同尺寸的数组可以进行计算。 | 处理形状不同的数组、矩阵运算等 |
| 线性代数       | Linear Algebra                | 提供了线性代数运算的函数，如矩阵乘法、特征值分解、奇异值分解等。 | 线性代数计算、矩阵运算等       |
| 随机数         | Random Sampling (random)      | 生成各种概率分布的随机数，包括均匀分布、正态分布、泊松分布等。 | 模拟实验、概率分析、随机抽样等 |
| 快速傅里叶变换 | Fast Fourier Transform (fft)  | 提供了快速傅里叶变换算法，用于信号处理、图像处理和频谱分析等。 | 信号处理、频谱分析、图像处理等 |
| 文件输入输出   | File Input/Output (IO)        | 读取和写入数组数据到磁盘文件，支持多种数据格式，如文本文件、二进制文件等。 | 数据存储、数据读取、数据导出等 |
| 结构化数组     | Structured Arrays             | 创建和操作具有复合数据类型（结构体）的数组，可以指定字段名称和数据类型。 | 处理结构化数据、数据库操作等   |
| 掩码数组       | Masked Arrays                 | 在数组中使用掩码标记无效或缺失的数据，进行计算时可以自动忽略掩码元素。 | 缺失数据处理、数据过滤等       |

### Ndarray

#### 数组属性

当谈论NumPy数组的属性时，我们通常指的是数组对象本身的一些特征和元数据。下面是一些常见的NumPy数组属性及其说明，我将以Markdown表格的形式呈现给你。

| 名称     | 说明                                                         |
| -------- | ------------------------------------------------------------ |
| shape    | 数组的维度，表示每个维度的大小。例如，(3, 4) 表示一个二维数组，有3行和4列。 |
| dtype    | 数组元素的数据类型，例如`int64`、`float32`、`bool`等。       |
| ndim     | 数组的维度数量，也称为数组的秩。例如，二维数组的ndim为2。    |
| size     | 数组中元素的总数，等于各个维度大小的乘积。                   |
| itemsize | 数组中每个元素的字节大小。例如，`int64`类型的元素占8个字节。 |
| nbytes   | 数组中所有元素的总字节数，等于`itemsize * size`。            |
| real     | 复数数组的实部。对于实数数组，返回数组本身。                 |
| imag     | 复数数组的虚部。对于实数数组，返回全零数组。                 |
| flat     | 返回一个迭代器，用于以扁平化方式迭代数组中的元素。           |
| strides  | 表示在每个维度上需要移动多少字节来获取下一个元素。           |
| data     | 数组的缓冲区，包含数组的实际元素。                           |

#### 创建数组

当使用NumPy库处理数据时，有多种方法可以创建数组。下面是一些常用的方法，并以Markdown表格的形式列出它们的名称和说明：

| 名称               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| `numpy.array()`    | 从列表、元组或其他数组对象创建一个NumPy数组。                |
| `numpy.zeros()`    | 创建一个指定形状的全零数组。                                 |
| `numpy.ones()`     | 创建一个指定形状的全1数组。                                  |
| `numpy.empty()`    | 创建一个指定形状的空数组，数组元素的值是未初始化的。         |
| `numpy.arange()`   | 根据指定的开始值、结束值和步长创建一个一维数组。             |
| `numpy.linspace()` | 在指定的开始值和结束值之间创建一个一维数组，可以指定数组的长度。 |
| `numpy.logspace()` | 在指定的开始值和结束值之间以对数刻度创建一个一维数组。       |
| `numpy.eye()`      | 创建一个具有对角线为1的二维数组，其他位置为0。               |

```python
import numpy as np

# numpy.array()
arr = np.array([1, 2, 3])  # 参数: 输入的列表、元组或其他数组对象

# numpy.zeros()
zeros_arr = np.zeros((3, 3))  # 参数: 形状

# numpy.ones()
ones_arr = np.ones((2, 2))  # 参数: 形状

# numpy.empty()
empty_arr = np.empty((2, 2))  # 参数: 形状

# numpy.arange()
arange_arr = np.arange(0, 10, 2)  # 参数: 开始值、结束值、步长

# numpy.linspace()
linspace_arr = np.linspace(0, 1, 5)  # 参数: 开始值、结束值、数组长度

# numpy.logspace()
logspace_arr = np.logspace(0, 3, 4)  # 参数: 开始指数、结束指数、数组长度

# numpy.eye()
eye_arr = np.eye(3)  # 参数: 数组的大小

# numpy.random.rand()
rand_arr = np.random.rand(3, 3)  # 参数: 形状

# numpy.random.randn()
randn_arr = np.random.randn(2, 2)  # 参数: 形状

# numpy.random.randint()
randint_arr = np.random.randint(0, 10, (2, 2))  # 参数: 最小值、最大值、形状

# numpy.full()
full_arr = np.full((2, 2), 7)  # 参数: 形状、填充值

# numpy.tile()
tile_arr = np.tile([1, 2], 3)  # 参数: 数组、重复次数

# numpy.repeat()
repeat_arr = np.repeat([1, 2, 3], 3)  # 参数: 数组、重复次数
```

#### Random 随机数

NumPy的random模块提供了多种随机数生成函数，用于生成各种类型的随机数。以下是random模块的全部随机数生成函数，并附有说明和对应的模板代码：

| 名称                         | 说明                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| `numpy.random.rand()`        | 生成指定形状的[0,1)之间均匀分布的随机数。                    |
| `numpy.random.randn()`       | 生成指定形状的符合标准正态分布的随机数（平均值为0，标准差为1）。 |
| `numpy.random.randint()`     | 生成指定范围内的随机整数。                                   |
| `numpy.random.random()`      | 生成[0,1)之间均匀分布的随机数，形状由参数指定。              |
| `numpy.random.uniform()`     | 生成指定范围内均匀分布的随机数。                             |
| `numpy.random.normal()`      | 生成指定均值和标准差的正态分布随机数。                       |
| `numpy.random.seed()`        | 设置随机数生成的种子，以便重现随机数序列。                   |
| `numpy.random.shuffle()`     | 随机打乱数组的顺序。                                         |
| `numpy.random.choice()`      | 从给定的一维数组中随机选择元素。                             |
| `numpy.random.permutation()` | 随机排列一个序列或数组的元素。                               |

以下是对应函数的模板代码，包括参数注释：

```python
import numpy as np

# numpy.random.rand()
rand_arr = np.random.rand(3, 3)  # 参数: 形状

# numpy.random.randn()
randn_arr = np.random.randn(2, 2)  # 参数: 形状

# numpy.random.randint()
randint_arr = np.random.randint(0, 10, (2, 2))  # 参数: 最小值、最大值、形状

# numpy.random.random()
random_arr = np.random.random((3, 3))  # 参数: 形状

# numpy.random.uniform()
uniform_arr = np.random.uniform(0, 1, (2, 2))  # 参数: 最小值、最大值、形状

# numpy.random.normal()
normal_arr = np.random.normal(0, 1, (2, 2))  # 参数: 均值、标准差、形状

# numpy.random.seed()
np.random.seed(0)  # 参数: 种子值

# numpy.random.shuffle()
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)  # 参数: 数组

# numpy.random.choice()
choice_arr = np.random.choice([1, 2, 3, 4, 5], size=(2, 2))  # 参数: 一维数组、形状

# numpy.random.permutation()
permutation_arr = np.random.permutation([1, 2, 3, 4, 5])  # 参数: 一维数组
```

#### 数组变换

下面是NumPy中一些常用的数组变换函数的详细说明、模板代码和参数注释。

| 函数名称         | 说明                                   | 模板代码                                                     |
| ---------------- | -------------------------------------- | ------------------------------------------------------------ |
| `np.reshape`     | 改变数组的形状，不改变数据本身         | `np.reshape(a, newshape, order='C')`<br>参数：<br>`a`：要改变形状的数组<br>`newshape`：新的形状，可以是整数或者元组<br>`order`：可选参数，指定元素在内存中的存储顺序，默认为'C' (按行) |
| `np.resize`      | 改变数组的形状，可以改变数据本身       | `np.resize(a, new_shape)`<br>参数：<br>`a`：要改变形状的数组<br>`new_shape`：新的形状，可以是整数或者元组 |
| `np.transpose`   | 将数组的维度进行转置                   | `np.transpose(a, axes=None)`<br>参数：<br>`a`：要转置的数组<br>`axes`：可选参数，指定转置的轴顺序 |
| `np.swapaxes`    | 交换数组的两个轴                       | `np.swapaxes(a, axis1, axis2)`<br>参数：<br>`a`：要交换轴的数组<br>`axis1`：第一个轴的索引<br>`axis2`：第二个轴的索引 |
| `np.flatten`     | 将多维数组转换为一维数组               | `np.flatten(a, order='C')`<br>参数：<br>`a`：要转换的数组<br>`order`：可选参数，指定元素在内存中的存储顺序，默认为'C' (按行) |
| `np.ravel`       | 将多维数组转换为一维数组，返回一个视图 | `np.ravel(a, order='C')`<br>参数：<br>`a`：要转换的数组<br>`order`：可选参数，指定元素在内存中的存储顺序，默认为'C' (按行) |
| `np.squeeze`     | 从数组的形状中删除单维度条目           | `np.squeeze(a, axis=None)`<br>参数：<br>`a`：要删除单维度条目的数组<br>`axis`：可选参数，指定要删除的轴 |
| `np.expand_dims` | 在数组形状中插入新的轴                 | `np.expand_dims(a, axis)`<br>参数：<br>`a`：要插入新轴的数组<br>`axis`：要插入的位置 |



### Matrix & Ufunc

#### Matrix 属性

| T    | 返回矩阵的转置       | `matrix.T` |
| ---- | -------------------- | ---------- |
| H    | 返回矩阵的共轭转置   | `matrix.H` |
| I    | 返回矩阵的逆矩阵     | `matrix.I` |
| A    | 返回矩阵的数据的副本 | `matrix.A` |

#### Ufunc 函数运算

实际上运用Ufunc函数比math库的函数要快很多。



## Matplotlib

`fig.savefig()`是Matplotlib库中Figure对象的方法，用于将Figure保存为图像文件。它的主要功能是将绘制的图形保存为常见的图像格式（如PNG、JPEG、SVG等），以便后续使用或共享。

以下是`fig.savefig()`的用法详解：

```python
fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None)
```

参数解释：

- `fname`（必需）：指定保存的文件路径和文件名。可以是相对路径或绝对路径。文件扩展名决定了保存的图像格式，例如".png"表示保存为PNG格式，".jpg"表示保存为JPEG格式等。
- `dpi`（可选）：指定保存图像的分辨率（每英寸点数）。默认值为`None`，表示使用默认分辨率。较高的dpi值将产生更高分辨率的图像文件，但文件大小也会相应增加。
- `facecolor`（可选）：指定图像的背景色，默认为白色（'w'）。
- `edgecolor`（可选）：指定图像边缘的颜色，默认为白色（'w'）。
- `orientation`（可选）：指定图像的方向，可选值为'portrait'（纵向）或'landscape'（横向）。默认为'portrait'。
- `format`（可选）：指定保存的图像格式。如果未指定，则根据文件名的扩展名自动选择格式。常见的格式有'png'、'jpeg'、'svg'等。
- `transparent`（可选）：指定是否将图像背景设置为透明。默认为`False`，表示不透明。
- `bbox_inches`（可选）：指定需要保存的部分。默认为`None`，表示保存整个Figure。可以使用不同的方式指定，如`'tight'`表示保存所有内容但裁剪空白区域，也可以传递一个元组表示需要保存的区域的边界框（以英寸为单位）。
- `pad_inches`（可选）：指定边界框周围的填充大小（以英寸为单位）。默认值为0.1。
- `metadata`（可选）：指定保存图像的元数据，如作者、标题等。默认为`None`。

下面是一个示例，演示如何使用`fig.savefig()`保存Figure为PNG图像文件：

```python
import matplotlib.pyplot as plt

# 创建Figure和Axes对象
fig, ax = plt.subplots()

# 绘制图形
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

# 保存Figure为PNG图像文件
fig.savefig('my_plot.png')
```

执行上述代码后，将在当前工作目录中生成一个名为"my_plot.png"的PNG图像文件，其中包含绘制的图形。

注意：在调用`fig.savefig()`之后，通常需要调用`plt.close()`或`plt.clf()`来释放Figure对象和相关资源，以避免内存泄漏。

要添加元数据（metadata）到保存的图像中，可以使用`metadata`参数来指定保存图像的元数据。`metadata`参数接受一个字典对象，其中包含要添加的元数据键值对。

以下是一个示例，演示如何将作者和标题信息添加到保存的图像元数据中：

```python
import matplotlib.pyplot as plt

# 创建Figure和Axes对象
fig, ax = plt.subplots()

# 绘制图形
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

# 创建元数据字典
metadata = {'Author': 'John Smith', 'Title': 'My Plot'}

# 保存Figure并添加元数据
fig.savefig('my_plot.png', metadata=metadata)
```

在上述示例中，我们创建了一个元数据字典`metadata`，其中包含作者和标题信息。然后，我们在调用`fig.savefig()`时通过`metadata`参数将元数据字典传递给函数。执行代码后，生成的图像文件"my_plot.png"将包含指定的元数据。

请注意，元数据的显示和访问取决于使用的图像查看器和操作系统。不同的查看器可能对元数据的支持和显示方式有所不同。

Matplotlib库所允许的图像格式包括以下几种：

1. PNG（Portable Network Graphics）：适用于Web显示和传输的无损压缩格式。

2. JPEG（Joint Photographic Experts Group）：适用于存储照片和彩色图像的有损压缩格式，适合高分辨率图像。

3. SVG（Scalable Vector Graphics）：矢量图形格式，适合无损放大缩小，用于制作可缩放的图标和矢量图形。

4. PDF（Portable Document Format）：可移植文档格式，适用于打印和文档交换。

5. EPS（Encapsulated PostScript）：用于存储矢量图形的页面描述语言，常用于印刷行业。

6. PGF（Portable Graphics Format）：用于TeX/LaTeX系统的矢量图形格式。

7. PS（PostScript）：页面描述语言，适用于打印和文档交换。

8. RAW（原始像素数据）：保存原始像素数据，需要额外的软件支持才能查看。

9. RGBA（原始像素数据，带透明通道）：保存原始像素数据及透明通道信息，需要额外的软件支持才能查看。

通常，你可以在`fig.savefig()`的`format`参数中指定所需的图像格式。例如，如果要将图像保存为PNG格式，可以使用如下代码：

```python
fig.savefig('my_plot.png', format='png')
```

如果不指定`format`参数，Matplotlib会根据文件名的扩展名自动选择图像格式。例如，上述代码中的文件名以".png"结尾，因此将自动保存为PNG格式。同理，如果文件名以".jpg"结尾，将自动保存为JPEG格式。

## Pandas

| 名称            | 介绍                                                         | 解决场景                       |
| --------------- | ------------------------------------------------------------ | ------------------------------ |
| `pandas`        | 是一个强大的数据处理和分析工具，提供了高效的数据结构和数据操作功能。 | 数据清洗、数据转换和数据分析   |
| `Series`        | 是一维带标签的数组，可以容纳不同类型的数据。                 | 时间序列数据、标量值的集合     |
| `DataFrame`     | 是二维表格数据结构，可以容纳不同类型的数据，并且每列都有标签。 | 结构化数据、表格数据、异构数据 |
| `Index`         | 是一种用于标识和访问`Series`和`DataFrame`中的行或列的结构。  | 数据定位、数据对齐、数据查询   |
| `Panel`         | 是三维数据结构，可以容纳多个`DataFrame`。                    | 多维数据集                     |
| `DatetimeIndex` | 是一种特殊类型的索引，用于处理时间序列数据。                 | 时间序列数据的索引和操作       |
| `GroupBy`       | 是一种分组操作，允许按照一个或多个键将数据分组，并对每个组进行聚合操作。 | 数据分组和聚合                 |
| `Merge`         | 是一种数据合并操作，可以将多个`DataFrame`按照一定的规则合并成一个。 | 数据合并和连接                 |
| `Reshape`       | 是一组用于重塑数据结构的函数，可以对数据进行透视、堆叠、旋转等操作。 | 数据重塑和转换                 |
| `IO Tools`      | 提供了读取和写入数据的功能，支持多种文件格式，如CSV、Excel、SQL等。 | 数据导入和导出                 |
| `Visualization` | 提供了绘制图表和可视化数据的功能，可以生成各种类型的图表。   | 数据可视化                     |



`pd.cut()`是Pandas库中的一个函数，用于将连续的数值数据划分为离散的区间。它可以根据指定的区间边界将数据进行分组，并为每个数据点分配一个对应的区间标签。

下面是`pd.cut()`函数的详细解释和一些常用参数的说明：

**函数签名:**
```python
pd.cut(x, bins, right=True, labels=None, retbins=False, precision=3, include_lowest=False, duplicates='raise')
```

**参数:**
- `x`：要划分的数值数据，可以是一维数组、Series或DataFrame的列。
- `bins`：指定的区间边界。有多种指定方式：
  - 整数：表示要将数据划分为多少个等宽区间。
  - 序列：表示自定义的区间边界，如`[0, 10, 20, 30]`表示区间为(0, 10]、(10, 20]、(20, 30]。
  - Pandas的`IntervalIndex`对象：可以使用`pd.interval_range()`函数生成。
- `right`：布尔值，表示区间的开闭方式。默认为True，表示右开左闭区间，即右边界不包含在区间内。
- `labels`：可选参数，用于指定区间的标签。可以是一个数组或布尔值False。如果是数组，则长度必须与划分后的区间数目相等；如果是False，则返回整数编码的区间。
- `retbins`：布尔值，表示是否返回划分后的区间边界。默认为False，只返回划分后的数据。
- `precision`：整数，表示显示区间边界的小数位数。
- `include_lowest`：布尔值，表示是否包含最低值所在的区间。默认为False，即不包含最低值所在的区间。
- `duplicates`：指定处理重复值的方式。可选值为'raise'、'drop'和'raise'。默认为'raise'，表示如果有重复的区间边界，会引发ValueError。

**返回值:**
- 如果`retbins`为False（默认情况），则返回一个Series，包含划分后的区间标签。
- 如果`retbins`为True，还会返回一个元组，包含划分后的区间标签和区间边界。

使用`pd.cut()`函数的示例代码如下：
```python
import pandas as pd

data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
bins = [0, 5, 10, 15, 20]

result = pd.cut(data, bins)
print(result)
```

输出结果：
```
[(0, 5], (0, 5], (5, 10], (5, 10], (5, 10], (10, 15], (10, 15], (15, 20], (15, 20], (15, 20]]
Categories (4, interval[int64]): [(0, 5] < (5, 10] < (10, 15] < (15, 20]]
```

以上示例中，`data`是要划分的数值数据，`bins`是指定的区间边界。`pd.cut(data, bins)`将数据划分为四个区间，并为每个数据点分配对应的区间标签。最后的输出结果中，每个数据点都被分配到了相应的区间，并且还显示了划分后的四个区间。