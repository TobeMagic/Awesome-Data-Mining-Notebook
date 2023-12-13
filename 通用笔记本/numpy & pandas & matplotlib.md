##  Numpy

Numpy 在数值运算上效率优于python内置的list, 所以熟练掌握是必要的。

Numpy提供两个核心的基本对象，**N维数组对象 Ndarry  和 通用函数对象 Ufunc**, （一个数据结构，一个操作的算法）下面是关于NumPy库的各个常用模块			

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

NumPy库的心脏，其为多维数组，具有矢量运算能力，且快速、节省空间。可对整组数据进行快速运算的标准数学函数、线性代数、随机数生成等功能。

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

#### 数组数据类型

NumPy的固定大小的数据类型，也称为dtype。这些数据类型具有特定的位数和取值范围，用于在NumPy数组中存储数据。下面是NumPy常见的基本数据类型及其描述和取值范围的表格：

| 数据类型   | 描述                       | 取值范围                                       |
| ---------- | -------------------------- | ---------------------------------------------- |
| bool       | 布尔值                     | True或False                                    |
| int8       | 有符号8位整数              | -128 到 127                                    |
| int16      | 有符号16位整数             | -32768 到 32767                                |
| int32      | 有符号32位整数             | -2147483648 到 2147483647                      |
| int64      | 有符号64位整数             | -9223372036854775808 到 9223372036854775807    |
| uint8      | 无符号8位整数              | 0 到 255                                       |
| uint16     | 无符号16位整数             | 0 到 65535                                     |
| uint32     | 无符号32位整数             | 0 到 4294967295                                |
| uint64     | 无符号64位整数             | 0 到 18446744073709551615                      |
| float16    | 半精度浮点数               | 约 -65500 到 65500                             |
| float32    | 单精度浮点数               | 约 -3.4 x 10^38 到 3.4 x 10^38                 |
| float64    | 双精度浮点数               | 约 -1.8 x 10^308 到 1.8 x 10^308               |
| complex64  | 复数，由两个32位浮点数组成 | 实部和虚部都约为 -3.4 x 10^38 到 3.4 x 10^38   |
| complex128 | 复数，由两个64位浮点数组成 | 实部和虚部都约为 -1.8 x 10^308 到 1.8 x 10^308 |

注意：这些取值范围是近似值，实际取值范围可能会略有不同。此外，NumPy还支持其他数据类型，如字符串类型和结构化类型，但它们不属于基本数据类型。上述表格列出的是NumPy中最常用的基本数据类型及其取值范围。

在数组类型转换中，可以通过对应的基础数据类型转换，如`np.float64()`

#### 创建Random 随机数组

NumPy的random模块提供了多种随机数生成函数，用于生成各种类型的随机数。以下是random模块的部分随机数生成函数，并附有说明和对应的模板代码：

| 名称                         | 说明                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| `numpy.random.rand()`        | 生成指定形状的[0,1)之间均匀分布的随机数。                    |
| `numpy.random.randn()`       | 生成指定形状的符合**标准正态分布**的随机数（平均值为0，标准差为1）。 |
| `numpy.random.randint()`     | 生成指定范围内的随机整数。                                   |
| `numpy.random.random()`      | 生成[0,1)之间均匀分布的随机数，形状由参数指定。（与rand()主要区别在于参数传递方式，对于生成的随机样本没有实质性的区别，两者都可以生成满足要求的随机数。 |
| `numpy.random.uniform()`     | 生成指定范围内**均匀分布**的随机数。                         |
| `numpy.random.normal()`      | 生成指定均值和标准差的正态分布随机数。                       |
| `numpy.random.seed()`        | 设置随机数生成的种子，以便**重现随机数序列**。               |
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

以上并不是全部还有许多其他，可以查看官方文档查看，如二项分布，伽马分布等。

#### 数组索引

当使用NumPy库进行数据处理和分析时，经常需要使用数据索引来访问和操作数组中的元素。NumPy提供了多种方法来进行数据索引，以满足不同的需求。下面是一些常用的NumPy数据索引方法及其相应的说明和模板代码：

| 名称                     | 说明                                                         | 模板代码                         |
| ------------------------ | ------------------------------------------------------------ | -------------------------------- |
| 通过整数索引             | 使用整数索引来访问数组中的元素。可以使用单个整数、切片或整数数组作为索引。 | `array[index]`                   |
| 通过布尔索引             | 使用布尔数组作为索引来选择满足条件的元素。**布尔数组的长度必须与被索引的数组的长度相同。** | `array[bool_array]`              |
| 通过切片索引             | 使用切片来访问数组中的一个范围。切片由起始索引、结束索引和步长组成。 | `array[start:end:step]`          |
| 通过花式索引             | 使用**整数数组**作为索引来选择数组中的特定元素。可以使用一维或多维整数数组进行索引。 | `array[index_array]`             |
| 通过整数和切片混合索引   | 可以同时使用整数和切片进行数组索引。这样可以选择数组的特定部分。 | `array[index, slice]`            |
| 通过条件索引             | 使用条件表达式来选择满足条件的元素。可以使用`np.where()`函数来实现条件索引。 | `array[np.where(condition)]`     |
| 多维数组索引             | 对于多维数组，可以使用逗号分隔的索引元组来访问特定的元素。   | `array[row_index, column_index]` |
| 布尔索引和整数索引的组合 | 可以将布尔索引和整数索引组合使用，以实现更复杂的索引操作。   | `array[bool_array, index]`       |

请注意，上述模板代码中的`array`表示要索引的NumPy数组，`index`表示索引的部分或全部，`bool_array`表示布尔索引数组，`start`表示切片的起始索引，`end`表示切片的结束索引，`step`表示切片的步长，`condition`表示条件表达式，`row_index`和`column_index`表示多维数组的行索引和列索引。

#### 数组变换（形状变换、数组组合&分割）

下面是NumPy中一些常用的数组形状变换函数的详细说明、模板代码和参数注释。

| 函数名称         | 说明                                                         | 模板代码                                                     |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `np.reshape`     | 改变数组的形状，不改变数据本身，要求新形状的总元素数量必须与原始数组的总元素数量相同。不会进行元素的重复或删除。 | `np.reshape(a, newshape, order='C')`<br>参数：<br>`a`：要改变形状的数组<br>`newshape`：新的形状，可以是整数或者元组<br>`order`：可选参数，指定元素在内存中的存储顺序，默认为'C' (按行) |
| `np.resize`      | 改变数组的形状，可以改变数据本身，**在必要时重复数组的元素以填充新形状。** | `np.resize(a, new_shape)`<br>参数：<br>`a`：要改变形状的数组<br>`new_shape`：新的形状，可以是整数或者元组 |
| `np.transpose`   | 将数组的维度进行转置                                         | `np.transpose(a, axes=None)`<br>参数：<br>`a`：要转置的数组<br>`axes`：可选参数，指定转置的轴顺序 |
| `np.swapaxes`    | 交换数组的两个轴                                             | `np.swapaxes(a, axis1, axis2)`<br>参数：<br>`a`：要交换轴的数组<br>`axis1`：第一个轴的索引<br>`axis2`：第二个轴的索引 |
| `np.flatten`     | 展平数组，将多维数组转换为一维数组，**与reavel区别在于可选择横向或纵向展平** | `np.flatten(a, order='C')`<br>参数：<br>`a`：要转换的数组<br>`order`：可选参数，指定元素在内存中的存储顺序，默认为'C' (按行) |
| `np.ravel`       | 展平数组，将多维数组转换为一维数组                           | `np.ravel(a, order='C')`<br>参数：<br>`a`：要转换的数组<br>`order`：可选参数，指定元素在内存中的存储顺序，默认为'C' (按行) |
| `np.squeeze`     | 从数组的形状中删除单维度条目                                 | `np.squeeze(a, axis=None)`<br>参数：<br>`a`：要删除单维度条目的数组<br>`axis`：可选参数，指定要删除的轴 |
| `np.expand_dims` | 在数组形状中插入新的轴                                       | `np.expand_dims(a, axis)`<br>参数：<br>`a`：要插入新轴的数组<br>`axis`：要插入的位置 |

以下是NumPy中常用的数组组合和分割函数的详细解释和模板代码：

**数组组合函数**

| 名称                | 说明                      | 模板代码                              |
| ------------------- | ------------------------- | ------------------------------------- |
| `numpy.concatenate` | 沿指定轴连接数组序列      | `numpy.concatenate((arrays, axis=0))` |
| `numpy.stack`       | 沿新轴连接数组序列        | `numpy.stack(arrays, axis=0)`         |
| `numpy.hstack`      | 水平（按列）堆叠数组序列  | `numpy.hstack(tup)`                   |
| `numpy.vstack`      | 垂直（按行）堆叠数组序列  | `numpy.vstack(tup)`                   |
| `numpy.dstack`      | 深度（按Z轴）堆叠数组序列 | `numpy.dstack(tup)`                   |

 **数组分割函数**

| 名称                | 说明                                | 模板代码                                              |
| ------------------- | ----------------------------------- | ----------------------------------------------------- |
| `numpy.split`       | 将数组沿指定轴分割为多个子数组      | `numpy.split(ary, indices_or_sections, axis=0)`       |
| `numpy.array_split` | 将数组沿指定轴不等分割为多个子数组  | `numpy.array_split(ary, indices_or_sections, axis=0)` |
| `numpy.hsplit`      | 将数组水平（按列）分割为多个子数组  | `numpy.hsplit(ary, indices_or_sections)`              |
| `numpy.vsplit`      | 将数组垂直（按行）分割为多个子数组  | `numpy.vsplit(ary, indices_or_sections)`              |
| `numpy.dsplit`      | 将数组深度（按Z轴）分割为多个子数组 | `numpy.dsplit(ary, indices_or_sections)`              |

在上述模板代码中，需要注意的是：

- `arrays`：要组合的数组序列。
- `axis`：指定连接的轴。默认为0，表示沿第一个轴连接。
- `tup`：要堆叠的数组序列，以元组形式提供。
- `ary`：要分割的数组。
- `indices_or_sections`：指定分割点的索引列表或分割的段数。
- `axis`：指定分割的轴。

### Matrix & Ufunc

#### Matrix 属性及其创建方法

在Numpy中，矩阵是ndarray的子类，数组和矩阵有着非常重要的区别，矩阵本身是继承与Numpy对象的二维数组对象，其是二维的

| **属性** | **说明**                        | 函数       |
| -------- | ------------------------------- | ---------- |
| **T**    | 返回自身的转置                  | `matrix.T` |
| **H**    | 返回自身的共轭转置              | `matrix.H` |
| **I**    | 返回自身的逆矩阵                | `matrix.I` |
| **A**    | 返回自身数据的2维数组的一个视图 | `matrix.A` |

以下则是创建matrix的函数，
| 函数名称  | 详细说明 | 模板代码                                                     |
| --------- | -------- | ------------------------------------------------------------ |
| np.mat    | 创建矩阵 | `np.mat(object, dtype=None)`<br>参数：<br>- `object`：输入的数组或字符串<br>- `dtype`：可选，指定所创建矩阵的数据类型 |
| np.matrix | 创建矩阵 | `np.matrix(data, dtype=None, copy=True)`<br>参数：<br>- `data`：输入的数组，列表或字符串<br>- `dtype`：可选，指定所创建矩阵的数据类型<br>- `copy`：可选，指定是否复制输入数据 |
| np.bmat   | 拼接矩阵 | `np.bmat(obj, ldict=None, gdict=None)`<br>参数：<br>- `obj`：输入的矩阵字符串<br>- `ldict`：可选，局部命名空间字典<br>- `gdict`：可选，全局命名空间字典 |

其中mat函数在输入matrix和ndarray对象，不会为其创建副本，matrix有参数copy可选（建议选择matrix)，不会影响原有数组，bmat（block matrix) 可以合并矩阵

#### Ufunc 函数运算

针对于数组进行操作，并将数组进行输出，不需要对数组每一个元素都操作，所以**运用Ufunc函数比math库的函数要快很多。**

通用函数（universal function），是一种能够对数组中所有元素进行操作的函数。

- 四则运算：加（+）、减（-）、乘（*）、除（/）、幂（**）。数组间的四则运算表示对每个数组中的元素分别进行四则运算，所以形状必须相同。
- 比较运算：>、<、==、>=、<=、!=。比较运算返回的结果是一个布尔数组，每个元素为每个数组对应元素的比较结果。
- 逻辑运算：np.any函数表示逻辑“or”，np.all函数表示逻辑“and”。运算结果返回布尔值。

#### Ufunc 广播机制（Broadcasting）

ufunc函数的广播机制允许对不同形状的数组进行操作，使得它们能够在一起进行运算，而不需要显式地进行数组形状的转换或复制。

广播机制的原则如下：

1. 维度较少的数组会在其缺失的维度上进行扩展，以匹配维度较多的数组。这意味着维度较少的数组会被扩展为与维度较多的数组具有相同的维度数。
2. 如果两个数组的形状在某个维度上不一致，而其中一个数组的形状在该维度上为1，那么可以将该数组沿着该维度进行复制，以使得两个数组的形状在该维度上一致。
3. 如果两个数组的形状在某个维度上既不相等，也不为1，那么广播操作将会失败，抛出一个异常。

下面是广播机制的应用示例，以说明以上原则：

```python
import numpy as np

# 创建两个数组
a = np.array([1, 2, 3])
b = np.array([[4], [5], [6]])

# 形状不一致的数组进行相加
c = a + b

print(c)
```

输出结果为：

```
[[5 6 7]
 [6 7 8]
 [7 8 9]]
```

在这个例子中，数组`a`的形状是`(3,)`，数组`b`的形状是`(3, 1)`。根据原则1，数组`a`会在缺失的维度上进行扩展，扩展后的形状变为`(1, 3)`。然后根据原则2，数组`b`会在第二个维度上进行复制，复制后的形状也变为`(3, 3)`。最终，两个形状一致的数组进行逐元素相加，得到结果数组`c`。

需要注意的是，广播机制在执行运算时并不会**实际复制或扩展数组**，它只是在逻辑上扩展数组的形状，以使得运算能够进行。这样可以避免不必要的内存消耗，提高运算效率。

`numpy.broadcast_arrays`  将输入数组广播到共同形状  `numpy.broadcast_arrays(*args, **kwargs)`

### 统计分析

#### 读/写数据（二进制文件 & 文本文件）

当涉及到使用 NumPy 读写文件时，主要使用的是 `numpy.save()`、`numpy.load()` 和 `numpy.savetxt()`、`numpy.loadtxt()` 函数。下面是对这些函数的详细说明和对应的模板代码：

| 函数名称             | 说明                                                         | 代码模板                                                     |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `numpy.save()`       | 将数组保存到以 `.npy` 扩展名结尾的二进制文件中               | `numpy.save(file, arr)`<br>参数：<br>`file`：要保存到的文件名，可以是字符串或类文件对象。<br>`arr`：要保存的数组。 |
| `numpy.load()`       | 从二进制 `.npy` 文件中加载数组                               | `numpy.load(file)`<br>参数：<br>`file`：要加载的文件名，可以是字符串或类文件对象。 |
| `numpy.savetxt()`    | 将数组保存到文本文件中                                       | `numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)`<br>参数：<br>`fname`：要保存到的文件名，可以是字符串或类文件对象。<br>`X`：要保存的数组。<br>`fmt`：写入文件时的格式，默认为 `%.18e`（科学计数法）。<br>`delimiter`：用于分隔值的字符串，默认为空格。<br>`newline`：用于分隔行的字符串，默认为换行符。<br>`header`：文件的头部字符串，默认为空字符串。<br>`footer`：文件的尾部字符串，默认为空字符串。<br>`comments`：注释标记字符串，默认为 `'#'`。<br>`encoding`：编码格式，默认为 `None`。 |
| `numpy.loadtxt()`    | 从文本文件中加载数组                                         | `numpy.loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)`<br>参数：<br>`fname`：要加载的文件名，可以是字符串或类文件对象。<br>`dtype`：返回数组的数据类型，默认为 `float`。<br>`comments`：注释标记字符串，默认为 `'#'`。<br>`delimiter`：用于分隔值的字符串，默认为 `None`，表示使用连续空格作为分隔符。<br>`converters`：用于将数据转换为合适格式的函数，默认为 `None`。<br>`skiprows`：要跳过的起始行数，默认为 `0`。<br>`usecols`：要加载的列索引，默认为 `None`，表示加载所有列。<br>`unpack`：如果为 `True`，返回每列作为独立数组的结果，默认为 `False`。<br>`ndmin`：返回数组的最小维数，默认为 `0`。<br>`encoding`：编码格式，默认为 `'bytes'`。<br>`max_rows`：要加载的最大行数，默认为 `None`，表示加载所有行。 |
| `numpy.savez()`      | 将多个数组保存到以 `.npz` 扩展名结尾的压缩文件中             | `numpy.savez(file, *args, **kwds)`<br>参数：<br>`file`：要保存到的文件名，可以是字符串或类文件对象。<br>`*args`：要保存的数组，可以是多个。<br>`**kwds`：关键字参数，用于指定数组的名称。 |
| `numpy.genfromtxt()` | 从文本文件中加载数组，**支持结构化数组和缺失值**处理和数据类型推断 | `numpy.genfromtxt(fname, dtype=float, comments='#', delimiter=None, skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')`<br>参数：<br>`fname`：要加载的文件名，可以是字符串或类文件对象。<br>`dtype`：返回数组的数据类型，默认为 `float`。<br>`comments`：注释标记字符串，默认为 `'#'`。<br>`delimiter`：用于分隔值的字符串，默认为 `None`，表示使用连续空格作为分隔符。<br>`skip_header`：要跳过的头部行数，默认为 `0`。<br>`skip_footer`：要跳过的尾部行数，默认为 `0`。<br>`converters`：用于将数据转换为合适格式的函数，默认为 `None`。<br>`missing_values`：用于标识缺失值的字符串或序列，默认为 `None`。<br>`filling_values`：用于替换缺失值的字符串或序列，默认为 `None`。<br>`usecols`：要加载的列索引，默认为 `None`，表示加载所有列。<br>`names`：返回数组的字段名称，默认为 `None`，表示不返回字段名称。<br>`excludelist`：要排除的列名称列表，默认为 `None`。<br>`deletechars`：要从列名称中删除的字符，默认为 `None`。<br>`replace_space`：在列名称中替换空格的字符，默认为 `'_'`。<br>`autostrip`：如果为 `True`，去除数据的前导和尾随空格，默认为 `False`。<br>`case_sensitive`：列名称是否区分大小写，默认为 `True`。<br>`defaultfmt`：用于生成未命名字段名称的格式字符串，默认为 `'f%i'`。<br>`unpack`：如果为 `True`，返回每列作为独立数组的结果，默认为 `None`。<br>`usemask`：如果为 `True`，返回带有掩码数组的结果，默认为 `False`。<br>`loose`：如果为 `True`，宽松处理列名称的匹配，默认为 `True`。<br>`invalid_raise`：如果为 `True`，遇到无效数据时引发异常，默认为 `True`。<br>`max_rows`：要加载的最大行数，默认为 `None`，表示加载所有行。<br>`encoding`：编码格式，默认为 `'bytes'`。 |



#### 排序

下面是NumPy中常用的排序函数的详细说明，包括函数名称、说明和模板代码：

| 函数                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `numpy.sort(a, axis=-1, kind=None, order=None)`              | 返回数组的排序副本。（直接排序），array数组对象本身也有sort方法，不返回值 |
| `numpy.argsort(a, axis=-1, kind=None, order=None)`           | 返回数组的索引，使得按指定轴的排序顺序重构数组。间接排序     |
| `numpy.lexsort(keys, axis=-1)`                               | 使用键序列执行间接排序。(多个键)                             |
| `numpy.searchsorted(a, v, side='left', sorter=None)`         | 查找应该插入元素以维持排序顺序的索引。                       |
| `numpy.partition(a, kth, axis=-1, kind='introselect', order=None)` | 返回数组的分区副本，其中第 `kth` 元素在排序后的位置。        |
| `numpy.argpartition(a, kth, axis=-1, kind='introselect', order=None)` | 返回数组的索引，以使第 `kth` 元素在排序后的位置。            |

以上是NumPy中常用的排序函数的详细说明和模板代码。你可以根据需要选择适合的函数并使用相应的模板代码。注意，这里的模板代码只是为了给出函数的参数注释，并不包含完整的使用示例。需要根据具体的情况进行调整和使用。

#### 去重或重复数据

以下是一些常用的 NumPy 重复函数及其模板代码。

| 名称             | 说明                                           | 模板代码                                                     |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| `numpy.unique()` | 返回输入数组中的**唯一元素数组，去除重复值**。 | `numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)` |
| `numpy.tile`     | 重复**整个数组**                               | `numpy.tile(A, reps)`                                        |
| `numpy.repeat`   | 重复**数组中的元素**                           | `numpy.repeat(a, repeats, axis=None)`                        |

这些函数提供了不同的方法来重复数据或操作数组的形状。你可以根据需要选择适当的函数，并根据模板代码中的参数注释进行调用。

#### 统计函数

| **函数**              | **说明**                          |
| --------------------- | --------------------------------- |
| `numpy.mean`          | 计算数组的平均值                  |
| `numpy.median`        | 计算数组的中位数                  |
| `numpy.std`           | 计算数组的标准差                  |
| `numpy.var`           | 计算数组的方差                    |
| `numpy.min`           | 返回数组的最小值                  |
| `numpy.max`           | 返回数组的最大值                  |
| `numpy.cumsum`        | 计算数组元素的累积和              |
| `numpy.prod`          | 计算数组元素的乘积                |
| `numpy.cumprod`       | 计算数组元素的累积乘积            |
| `numpy.diff`          | 计算数组元素之间的差值            |
| `numpy.gradient`      | 计算数组元素的梯度                |
| `numpy.average`       | 计算数组的加权平均值              |
| `numpy.median`        | 计算数组的中位数                  |
| `numpy.percentile`    | 计算数组的百分位数                |
| `numpy.cov`           | 计算数组的协方差矩阵              |
| `numpy.corrcoef`      | 计算数组的相关系数矩阵            |
| `numpy.nanmean`       | 计算数组中忽略NaN值的平均值       |
| `numpy.nanmedian`     | 计算数组中忽略NaN值的中位数       |
| `numpy.nanpercentile` | 计算数组中忽略NaN值的百分位数     |
| `numpy.nanvar`        | 计算数组中忽略NaN值的方差         |
| `numpy.nanstd`        | 计算数组中忽略NaN值的标准差       |
| `numpy.nanmin`        | 返回数组中的最小值，忽略NaN值     |
| `numpy.nanmax`        | 返回数组中的最大值，忽略NaN值     |
| `numpy.nanargmin`     | 返回数组中最小值的索引，忽略NaN值 |
| `numpy.nanargmax`     | 返回数组中最大值的索引，忽略NaN值 |

## Matplotlib

### 绘图Pipeline

Matplotlib是Python中最常用的可视化工具之一，可以非常方便地创建海量类型的2D图表和一些基本的3D图表，可根据数据集（DataFrame，Series）自行定义x,y轴，绘制图形（线形图，柱状图，直方图，密度图，散布图等等），能够满足大部分需要。

Matplotlib最早是为了可视化癫痫病人的脑皮层电图相关的信号而研发，因为在函数的设计上参考了MATLAB，所以叫做Matplotlib。其中最基础的模块是pyplot，以下讲解均围绕该模块。

官方文档： https://matplotlib.org/

![image-20231204170322913](numpy%20&%20pandas%20&%20matplotlib.assets/image-20231204170322913.png)

**1. 创建画布与创建子图**
第一部分主要作用是构建出一张空白的画布，并可以选择是否将整个画布划分为**多个部分**，方便在同一幅图上绘制多个图形的情况。最简单的绘图可以省略第一部分，而后直接在默认的画布上进行图形绘制。

**2. 添加画布内容**
第二部分是绘图的主体部分。其中添加标题，坐标轴名称，绘制图形等步骤是并列的，没有先后顺序，可以先绘制图形，也可以先添加各类标签。但是添加图例一定要在绘制图形之后。以下是图的各个参数。

<img src="numpy%20&%20pandas%20&%20matplotlib.assets/image-20231204192630593.png" alt="image-20231204192630593" style="zoom:50%;" />

**3. 保存与展示图形**
第三部分主要用于保存和显示图形。

| **函数名称** | **函数作用**                                             |
| ------------ | -------------------------------------------------------- |
| plt.savafig  | 保存绘制的图片，可以指定图片的分辨率、边缘的颜色等参数。 |
| plt.show     | 在本机显示图形。                                         |

`fig.savefig()`是Matplotlib库中Figure对象的方法，用于将Figure保存为图像文件。它的主要功能是将绘制的图形保存为常见的图像格式（如PNG、JPEG、SVG等），以便后续使用或共享。

以下是`fig.savefig()`的用法详解：

```python
fig.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None)
"""
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
"""
```

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

### pyplot的动态rc参数

### 常用的绘制的图像全类型

请你详解常用的绘制的图像全类型，并以markdown表格列出，第一列是名称，第二列是说明， 第三列给出对应使用场景

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

### IO Tools

读取方式（文本，exls, database)

#### 数据库

#### 文本文件 csv

#### excel

### DataFrame

#### 常用属性

#### CRUD

#### 描述性统计

##### 类别型

##### 数值型

### DatetimeIndex

### 分组聚合（组内操作）

### 透视表与交叉表

### * 数据清洗

#### 数据归约

##### 特征编码（哑变量 & 独热编码 & 标签编码）

> - 我们首先将类别型数据分为两个类

1.  **定类型变量**
   定类类型就是离散数据，不排序，没有逻辑关系.
   当某特征具有k个属性值，那么：
   a.  哑变量（Dummy variable，也称为虚拟变量或指示变量）—— 具有k-1个二进制特征，基准类别将被忽略, 若基准类别选择不合理，仍**存在共线性**(高度相关线性），**建议众数的类别为基准类别**。
   b.  独热编码——具有k个特征二进制特征。

   > 二者区别：
   >
   > 1，如果有N个特征，已知前N-1个特征的特征值之后，第N个特征的特征值也就知道了，因此**独热编码有冗余，虚拟变量没有冗余**；2**，**独热编码可以直接从激活状态看出所对应的类别，而虚拟变量需要进行推论，因此**独热编码比较直观，虚拟变量没有那么直观**。

2.  **定序型变量**
   标签编码——用自定义的数字对原始特征进行打标签，**适用于有序**的分类变量。

> - 编码的意义

不用对变量归一化，加速参数的更新速度；使得一个很大权值管理一个特征，拆分成了许多小的权值管理这个特征多个表示，降低了特征值扰动对模型的影响，模型具有更好的鲁棒性，将数据转换成可训练的格式

> - 编码优缺点

1. 定类变量
   异常数据具有很强的鲁棒性；离散化之后可以进行特征交叉，引入非线性，提高模型表达能力。
   一个特征被分割多份，**损失部分统计信息**，学习效果差。
   a. 哑变量：从k-1个变量推论第k个类别，不太直观，但不冗余；
   b. 独热编码：从k个变量看出所有变量类别，比较直观，但特征冗余；独热特征高度相关，易导致共线，自变量之间存在高度相关关系，从而使模型参数估计不准确；

2.  定序变量

   标签编码：可以自定义量化数字，但数值本身没有含义，仅用作排序；可解释性比较差，比如[‘大学’,‘高中’,‘初中’,‘小学’] —>[1，2，3，4]，’大学‘和’小学相隔的距离更远。‘

> - 用法

1. 定类变量
   对数值大小较敏感的模型，如LR SVM

   > 截距（intercept）是线性模型中的一个参数，它表示当所有自变量（或哑变量）都为零时，因变量的预期平均值。在线性回归模型中，截距是一个常数，它对应于自变量取值为零时的因变量取值。

   a. 独热编码的截距表示均值，回归系数是与均值之间的差距；而虚拟变量的截距是参照类的值，回归系数表示与参照类的差距。

   b. 在线性模型中，如果有截距项，使用哑变量编码可以处理多余的自由度，因为多余的自由度可以被统摄到截距项中。这意味着，当使用哑变量编码时，只需要使用n-1个哑变量来表示n个类别，其中n是类别的数量。剩下的一个类别可以被认为是基准类别，截距项对应于基准类别的取值。

   c. 如果线性模型有截距项，那么请使用虚拟变量；如果线性模型无截距项，那么使用独热编码。此外，如果线性模型有截距项，但在加了正则化之后，也可以使用独热编码，因为这相当于约束了 w 的解的空间。

   d. 如果线性模型没有截距项，而且使用独热编码，那么每个类别都将有一个独立的变量。这种情况下，**模型将完全依赖于这些变量的取值来预测因变量，而没有一个基准类别**。这种编码方式通常用于特定需求的模型，例如需要明确控制每个类别的影响。

   > **如果使用正则化，那么推荐使用独热编码**，因为regularization能够处理多余的自由度，使用正则化手段去约束参数，同时类别型变量的各个值的地位是对等的。**如果不使用正则化，那么使用虚拟变量**（这样多余的自由度都被统摄到截距项intercept里去了）。

2. 定序型变量
   既分类又排序，自定义的数字顺序可以不破坏原有逻辑，并与这个逻辑相对应。对数值大小不敏感的模型（如树模型）不建议使用one-hotencoding

**选择建议**：

算法上：最好是选择正则化 + one-hot，哑变量编码也可以使用，不过最好选择前者。

对于树模型，不推荐使用定类编码，因为样本切分不均衡时，增益效果甚微(如较小的那个拆分样本集，它占总样本的比例太小。无论增益多大，乘以该比例之后几乎可以忽略)；

实现上：

哑变量在pandas的get_dummy方法，one-hot在`from sklearn.preprocessing import OneHotEncoder`

pandas机制问题需要在内存中把数据集都读入进来，要是数据量大的话，太消耗资源，one-hot可以读数组，因此大规模数据集很方便。

模板代码

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
df = pd.DataFrame([  
            ['green' , 'A'],   
            ['red'   , 'B'],   
            ['blue'  , 'A']])  
df.columns = ['color',  'class'] 
#one-hot编码
onehot=OneHotEncoder(sparse=False)
data=onehot.fit_transform(df[['color']])
print("one-hot编码结果如下：")
print(data)
#哑变量编码
#pd.get_dummies()方法即可以用于产生One-Hot编码，也可以用于产生哑变量编码
#当drop_first=True时为哑变量编码，当为False时为One-Hot编码
#哑变量编码是将One-Hot编码的第一列结果去掉即可。
data=pd.get_dummies(df['color'],drop_first=True)
print("哑变量编码结果如下：")
print(data)
```

下面是`pd.get_dummies()`函数的参数及其说明：

| 参数名       | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| `data`       | 必需参数，指定要进行独热编码的数据，可以是Series、DataFrame或者数组。 |
| `prefix`     | 字符串或者字符串列表，用于添加到生成的列名前面作为前缀。如果传入的`data`是DataFrame，则`prefix`可以是与`data`的列对应的字符串列表，或者一个字符串，将为所有列添加相同的前缀。默认值为`None`，表示不添加前缀。 |
| `prefix_sep` | 用于在前缀和原始列名之间添加的分隔符字符串。默认值为`'_'`。  |
| `columns`    | 列表或者字符串，用于指定需要进行独热编码的列。如果传入的`data`是DataFrame，则`columns`可以是DataFrame的列名列表或者一个列名。如果传入的`data`是Series，则`columns`参数无效，因为Series只有一个列。默认值为`None`，表示对所有列进行独热编码。 |
| `drop_first` | 布尔值，用于指定是否删除每个特征的第一个类别。如果设置为`True`，则将会删除每个特征的第一个类别，并且返回的结果将包含`n-1`列，其中`n`是原始特征的类别数。默认值为`False`，表示保留所有类别。 |
| `dtype`      | 结果的数据类型。默认为`np.uint8`。                           |
| `dummy_na`   | 布尔值，用于指定是否为缺失值创建一个哑变量列。如果设置为`True`，则将会为缺失值创建一个名为`prefix_nan`的列，并将其视为一个类别。默认值为`False`，表示不创建缺失值列。 |



参考文章：

https://blog.51cto.com/u_16099322/8207171

https://www.cnblogs.com/HuZihu/p/9692554.html

https://blog.csdn.net/yeshang_lady/article/details/103940513

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