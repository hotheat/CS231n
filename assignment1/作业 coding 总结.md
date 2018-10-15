# cs231n.data_utils

## pickle 压缩保存文件

CIFAR-10 数据集的每个文件都是 1 个 Python Pickle 对象，读取和保存需要用到 Pickle 模块，cPickle 是用 C 语言实现的，拥有更快的速度。

### pickle 保存

以 `wb` 的形式写入文件，最终会多出 `pickle_example.pickle` 文件，列表和字典都能被保存，`dump` 第三个参数 `True` 能够提高压缩比。

```python
In [97]: from six.moves import cPickle as pickle

In [101]: with open('pickle_example.pickle', 'wb') as w:
     ...:     a_dict = {'da': 111, 2: [23,1,4], '23': {1:2,'d':'sad'}}
     ...:     pickle.dump(a_dict, w, True)
```

### pickle 提取文件

```python
In [93]: with open('data_batch_1', 'rb') as f:
    ...:     data = pickle.load(f, encoding='latin1')
    ...:

In [94]: list(data.keys())
Out[94]: ['batch_label', 'labels', 'data', 'filenames']

In [95]: len(data['data'])
Out[95]: 10000

In [96]: len(data['data'][0])
Out[96]: 3072
```

`load`  后以字典的形式保存。

## 查看 Python 版本

```python
In [102]: import platform

In [103]: platform.python_version_tuple()
Out[103]: ('3', '6', '2')
```

```python
>>> import platform
>>> platform.python_version_tuple()
('2', '7', '14')
```

## array.transpose()

高维数组的转置用 `array.transpose()`，一维或二维数组的转置一般用 `array.T`。

二维数组的 `shape` 返回的元组下标对应为 `0` 和 `1`，而 `array.transpose()` 接受的参数 `shape` 的元组索引，比如 `b.transpose((1, 0))` 是交换 `0` 和 `1` 轴，数值 `4` 对应的索引 `[1, 0]` 变成 `[0, 1]`。

```python
In [109]: b = np.arange(12).reshape(3, 4)

In [110]: b
Out[110]:
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

In [111]: b.transpose((1, 0))
Out[111]:
array([[ 0,  4,  8],
       [ 1,  5,  9],
       [ 2,  6, 10],
       [ 3,  7, 11]])
```

三维数组 `shape` 返回的下标索引为 `(0, 1, 2)`，`a.transpose((1, 2, 0))` 则是先交换 `0`和 `1`  轴，再交换 `1` 和 `2` 轴，元素 `9` 对应的下标由 `[0, 2, 1]` 变成 `[2, 1, 0]`。

```python
In [105]: a = np.arange(24).reshape(2, 3, 4)

In [106]: a
Out[106]:
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
                                  
In [108]: a.transpose((1, 2, 0))  
Out[108]:                         
array([[[ 0, 12],                 
        [ 1, 13],                 
        [ 2, 14],                 
        [ 3, 15]],                
                                  
       [[ 4, 16],                 
        [ 5, 17],                 
        [ 6, 18],                 
        [ 7, 19]],                
                                  
       [[ 8, 20],                 
        [ 9, 21],                 
        [10, 22],                 
        [11, 23]]])               

In [112]: a[0, 2, 1]
Out[112]: 9

In [113]: a.transpose((1, 2, 0))[2, 1, 0]
Out[113]: 9
```

`a.transpose((1, 2, 0))` 与 `a.swapaxes(1, 0).swapaxes(1, 2)` 等价

```python
In [116]: a.swapaxes(1, 0).swapaxes(1, 2)
Out[116]:
array([[[ 0, 12],
        [ 1, 13],
        [ 2, 14],
        [ 3, 15]],

       [[ 4, 16],
        [ 5, 17],
        [ 6, 18],
        [ 7, 19]],

       [[ 8, 20],
        [ 9, 21],
        [10, 22],
        [11, 23]]])
```

## `np.concatenate()`

在合并多个 `array` 或 `list` 中常用，`axis` 控制了合并的轴。比 `np.vstack()` 和 `np.hstack()` 更灵活。

```python
In [122]: a = [[1, 2, 3, 4]]

In [123]: b = [[5, 6, 7, 8]]

In [124]: np.concatenate((a, b), axis=0)
Out[124]:
array([[1, 2, 3, 4],
       [5, 6, 7, 8]])

In [125]: np.concatenate((a, b), axis=1)
Out[125]: array([[1, 2, 3, 4, 5, 6, 7, 8]])
```

# KNN.ipynb

## matplotlib

#### plt.rcParams

设置图像中默认字体大小

```python
plt.rcParams['figure.figsize'] = (10.0, 8.0)
```

#### plt.imshow()

接受一个三维数组，显示一张图片

```python
In [130]: X_train[1000].shape
Out[130]:
(32, 32, 3)

In [131]: plt.imshow(X_train[idx].astype('uint8'))
```

### `plt.errorbar()`

对散点图添加误差棒，第一个参数为 x 轴坐标，第二个参数为 y 轴坐标，`yerr` 为计算的误差，多为标准差。

```python
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
```

## ipython config

在 notebook 里 `import` 模块后，如果模块发生变化，自动重新加载模块。

```python
%load_ext autoreload
%autoreload 2
```

可以在 `~/.ipython/profile_default/ipython_config.py` 中添加以下代码，在打开 `ipython` 后自动设置

```python
c.InteractiveShellApp.extensions = ['autoreload']     
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
```

与 Python 解释器中的 `from importlib import reload; reload(module)`相同。

## Numpy

### np.flatnonzero()

返回扁平化矩阵后，非零元素的索引

```python
In [126]: a = np.arange(-5, 7).reshape(2, -1)

In [127]: a
Out[127]:
array([[-5, -4, -3, -2, -1,  0],
       [ 1,  2,  3,  4,  5,  6]])

In [128]: np.flatnonzero(a)
Out[128]: array([ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10, 11], dtype=int64)

In [129]: np.flatnonzero(a > 0)
Out[129]: array([ 6,  7,  8,  9, 10, 11], dtype=int64)
```

### `np.linalg.norm()`

numpy 下求范数的方法，默认是二范数，即向量的每个元素平方的加和再开方的形式。

```python
In [182]: a = np.array([3, 4])

In [183]: np.linalg.norm(a)
Out[183]: 5.0
```

`orf` 参数也可以指定其他范数，比如一范数 `ord=1` 和无穷范数 `ord = np.inf 或 (np.-inf)`

|      参数      | 说明                   | 计算方法                                     |
| :------------: | ---------------------- | :------------------------------------------- |
|      默认      | 二范数：$l_2$          | = $\sqrt{x_1^2 + x_2^2 + x_3^2} $            |
|    $ord=2$     | 二范数：$l_2$          | 同上                                         |
|   $ord = 1$    | 一范数：$l_1$          | = $\|x_1\| + \|x_2\| + ... + \|x_n\|$        |
| $ord = np.inf$ | 无穷范数：$l_\infty$   | = $max(|x_i|)$                               |
| $ord = 'fro'$  | F 范数，即矩阵的二范数 | = $\sqrt{x_{11}^2+x_{12}^2+... + x_{nn}^2} $ |

```python
In [187]: a
Out[187]: array([3, 4])

In [188]:  np.linalg.norm(a, ord=1)
Out[188]: 7.0

In [189]:  np.linalg.norm(a, ord=2)
Out[189]: 5.0

In [190]:  np.linalg.norm(a)
Out[190]: 5.0

In [194]: np.linalg.norm(a, ord=np.inf)
Out[194]: 4.0
    
In [191]: b = np.arange(4).reshape(2, 2)

In [192]:  np.linalg.norm(b, ord='fro')
Out[192]: 3.7416573867739413
```

`axis` 控制范数计算的轴

```python
In [201]: b
Out[201]:
array([[0, 1],
       [2, 3]])

In [199]: np.linalg.norm(b, ord=2, axis=1) # 行上计算二范数
Out[199]: array([1.        , 3.60555128])

In [200]: np.linalg.norm(b, ord=2, axis=0) # 列上计算二范数
Out[200]: array([2.        , 3.16227766])
```

`keepdims` 维持了范数计算后矩阵的维度特性

```python
In [202]: np.linalg.norm(b, ord=2, axis=1, keepdims=True)
Out[202]:
array([[1.        ],
       [3.60555128]])

In [203]: np.linalg.norm(b, ord=2, axis=1, keepdims=True).shape
Out[203]: (2, 1)

In [205]: np.linalg.norm(b, ord=2, axis=1).shape
Out[205]: (2,)

In [204]: b.shape
Out[204]: (2, 2)
```

### `np.split()`

切分数组，第二个可以切分数量或索引，`axis` 指定切分的轴，默认 `axis=0`

```python
In [211]: a = np.arange(24).reshape(4, -1)

In [212]: a
Out[212]:
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23]])

In [213]: np.split(a, 2, axis=1)
Out[213]:
[array([[ 0,  1,  2],
        [ 6,  7,  8],
        [12, 13, 14],
        [18, 19, 20]]), array([[ 3,  4,  5],
        [ 9, 10, 11],
        [15, 16, 17],
        [21, 22, 23]])]

In [214]: np.split(a, 2, axis=0)
Out[214]:
[array([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]]), array([[12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])]

In [215]: np.split(a, [1, 2])
Out[215]:
[array([[0, 1, 2, 3, 4, 5]]),
 array([[ 6,  7,  8,  9, 10, 11]]),
 array([[12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])]

In [216]: np.split(a, [1, 2], axis=1)
Out[216]:
[array([[ 0],
        [ 6],
        [12],
        [18]]), array([[ 1],
        [ 7],
        [13],
        [19]]), array([[ 2,  3,  4,  5],
        [ 8,  9, 10, 11],
        [14, 15, 16, 17],
        [20, 21, 22, 23]])]
```

### `np.array_split()`

仅对索引进行切分，参数与 `np.split()` 相同

```python
In [217]: np.array_split(a, 2, axis=0)
Out[217]:
[array([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]]), array([[12, 13, 14, 15, 16, 17],
        [18, 19, 20, 21, 22, 23]])]
```

# k_nearest_neighbor.py

## Numpy

### np.argsort()

返回数组升序排序后的索引

```python
In [136]: a = np.random.randint(10, size=5)

In [137]: a
Out[137]: array([6, 1, 3, 8, 5])

In [138]: np.argsort(a)
Out[138]: array([1, 2, 4, 0, 3], dtype=int64)

In [139]: np.argsort(-a) 
Out[139]: array([3, 0, 4, 2, 1], dtype=int64)
```

对于多维数组数组，`axis` 指定轴上的排序方式

```python
In [147]: b = np.random.randint(10, size=6).reshape(2, 3)

In [148]: b
Out[148]:
array([[9, 7, 8],
       [5, 6, 7]])

In [149]: np.argsort(b, axis=0) # 列上排序
Out[149]:
array([[1, 1, 1],
       [0, 0, 0]], dtype=int64)

In [150]: np.argsort(b, axis=1) # 行上排序
Out[150]:
array([[1, 2, 0],
       [0, 1, 2]], dtype=int64)
```

### np.unique()

返回数组的非重复元素

```python
In [158]: np.unique([1, 1, 2, 2, 3, 3])
Out[158]: array([1, 2, 3])
```

二维数组默认返回展平后的序列，如果有 `axis` 参数，则以某一轴为参考

```python
In [165]: a = np.array([[1, 1, 0], [1, 1, 0], [2, 2, 4]])

In [166]: a
Out[166]:
array([[1, 1, 0],
       [1, 1, 0],
       [2, 2, 4]])

In [167]: np.unique(a)
Out[167]: array([0, 1, 2, 4])

In [168]: np.unique(a, axis=1)
Out[168]:
array([[0, 1],
       [0, 1],
       [4, 2]])

In [169]: np.unique(a, axis=0)
Out[169]:
array([[1, 1, 0],
       [2, 2, 4]])
```

`return_index` 返回 `unique` 元素及 `unique` 元素第一次出现时的索引

```python
In [172]: u, idx = np.unique(a, return_index=True)

In [173]: u
Out[173]: array(['a', 'b', 'c'], dtype='<U1')

In [174]: idx
Out[174]: array([0, 1, 3], dtype=int64)
```

`return_inverse` 返回原数组在 `unique` 数组中的索引

```python
In [175]: u, inv = np.unique(a, return_inverse=True)

In [176]: u
Out[176]: array(['a', 'b', 'c'], dtype='<U1')

In [177]: inv
Out[177]: array([0, 1, 1, 2, 0], dtype=int64)

In [178]: u[inv]
Out[178]: array(['a', 'b', 'b', 'c', 'a'], dtype='<U1')
```

`return_counts` 返回 `unique` 元素的出现次数

```python
In [179]: u, cnt = np.unique(a, return_counts=True)

In [180]: cnt
Out[180]: array([2, 2, 1], dtype=int64)
```

### `np.argmax()`

返回最大数字的索引

```python
In [181]: np.argmax([1, 1, 2, 2, 3, 3])
Out[181]: 4
```

### `np.broadcast_to()`

将数组广播到新的形状。比如 shape 为 `(1, 2)` 的数组只能广播到 `(n, 2)` 维，没办法广播到 列数不等于 2 维的数组。

```python
In [206]: a
Out[206]: array([3, 4])
    
In [208]: np.broadcast_to(a, shape=(4, 2))
Out[208]:
array([[3, 4],
       [3, 4],
       [3, 4],
       [3, 4]])
```

