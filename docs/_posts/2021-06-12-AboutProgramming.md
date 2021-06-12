---
author: rym
catagory: blog
excerpt: 关于写程序的一些经验总结。这里没有太详细的技术细节，那些东西有了关键词，在网上搜索会很容易。
---

# 写程序

我不会讲太详细的技术细节，那些东西有了关键词，在网上搜索会很容易。

就神经网络而言，它的程序结构并不复杂。神经网络需要的是程序能够快速修改、清楚记录大量实验，以及用简短的脚本清洗混乱的输入数据。大多数功能在框架里都有了，照着例子改就是了。

## 凡例

*关键词：git, git fork*
表示搜索以下关键词， git 和 git fork

> Fluent Python, Ch21
表示引用材料

## 程序可以多复杂

关于复杂的程序，我举个例子，用python：

> Fluent Python, Ch21 Class metaprogramming : the evaluation time exercise

使用两种方式运行python脚本，并预测它们的输出。这个例子展示了python脚本的两种执行模式，以及我们可以用多么丰富的手段控制程序的执行顺序。

程序中的print，依书写时的出现顺序进行了编号(1,2... ; 100,200,...).

python是解释型语言，这里展示的是evaluation time的执行顺序。像cpp是有编译期的，典型的技术如函数模板。

*Scenario #1*
The module evaltime.py is imported interactively in the Python console:

````python
>>> import evaltime
````

*Scenario #2*
The module evaltime.py is run from the command shell:

````sh
$ python3 evaltime.py
````

*evaltime.py*

```` python
from evalsupport import deco_alpha
print('<[1]> evaltime module start')
class ClassOne():
    print('<[2]> ClassOne body')
    def __init__(self):
        print('<[3]> ClassOne.__init__')
    def __del__(self):
        print('<[4]> ClassOne.__del__')
    def method_x(self):
        print('<[5]> ClassOne.method_x')
    class ClassTwo(object):
        print('<[6]> ClassTwo body')
@deco_alpha
class ClassThree():
    print('<[7]> ClassThree body')
    def method_y(self):
        print('<[8]> ClassThree.method_y')

class ClassFour(ClassThree):
    print('<[9]> ClassFour body')
    def method_y(self):
        print('<[10]> ClassFour.method_y')

if __name__ == '__main__':
    print('<[11]> ClassOne tests', 30 * '.')
    one = ClassOne()
    one.method_x()
    print('<[12]> ClassThree tests', 30 * '.')
    three = ClassThree()
    three.method_y()
    print('<[13]> ClassFour tests', 30 * '.')
    four = ClassFour()
    four.method_y()
print('<[14]> evaltime module end')
````

*evalsupport.py*

````python
print('<[100]> evalsupport module start')
def deco_alpha(cls):
    print('<[200]> deco_alpha')
    def inner_1(self):
        print('<[300]> deco_alpha:inner_1')
    cls.method_y = inner_1
    return cls
# BEGIN META_ALEPH
class MetaAleph(type):
    print('<[400]> MetaAleph body')
    def __init__(cls, name, bases, dic):
        print('<[500]> MetaAleph.__init__')
        def inner_2(self):
            print('<[600]> MetaAleph.__init__:inner_2')
        cls.method_z = inner_2
# END META_ALEPH
print('<[700]> evalsupport module end')
````

答案：

Solution for Scenario #1: importing evaltime in the Python console.

````sh
>>> import evaltime
<[100]> evalsupport module start
<[400]> MetaAleph body
<[700]> evalsupport module end
<[1]> evaltime module start
<[2]> ClassOne body
<[6]> ClassTwo body
<[7]> ClassThree body
<[200]> deco_alpha
<[9]> ClassFour body
<[14]> evaltime module end
````

Solution for Scenario #2: Running evaltime.py from the shell.

````sh
$ python3 evaltime.py
<[100]> evalsupport module start
<[400]> MetaAleph body
<[700]> evalsupport module end
<[1]> evaltime module start
<[2]> ClassOne body
<[6]> ClassTwo body
<[7]> ClassThree body
<[200]> deco_alpha
<[9]> ClassFour body
<[11]> ClassOne tests ..............................
<[3]> ClassOne.__init__
<[5]> ClassOne.method_x
<[12]> ClassThree tests ..............................
<[300]> deco_alpha:inner_1
<[13]> ClassFour tests ..............................
<[10]> ClassFour.method_y
````

在引入协程、异步等概念后，执行顺序会更加复杂。

cpp的例子就不举了，有兴趣的可以看看《泛型编程（侯杰）》，是个理解模板的极佳案例。

## 如何设计程序


设计程序有很多方式，很多理念。我从一个单线程的的、单文件的、一次性的程序开始，它天然地因为功能而可以被看成一系列“代码片段”。那么，程序设计某种程度上是控制代码片段的执行顺序(flow control)。一系列状态(finite state)被输入，它们被一系列代码片段改变，然后输出。

程序开发不是一次性的，而是不断迭代，不断重构，所以要在整个程序运行周期内保持代码片段的一致性。简单的复制、修改是不行的。要引用名字而不是具体实现，要给每一个片段起合适的名字，要让代码有能力在执行时才展开成需要的模样，比如模板、虚函数和重载。

程序开发是合作性的，你要用别人的库(library)，别人也要用你的(package)。代码片段的功能要被抽象出来，接近自然语义，这样才能被理解，才能生成文档(documetation)。代码片段的状态要相互隔离，要使用接口(interface)和属性(property)，这样才能被包裹(wrapper)，被扩展。

这里有个很棒的教程：Composing Programs

http://composingprograms.com/


* Chapter 1: Building Abstractions with Functions
    1.1 Getting Started
    1.2 Elements of Programming
    1.3 Defining New Functions
    1.4 Designing Functions
    1.5 Control
    1.6 Higher-Order Functions
    1.7 Recursive Functions
* Chapter 2: Building Abstractions with Data
2.1 Introduction
2.2 Data Abstraction
2.3 Sequences
2.4 Mutable Data
2.5 Object-Oriented Programming
2.6 Implementing Classes and Objects
2.7 Object Abstraction
2.8 Efficiency
2.9 Recursive Objects
* Chapter 3: Interpreting Computer Programs
3.1 Introduction
3.2 Functional Programming
3.3 Exceptions
3.4 Interpreters for Languages with Combination
3.5 Interpreters for Languages with Abstraction
* Chapter 4: Data Processing
4.1 Introduction
4.2 Implicit Sequences
4.3 Declarative Programming
4.4 Logic Programming
4.5 Unification
4.6 Distributed Computing
4.7 Distributed Data Processing
4.8 Parallel Computing

## 写一个深度学习程序

1. 找现有的网络资源

    在paperwithcode上找到感兴趣的方向： 
    * benchmark
    * dataset
    * paper
    * code

    然后在下载代码、数据和文章：
    * researchgate
    * scihub(vpn)
    * arxiv
    * google scholar
    * github

    数据集可能是各种格式：
    * tiff/png/bmp
    * h5 (hdf5)
    * torch/tensorflow专用格式
    * db(数据库文件)
    * csv(表格文件)

2. 第一次运行：复现

    看代码文件夹里的`Readme.txt`或`requirements.txt`，在anaconda里创建相应的虚拟环境。

    在你喜欢的IDE或控制台运行程序，进行训练或测试：
    * vscode
    * pycharm
    * jupyter notebook
    * cmd/bash/mobaxterm(remote)
    * idle(python 自带)

    在复现过程中，你可能要改一些bug :-P

    `runoob.com`可以帮你快速入门一些工具。


3. 创作代码

    使用你熟悉的框架，构建神经网络和训练流程:
    * pytorch
    * tensorflow1
    * keras
    * tensorflow2
    * ...
    你会需要一些辅助程序包来处理数据：
    * pandas
    * numpy
    * scipy
    * cv2(opencv-python)
    * PIL(Pillow)
    * sklearn(scikit-learn)
    * skimage(scikit-image)
  
    最开始的时候，它们可能都包含在一个.py或.ipynb文件里。先让程序跑起来，再决定结构。

    写代码时可能出现各种意外情况，尤其是神经网络实验需要将一个程序多次运行在不同的参数和数据上。在每个关键步骤前都要写代码进行检查，抛出异常。这避免程序运行数个小时才“意外终止”，尤其是python这种没有静态检查的解释型语言，如果出错的代码不被执行，程序不会有任何提示。参见`python 官方手册`中关于异常的章节。

    *关键词： try-exception , class Exception, assert*

    使用第三方静态类型检查是个好方法，就像cpp那样。不过未通过的程序不会像cpp那样终止运行。


    有时对代码的改动会引发意外的连锁反应，同过异常信息没法定位问题所在。这时你需要“回退”到上一个能正常执行的版本，并与现在的程序作对比。或着，你需要编写一些实验性的功能，但想保留稳定的当前版本。你需要一个程序版本管理软件：
    * git(local), github(cloud)

    *关键词：git, github, branch, commit, fork, git ssh*


4. 生成日志，记录实验

   
   你会调整网络参数，进行大量的实验，代码只有细微的改变。你需要清晰的记录下来每一次实验的设定、源码和结果。重新做一组实验可能代价高昂。保存训练后的网络可以使用torch/tensorflow自带的函数。保存矩阵可以使用pandas/hdf5/torch/tensorflow/numpy/scipy提供的函数。而保存训练日志(loss,config,note,datetime，hyperparameter)，比如以下的训练设定，一个`dict`，你将如何记录它呢？（数据读取请仔细搜索）
    ````python
    config = {
        "net": "UNet",
        "n_layer":32,
        "optim":"SGD"
    }
    ````

   1. json + open
        保存的文件是纯文本，你可以用notepad打开、阅读甚至修改。
        ````python
        import json
        s = json.dumps(config)
        with open('config.json','w+') as f:
            f.write(s)  
        ````

   2. torch
        自带功能，使用简单。
        ````python
        import torch
        torch.save(config)
        ````

   3. sqlite3
        储存在database里的数据，可以使用sql语句进行处理。就像excel里的各种筛选，但更加复杂。
        ````python 
        import sqlite3
        conn = sqlite3.connect('test.db')
        print "Opened database successfully"
        c = conn.cursor()
        c.execute('''CREATE TABLE CONFIG
            (ID INT PRIMARY KEY     NOT NULL,
            net           TEXT    NOT NULL,
            n_layer        INT     NOT NULL,
            optim        TEXT    NOT NULL);''')
        print "Table created successfully"
        c.execute(f"""INSERT INTO CONFIG (ID,net,n_layer,optim) \
            VALUES (1,"{config['net']}",{config['n_layer']},"{config['optim']}")""")
        conn.commit()
        conn.close()

        ````
        或者
        ````python
        import json
        s = json.dumps(config)
        conn = sqlite3.connect('test.db')
        print "Opened database successfully"
        c = conn.cursor()
        c.execute('''CREATE TABLE CONFIG
            (ID INT PRIMARY KEY     NOT NULL,
            raw         json NOT NULL
            );''')
        print "Table created successfully"
        c.execute(f"""INSERT INTO CONFIG (ID,raw) \
            VALUES (1,{s})""")
        conn.commit()
        conn.close()
        ````

        或者使用 pyside2提供的sql功能，更全面一点
   4. logging
        使用python自带的loggin模块，跟直接写到txt里查不多。自带一些格式化(format)功能。
   5. hdf5
        hdf5擅长存储大量数值矩阵，存储任意格式的其他数据也可以。可以把元信息附加在数值矩阵上，比如“网络在单张图片上的成功率”
   6. tensorboard
    提供专为神经网络打造的GUI界面来查看数据。导出数据有点麻烦。
    ````python
    from torch.utils.tensorboard import SummaryWriter
    log_dir='log'
    writer = SummaryWriter(log_dir = log_dir,comment=__doc__)
    writer.add_hparam(config,{})
    write.close()
    ````

   7. argparse + .bat/.sh
        把参数记录在批处理中，然后用argparse读取命令行参数。一般发布到github上的代码，会用这种方法表示对一个脚本的多次运行。这其实说不上是一种“日志”。

   8. 直接复制很多份代码，每份包含不同的config.
        很方便。代码和数据在一起。改代码时就麻烦了。你得确保每一份拷贝都被修改。

     


5. 程序注释和文档

    写程序必须写注释，没有比读无注释代码更恶心的事了。把所有变量起成`a,b,aaa,bb,t2`这样的名字，程序写起来很方便。但一周过去，谁还能记得这是干什么用的？`python官方手册`里有大量关于写注释、写文档的内容，那是最好的参考。

    *关键词: `__doc__`, docutil, sphinx, PEP 8*

    举个例子，下面的代码是不是比较易懂：

    ````python 
    """
    Abstract PDE solver

    vaersion: a0.1

    NOTICE
    ======
    * ODE3 solver doesnt work!
    * alpha = 3 by default
    """
    from typing import Dict,Optional,Any
    from MyData.MyPDE import MyPDE
    from MyCore import getDevice
    class AbstractPDESolver(Solver):
        """Wrapper class of all solvers.

        For example,
        ```python
        s = AbstractPDESolver(pde3("neumann"))
        ```
        """
        keys_in_config = ['alpha','beta']
        def __init__(self,config:Optional[Dict[str,Any]]=None):

            # check config
            from pprint import pformat
            missing_keys = []
            for k in self.keys_in_config:
                if k not in config:
                    missing_keys.append(k)
            assert len(missing_keys)==0,f"{missing_keys} not in config.\n config= {pformat(config)}"
            super().__init__("pde",config)
        
        def run(x:MyPDE):
            """Solve PDE `x`
            """
            context = {}
            for i in range(n_iter):
                context = x.step(context)
                x.update(context)
                self.hook_after_update(x,context)
            return x.result()
        
        def __str__(self):
            pass #TODO:  description for AbstractPDESolver
    
    def module_test(device):
        pass # TODO : test solver on a dummy pde.
    
    if __name__=='__main__':
        # Test module
        d = getDevice(None)
        module_test(d)
        print('Success.')
    

    ````

    单纯的注释在修改或阅读时有很大帮助，但却没法搜索、交叉引用或者生成摘要。这时需要使用sphinx等软件来生成文档，就像`python 官方文档`一样。sphinx可以使用`autodoc`插件来提取代码里的注释生成文档，你也可以手动撰写章节——它的语法可比latex简单多了。文档以html呈现，打印成pdf也没有问题。sphinx使用rst语言，当然markdown和html也可以。你可以把文档发布到readthedocs上或者github的免费个人主页上（静态站）。

    *关键词: sphinx, reStructuredText, github.io, jekyll ruby, readthedocs*

6. 重构代码
    随着程序膨胀，以及实验变体的增加，你需要拆分它们，并构建自己的程序包。构建程序包的技术，参见`python 官方手册`。

    *关键词：import , package*

    举个例子，你可以拆分出以下模块：
    * Data: 载入数据，子类化torch.util.data.Dataset
    * Device: 分配GPU。迁移到服务器上需要这个。
    * Util:     一些辅助功能，比如批量重命名、删除备份、预处理图片等等
    * Net:  定义网络模型
    * Run: 一组实验
    * Log:  日志记录

    这时你需要一些更高级的python语法。
    *关键词：decorator, class, generator, `__init__`*

    记得使用git来管理程序版本。
    

## python常用程序包和软件

写深度学习，调包就完了。

### 数值计算和算法

|name|example|description|
|----|-------------------------------|----------------------------|
|numpy|             numpy.array([1,2,3],dtype=numpy.float32)|数值计算，矩阵类|
|scipy|             scipy.fft.fft(x)                        |数值计算，算法类|
|pytorch|           torch.nn.MaxUnpool1d(kernel_size, stride=None, padding=0)|深度学习，autograd,tensor|
|tensorflow,keras|  import tensorflow   |深度学习|
|python-opencv|     import cv2          |图像处理|
|scikit-learn||机器学习|
|Pillow||图像处理|
|scikit-image|from skimage.io import imread|图像处理|

### 其他

|name|example|description|
|---|---|---|
|logging||自带的日志模块|
|shutil||命令行工具，比如移动文件、重命名|
|glob||匹配文件名，比如读取所有匹配rex_*.png的文件|
|tensorboard||训练过程可视化，用于pytorch,tensorflow|
|pandas||数据处理|
|matplotlib||绘图|
|hdf5||储存数值矩阵到硬盘|
|tqdm||简单的进度条|
|sqlite3||最简单的python自带数据库sqlite3|
|pyside2||qt5的python版本。在GUI,database等方面提供更完善的支持|
|libgit2||访问git信息|

### 并非在程序脚本中使用的程序

|name&description|example picture|
|-----|----------------------|
anaconda|包管理器，虚拟环境。**必用**。当你要复现实验，要维持大量环境。|
|pip|python自带的包管理器。有些包conda里没有。
增强的python控制台|
ipython|![picture 4]({{site.url}}/assets/image/2021-06-12-AboutProgramming/96111932bc1c1ff05b9e1a783d9918569ae376b9234e4d5409bfe1e8699458b5.png)
交互的python笔记本|
jupyter|![picture 3]({{site.url}}/assets/image/2021-06-12-AboutProgramming/939ec7c5f8b9aa90ba547c4fd1130520bb620e9f73cfeca10353e28e1da6ce69.png) 
生成文档|
sphinx|![picture 2]({{site.url}}/assets/image/2021-06-12-AboutProgramming/24342d339b39dad17eece0a0a398209e1d7bb9ee00f728bb41f3d5e642066bfa.png) 
|训练可视化、生成记录
tensorboard|![picture 1]({{site.url}}/assets/image/2021-06-12-AboutProgramming/7db5161013b171202b52ab99a2f7725e35ff2b4b274250318df1ea92bf4af86d.png)  
程序版本管理|
git|(vscode/gitlens)![picture 6]({{site.url}}/assets/image/2021-06-12-AboutProgramming/373ab3d99865f434c1d29c5edc9452daaae0c6e4f352670bf849ab41ee5ae431.png)  
|查看数据库文件
DB Browser|DB Browser for sqlite![picture 7]({{site.url}}/assets/image/2021-06-12-AboutProgramming/1ef7d5af8a4b9abbd1a7a9fa547bafafb16d9aac7378085aba2ef9b0218e8cac.png)  
|写代码，调试，运行。通过插件扩展，功能超多比如：cpp,markdown,python,latex,sphinx,ruby.你也可以自己写插件
|vs code|![picture 8]({{site.url}}/assets/image/2021-06-12-AboutProgramming/4368fec9c5ec705b18861bd90983852082593c71932ca3baef4c88ae952a8d17.png)  
|pycharm|写代码，调试，运行
docker|虚拟机镜像。有的比赛要求使用docker，或提供docker作为数据集。
|文献管理
|Zotero|![picture 9]({{site.url}}/assets/image/2021-06-12-AboutProgramming/c5538806a6ee46f4c1315341b37af74b394831b8103ff3652720859d234239ca.png)  
|Tex Live|latex 的官方版本。现在不需要CTex也能用中文了。

### 一些在线服务

|url|description|
|-----|-----|
github|**必用**。写论文要附上源码链接的。
pypi|python 程序包的官方来源
https://www.lfd.uci.edu/~gohlke/pythonlibs/ |非官方编译的程序包。官方的出问题了，试试这里。除非你愿意自己下载源码编译。
stackoverflow|问答社区。直接复制错误信息来搜索吧。

### 语言

|name|description|
|----|----|
python|
cuda|除非你自己实现gpu计算细节。
reStructuredText|用于sphinx，类似markdown，更复杂一些
markdown|很通用的写文档的语言，比html简单。支持内嵌latex。
bash|linux的命令行。服务器用linux
bat|dos时代就有了，windows批处理。
powershell|现代的windows批处理。

## 那里找教程？


大多数情况，搜索包的名字，官网上就有最好的教程，尤其是pytorch或tensorflow。或者利用IDE的“源代码跳转”功能看看源代码。因为python语法规范里有超大篇幅讲如何写注释，而且有pypi,github作为统一的来源，所以源代码是很容易阅读的。

> 一些搜索关键词
> * *.pyi python
> * sphinx
> * docutil
> * pylance vscode
> * `__doc__`

## 有用的网站

## 官方网站

大多数情况，搜索包的名字，官网上就有最好的教程，尤其是pytorch或tensorflow。

### pypi

官方软件包

## 教程 QA Example

### runoob

https://www.runoob.com/

覆盖了大多数程序工具，快速入门。

### w3school

### stackoverflow

问答社区。终级加强版的csdn。

## 深度学习

### paperwithcode

https://paperwithcode.com

提供最新的trend,paper,code,benchmark,dataset,etc.

### grandchallege

深度学习，数据集和竞赛。

https://grand-challenge.org

### kaggle

深度学习，数据集和竞赛。

https://www.kaggle.com

### ReScience C

一个网络杂志：“可复现的科学计算程序”。

https://rescience.github.io/

