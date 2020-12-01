# UCAS-NLP
UCAS自然语言处理的编程大作业，一个完善的文本分类系统。

> 本项目已上传Github，[地址](https://github.com/GentleCP/UCAS-NLP)

## 作者
- 董超鹏
- 胡枭
- 黄振洋
- 宾浩宇
- 王荟昭

## 系统结构
```text
UCAS-NLP
|--CNN  # TextCNN做文本分类,包含训练的模型
    |--compress_train_num.tsv  # 处理后的训练数据
    |--compress_val_num.tsv  # 处理后的验证数据
    |--compress_test_num.tsv  # 处理后的测试数据
|--lstm  # LSTM做文本分类
    |--THUCNews.csv.xz  # 处理过后的训练数据
    |--sub-THUCNews.csv.xz   # 处理过后测试数据
|--NaiveBayes  # 朴素贝叶斯做文本分类
    |--data  # 原始数据
    |--train.txt  # 处理后的训练数据
    |--test.txt  # 处理后的测试数据
|--Website  # 网页站点配置demo
document.pdf  # 项目文档
manage.py  # 站点管理文件
requirements.txt  # 依赖包文件
```

## 使用

### 训练测试模型
由于各个模型的设计者不同，因此三个分类器的训练测试过程也不尽相同，具体
分别如下：
- CNN  
    直接运行`cnn_train.py`进行训练、测试
- LSTM  
    直接运行`train.py`进行训练，运行`test.py`进行测试
- Naive Bayes  
    在终端使用`jupyter notebook` 命令，打开`naive_bayes.ipynb`, 
    依次执行每个代码框内代码（第一个是对数据预处理形成train和test数据，可跳过，否则，
    需删除目录下已有`train.txt`和`test.txt`文件）。


### 文本分类
通过在根目录下执行`python manage.py runserver` 运行网页demo，
详情参考项目文档的第六部分，部署与使用


### 测试站点
~~我们已将系统部署在服务器，[点我前往](http://nlp.scuseek.com/)~~
> ~~如发现站点服务宕机，失效，请及时联系我们小组成员，我们会及时恢复，并予以反馈~~

你也可以选择本地部署，部署过程参考`document.pdf`

