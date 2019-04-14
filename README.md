# Multi-label_Text_classification
PyTorch实现的多标签的文本分类

这一项目与之前commit的单标签文本分类(https://github.com/TianWuYuJiangHenShou/textClassifier)
不同，这里每条数据的label可能是多个的，并且可能这些标签之间是父子标签关系。
标签之间的父子关系在元数据里面有专门文件表示，但是这一信息暂时没有用上。

工具：
word2vec:词向量之前已经训练好了，直接加载就行

Pytorch：这次实验的数据量较大，不能像之前的单标签分类任务一样用TorchText加载数据，一次性读取数据，然后批量加载

这次采用Pytorch自带的Torch.utils.data.DataLoader加载数据，通过调用__getitem__方法，一次调用getitem只返回一个样本，这样节省了计算资源的压力。
(总的标签有1999个，训练的时候每个data都有一个Tensor(1999),加载起来最少32G，使用GPU的话，最起码要更大的GPU才能完成训练）


