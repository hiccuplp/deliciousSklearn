# Authors: Oliver Rausch <rauscho@ethz.ch>
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

n_splits = 3

X, y = datasets.load_breast_cancer(return_X_y=True)
# Load and return the breast cancer wisconsin dataset
# 加载返回 乳腺癌威斯康辛数据集 分类
X, y = shuffle(X, y, random_state=42)
y_true = y.copy()
# 为什么使用 y.copy()来进行复制？  y的数据类型：ndarry
# ndarray是一个N维数组类型的对象，与python的基本数据类型列表相比，同一个ndarray中所有元素的数据类型都相同，而列表中可以存储不同类型的数据。
# ndarray在存储的数据类型上做限制，换取了运算效率的提升和数据处理的便捷，在数据分析中非常实用。
# asarray()函数是浅拷贝，copy()函数是深拷贝。对可变数据类型，使用深拷贝避免后续源数据的改变而影响拷贝数据。
# 上述参考链接--- https://zhuanlan.zhihu.com/p/370945563

y[50:] = -1   # 构造无标签数据
total_samples = y.shape[0]

base_classifier = SVC(probability=True, gamma=0.001, random_state=42)

x_values = np.arange(0.4, 1.05, 0.05)
x_values = np.append(x_values, 0.99999)  # 这个是什么操作？为什么要这么操作？
# np.append 为原始array添加一些values
scores = np.empty((x_values.shape[0], n_splits))
amount_labeled = np.empty((x_values.shape[0], n_splits))
amount_iterations = np.empty((x_values.shape[0], n_splits))

for i, threshold in enumerate(x_values):
    self_training_clf = SelfTrainingClassifier(base_classifier, threshold=threshold)

    # We need manual cross validation so that we don't treat -1 as a separate
    # class when computing accuracy
    # 我们需要手动交叉验证，这样在计算精度时就不会把-1当作一个单独的类
    skfolds = StratifiedKFold(n_splits=n_splits)
    for fold, (train_index, test_index) in enumerate(skfolds.split(X, y)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        y_test_true = y_true[test_index]

        self_training_clf.fit(X_train, y_train)

        # The amount of labeled samples that at the end of fitting
        # 拟合结束时标记样品的数量
        amount_labeled[i, fold] = (
            total_samples
            - np.unique(self_training_clf.labeled_iter_, return_counts=True)[1][0]
        )
        # The last iteration the classifier labeled a sample in
        # 在最后一次迭代中，分类器标记一个样本
        amount_iterations[i, fold] = np.max(self_training_clf.labeled_iter_)

        y_pred = self_training_clf.predict(X_test)
        scores[i, fold] = accuracy_score(y_test_true, y_pred)


ax1 = plt.subplot(211)
# 每次都三次交叉一下，最后的得分取三次的平均值，并且计算三次的方差，来分析稳定性
# errorbar 函数的作用是在plot函数的基础上，在数据点位置绘制误差棒。
ax1.errorbar(
    x_values, scores.mean(axis=1), yerr=scores.std(axis=1), capsize=2, color="b"
)
ax1.set_ylabel("Accuracy", color="b")
ax1.tick_params("y", colors="b")

ax2 = ax1.twinx()
ax2.errorbar(
    x_values,
    amount_labeled.mean(axis=1),
    yerr=amount_labeled.std(axis=1),
    capsize=2,
    color="g",
)
ax2.set_ylim(bottom=0)
ax2.set_ylabel("Amount of labeled samples", color="g")
ax2.tick_params("y", colors="g")

ax3 = plt.subplot(212, sharex=ax1)
ax3.errorbar(
    x_values,
    amount_iterations.mean(axis=1),
    yerr=amount_iterations.std(axis=1),
    capsize=2,
    color="b",
)
ax3.set_ylim(bottom=0)
ax3.set_ylabel("Amount of iterations")
ax3.set_xlabel("Threshold")

plt.show()