# optimize

新的优化库，主要改动：
1，加入对优化区域优先级考虑
2，加入对目标函数重叠区域的考虑
3，改动优化函数 loss 的求导，使用 ENZYME 框架

文件组织顺序
. --> 当前文件夹
opt.py. --> 优化逻辑主体设计
enzymeAutoGrad --> ENZYME 框架
optEngine --> python 程序与 C++程序的接口，FISTA，IPOPT 等算法定义部分
docs --> 文档文件夹

相应 git 操作

如果 thrust 和 pybind11 文件夹内为空，请运行 git submodule update --init
