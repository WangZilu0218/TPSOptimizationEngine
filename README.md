# optimize

新的优化库，主要改动：
1，加入对优化区域优先级考虑，
2，加入对目标函数重叠区域的考虑
3，改动优化函数loss的求导，使用ENZYME框架

文件组织顺序
.   --> 当前文件夹
  opt.py. --> 优化逻辑主体设计
  enzymeAutoGrad --> ENZYME框架
  
  


