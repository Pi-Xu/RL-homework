# Assignment2: PG

## 代码总体思路

此次的代码思路较为清晰, 除开需要hw1中的填充内容, 根据`hw2.pdf`中的提示即可

## Q1

### 代码逻辑(啰嗦版)

- -n : Number of iterations.
- -b : Batch size (number of state-action pairs sampled while acting according to the current policy at
each iteration).
- -dsa : Flag: if present, sets standardize_advantages to False. Otherwise, by default, standardizes
advantages to have a mean of zero and standard deviation of one.
- -rtg : Flag: if present, sets reward_to_go=True. Otherwise, reward_to_go=False by default.
- --exp_name : Name for experiment, which goes into the name for the data logging directory.

为了防止自己忘记, 再对以上的参数进行解释:

- n: 整个大循环的次数, 大循环中总结起来就是sample数据, 加入到replay buffer中. replay buffer的数据组织形式可以参见`hw1`. 加入到replay buffer之后, 在训练的时候需要重新sample出数据, 使用`num_agent_train_steps_per_iter`次step对数据进行更新(此时根据模型此处应该设置为1), 对policy进行更新.
- b: 即batch size, n的大小

剩下的参数可以根据其英文含义进行理解.

### Experiment 1

- Which value estimator has better performance without advantage-standardization: the trajectory-centric one, or the one using reward-to-go?
  - trajectorty-centric要更好
- Did advantage standardization help?
  - 有较大帮助
- Did the batch size make an impact?
  - 影响不大

## Q2

TODO: 完成代码运行分析, 能够跑通其他代码
