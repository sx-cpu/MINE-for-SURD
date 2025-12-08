# MINE for SURD
使用互信息估计器进行因果分解。
# Structure
1. `High_D_surd.py`: 高维互信息估计器模型，暂时未用到。
2. 互信息计算：
    - `diagnose`: 通过已知互信息的高斯分布样本，来验证互信息估计器的准确性。
    - `blocks.py`: 通过修改脚本中 config 里面的 `target_var` 和 `block` 来选择计算哪种情况、哪个果变量的互信息。
    - `energy_cascade.py`: 通过修改 `target_var` 来选择计算哪一个哪变量的互信息。
3. surd 分解：
    - `surd_blocks`: 对计算出的 blocks 的互信息进行因果分解。注意脚本中`Nt`的设定，需与互信息计算脚本中的`Nt`一致。
    - `surd_energy_cascade.py`: 对计算出的 energy_cascade 的互信息进行因果分解。
        - **绘图脚本能用，但可以参考`SURD`做修改。**
4. utils: 
    - diagnose:
        - `ground_truth.py`: 高斯样本数据生成器。
        - `diagnose_mine.py`: 目前只用到了其中的互信息绘图函数。
        - `train_mine.py`: 弃用。
    - `analytic_eqs.py`: blocks 数据样本生成器。
    - `datasets.py`: 求因变量的组合；求果变量滞后样本。
    - `seed.py`: 随机种子，便于复现。
    - `surd.py`: surd 分解核心脚本。 
    
5. results:
    - MI_results: 保存互信息计算脚本运算出的互信息，便于surd分解脚本的调用。
    - surd_results: 保存分解结果。
6. model:
    - `MLP.py`: 互信息估计器，包括模型结构，训练脚本，评估脚本。
    - 防止不收敛的方法及相应的高参：ema(ema_rate)，正则项(lambda_reg, C_reg)，移动窗口平滑(window_size)。
    - 评估：区别于训练时采用 **batch内部打乱**，求取边缘样本需将果变量样本**全局打乱**，即`use_window=False`，目前来看这样效果更好。
7. logs: 记录互信息计算的训练进程。修改互信息计算脚本内的`block`，便可以自动保存。
8. data: 生成数据和真实数据的保存位置。
    - 真实数据：[SURD 代码仓库内查找](https://github.com/ALD-Lab/SURD/tree/main/data)。
    - 生成数据：选择样本数量(Nt)和样本类型(block)，运行互信息计算脚本自动生成并保存。
9. .vscode: vscode 调试文件 。