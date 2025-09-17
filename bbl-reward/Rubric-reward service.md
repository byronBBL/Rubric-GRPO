# Rubric-reward service





1. 首先切到可以使用vllm的环境，pip install flask

2. 打开.../bbl-reward/reward.py

   ...修改为对应路径 ![截屏2025-09-17 16.35.17](/Users/bibaolong/Library/Application Support/typora-user-images/截屏2025-09-17 16.35.17.png)

   

3. 打开.../bbl-reward/submit_reward_service.slurm

   ...修改为对应log路径 

![截屏2025-09-17 16.37.06](/Users/bibaolong/Library/Application Support/typora-user-images/截屏2025-09-17 16.37.06.png)

​	改成vllm+flask conda环境：

![截屏2025-09-17 16.37.47](/Users/bibaolong/Library/Application Support/typora-user-images/截屏2025-09-17 16.37.47.png)

​	...修改为对应路径 

![截屏2025-09-17 16.39.14](/Users/bibaolong/Library/Application Support/typora-user-images/截屏2025-09-17 16.39.14.png)

4.sbatch  .../bbl-reward/submit_reward_service.slurm 4次 （general一般每个用户8个gpu，我的一个slurm任务需要两个gpu，可以交四个）。

5.sbatch之后把logs里面打印的网络接口发给bbl就可以。