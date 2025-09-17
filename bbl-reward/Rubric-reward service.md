# Rubric-reward service





1. 首先切到可以使用vllm的环境，pip install flask

2. 打开.../bbl-reward/reward.py

   ...修改为对应路径
   
   <img width="728" height="224" alt="截屏2025-09-17 16 35 17" src="https://github.com/user-attachments/assets/7dd46aaa-21f0-4f78-bc8b-988c1ddde934" />


   

4. 打开.../bbl-reward/submit_reward_service.slurm

   ...修改为对应log路径 

<img width="494" height="53" alt="截屏2025-09-17 16 37 06" src="https://github.com/user-attachments/assets/ee8f2210-8ee4-408b-91be-f7fbb838a583" />


​	改成vllm+flask conda环境：

<img width="469" height="81" alt="截屏2025-09-17 16 37 47" src="https://github.com/user-attachments/assets/32d51df1-396a-4133-8085-a70f0ebf02bb" />


​	...修改为对应路径 


<img width="407" height="67" alt="截屏2025-09-17 16 39 14" src="https://github.com/user-attachments/assets/67458704-2755-4107-a0d8-5b2967bbeb37" />


4.sbatch  .../bbl-reward/submit_reward_service.slurm 4次 （general一般每个用户8个gpu，我的一个slurm任务需要两个gpu，可以交四个）。

5.sbatch之后把logs里面打印的网络接口发给bbl就可以。
