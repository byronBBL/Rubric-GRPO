#!/bin/bash
#SBATCH --job-name=rubric_reward
#SBATCH --output=.../logs/reward/rubric_reward_%j.out
#SBATCH --error=.../logs/rubric_reward_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=general
# 只使用L40S GPU
#SBATCH --gres=gpu:L40S:2
#SBATCH --exclude=babel-10-9,babel-11-25,babel-6-29,babel-10-13,babel-3-17,babel-14-37,babel-3-9,babel-12-13,babel-10-5,babel-12-29,babel-3-5,babel-7-9,babel-13-1,babel-2-17,babel-2-13,babel-2-25,babel-12-9,babel-8-9,babel-8-5,babel-7-1,babel-8-13

# 多GPU负载均衡版本

# 设置作业信息
echo "=========================================="
echo "多GPU Skywork奖励服务"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

# 创建日志目录
mkdir -p logs

# 可配置的GPU数量
NUM_GPUS=${NUM_GPUS:-2}
echo "配置使用GPU数量: $NUM_GPUS"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export NUM_GPUS=$NUM_GPUS


# 激活Python环境
echo "激活Python环境..."
source /home/baolongb/miniconda3/etc/profile.d/conda.sh
conda activate verl

# 显示GPU信息
echo "GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# 获取节点IP地址 - 使用第二个IP地址
echo "所有网络接口IP: $(hostname -I)"
node_ip=$(hostname -I | awk '{print $2}')
echo "选择的节点IP地址: $node_ip (第二个IP)"

# 启动服务
echo "=========================================="
echo "启动Rubric奖励服务..."
echo "监听地址: http://$node_ip:8003"
echo "使用GPU数量: $NUM_GPUS"
echo "健康检查: http://$node_ip:8003/health"
echo "统计信息: http://$node_ip:8003/stats"
echo "=========================================="

# 使用多GPU服务
python .../bbl-reward/reward.py

# 作业完成信息
echo "=========================================="
echo "作业完成时间: $(date)"
echo "=========================================="