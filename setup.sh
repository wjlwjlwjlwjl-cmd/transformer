#!/usr/bin/env bash
# 一键搭建 transformer 机器翻译训练环境
# 用法：bash setup.sh
set -e  # 任一命令失败则退出

ENV_NAME="transformer"
PYTHON_VERSION="3.10"

# ---- 镜像配置（国内服务器建议开启，加速下载）----
# PyTorch wheel 镜像：可选 sjtu / aliyun / tuna
# PyTorch wheel 源：直接使用官方源（实测 8~14 MB/s，且格式兼容性最好）
PYTORCH_MIRROR="https://download.pytorch.org/whl/cu124"
# pip 全局源（清华）
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "=============================================="
echo "  Transformer 机器翻译环境搭建脚本"
echo "=============================================="

# 1. 检查 conda
if ! command -v conda &> /dev/null; then
    echo "[错误] 未检测到 conda，请先安装 Miniconda/Anaconda："
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "[1/5] 检测到 conda: $(conda --version)"

# 2. 创建 conda 环境（若已存在则跳过）
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "[2/5] conda 环境 '${ENV_NAME}' 已存在，跳过创建"
else
    echo "[2/5] 创建 conda 环境 '${ENV_NAME}' (Python ${PYTHON_VERSION}) ..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# 3. 激活环境
echo "[3/5] 激活环境 ..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "      当前 Python: $(python --version)"

# 4. 安装 PyTorch（仅 GPU/CUDA 版）
echo "[4/5] 安装 PyTorch 及依赖 ..."

if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    echo "[错误] 未检测到可用的 NVIDIA GPU，本脚本仅支持 GPU 机器。"
    exit 1
fi
echo "      检测到 NVIDIA GPU，安装 CUDA 版 PyTorch ..."

# 固定版本避免 pip 回溯到已弃用的 torchtext 等旧依赖
# --index-url 指定 PyTorch 源，--extra-index-url 让依赖包（sympy/numpy等）走清华源加速
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url "${PYTORCH_MIRROR}" \
    --extra-index-url "${PIP_INDEX}"

pip install datasets tokenizers torchmetrics tensorboard tqdm -i "${PIP_INDEX}"

# 5. 验证环境
echo "[5/5] 验证环境 ..."
python -c "import torch, datasets, tokenizers, torchmetrics, tensorboard, tqdm; print('torch', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('依赖安装成功')"

echo ""
echo "=============================================="
echo "  环境搭建完成！"
echo "=============================================="
echo "训练前先激活环境："
echo "  conda activate ${ENV_NAME}"
echo "然后运行训练："
echo "  python train.py"
echo ""
echo "如已准备好数据集（opus_books_en-ru.hf 与 tokenizer_*.json 已随仓库提交），"
echo "则无需联网下载，直接以上命令即可离线训练。"
