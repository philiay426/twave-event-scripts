#!/usr/bin/env bash
# ============================================================
# Codespaces 云端容器初始化 —— T波事件分析脚本
# 【注】依赖清单依据本目录 *.py 的实际 import 统计:
#       matplotlib numpy scipy obspy pygmt geographiclib
# 【注】本目录脚本内含大量本地绝对路径 (/mnt/e/sharedate/..., /home/sy/...),
#       云端无数据, 只能用于代码审阅与逻辑检查
# ============================================================
set -e

echo "[1/3] 安装系统级依赖 GMT (pygmt 的底层库)..."
sudo apt-get update -qq
sudo apt-get install -y -qq gmt gmt-dcw gmt-gshhg \
  || sudo apt-get install -y -qq gmt gshhg-gmt-nc4 \
  || echo "[warn] GMT 安装失败, pygmt 功能不可用"

echo "[2/3] 安装 Python 依赖..."
python -m pip install --upgrade pip
python -m pip install numpy scipy obspy matplotlib pandas \
                        pygmt geographiclib requests
python -m pip install "git+https://github.com/philiay426/T_Phase_Analysis.git" \
  || echo "[warn] t_phase_analysis 未安装, 相关 import 会失败"

echo "[3/3] 自检..."
python - <<'PY'
import sys
for m in ["numpy", "scipy", "obspy", "matplotlib", "pandas", "geographiclib"]:
    try:
        __import__(m)
        print(f"  [OK]  {m}")
    except ImportError as e:
        print(f"  [MISS] {m}: {e}")
for m in ["pygmt", "t_phase_analysis"]:
    try:
        __import__(m)
        print(f"  [OK]  {m}")
    except Exception as e:
        print(f"  [warn] {m}: {e}")
print(f"Python: {sys.version}")
PY

echo "[DONE] 云端环境就绪"
echo "[NOTE] 云端无 SAC 数据, 如需真实处理请在本地运行:"
echo "       /home/sy/miniforge3/envs/obspy/bin/python <script>.py"
