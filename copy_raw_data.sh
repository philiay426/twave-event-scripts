#!/bin/bash
# 将 2014-01-13-mw63-raw 的数据拷贝到 case/raw
SRC="/mnt/f/sharedate/2014-01-13-mw63-raw"
DST="/mnt/f/sharedate/case/raw"

# 1. 拷贝 SAC 文件（不覆盖已有）
echo "=== 拷贝 SAC 文件 ==="
cp -nv "$SRC"/*.SAC "$DST/" 2>&1

# 2. 拷贝 SACPZ 文件（不覆盖已有）
echo ""
echo "=== 拷贝 SACPZ 文件 ==="
mkdir -p "$DST/SACPZ"
cp -nv "$SRC"/SACPZ/* "$DST/SACPZ/" 2>&1

# 3. 统计结果
echo ""
echo "=== 拷贝后统计 ==="
echo "SAC 文件数: $(ls "$DST"/*.SAC 2>/dev/null | wc -l)"
echo "SACPZ 文件数: $(ls "$DST"/SACPZ/* 2>/dev/null | wc -l)"
echo "台站数: $(ls "$DST"/*.SAC 2>/dev/null | sed 's/.*\.\(TA\.[A-Z0-9]*\)\..*/\1/' | sort -u | wc -l)"
