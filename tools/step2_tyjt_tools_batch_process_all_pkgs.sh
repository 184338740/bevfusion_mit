#!/bin/bash
# 命令：bash ./step2_tyjt_tools_batch_process_all_pkgs.sh

TOOLS_DIR="/data2/xmy/01_project/bevfusion_mit_xmy_1205/tools"
cd $TOOLS_DIR || exit 1

PKG_LIST=(
"2d3d_20250114"
"2d3d_20250218"
"2d3d_20250221"
"2d3d4d_20241122_wuxi"
"2d3d4d_20241218"
"2d3d4d_20250117"
"2d3d4d_20250213"
"2d3d4d_20250218"
"2d3d4d_20250403"
"2d3d4d_20250408"
"2d3d4d_20250618_weiyuan"
"2d3d4d_20250708_weiyuan"
"2d3d4d_20250728_weiyuan"
)

MAX_JOBS=4          # 同时处理的最大数据包数量（根据CPU/内存调整）
RUN_VIS=true        # 是否自动执行可视化

# 定义单个数据包的完整处理函数（转换+可视化）
process_package() {
    local PKG_NAME=$1
    local OUT_DIR="../datasets/tyjt2pkl_pkgs/${PKG_NAME}"
    mkdir -p "$OUT_DIR"
    echo "$PKG_NAME" > "$OUT_DIR/pkg.txt"

    echo "[$(date)] 开始转换: $PKG_NAME"
    python -u tyjt_converter.py \
      --data-root /cephfsdata/users/lishan/00_Data/00_RawData \
      --train-split "$OUT_DIR/pkg.txt" \
      --out-dir "$OUT_DIR/" \
      --max-sweeps 10 \
      --generate-info true \
      --generate-database false \
      --version vis_0427 \
      > "$OUT_DIR/pkg.log" 2>&1
    local ret=$?
    if [ $ret -ne 0 ]; then
        echo "[$(date)] 转换失败: $PKG_NAME (错误码 $ret), 日志: $OUT_DIR/pkg.log"
        return $ret
    fi
    echo "[$(date)] 转换完成: $PKG_NAME"

    if [ "$RUN_VIS" = true ]; then
        local PKL_FILE="$OUT_DIR/tyjt_infos_train.pkl"
        if [ -f "$PKL_FILE" ]; then
            echo "[$(date)] 开始可视化: $PKG_NAME"
            python step2_tyjt_tools_Vis_Gt_pkl.py \
                --pkl "$PKL_FILE" \
                --out_dir "$OUT_DIR/Vis/" \
                --num_samples=20 >> "$OUT_DIR/pkg.log" 2>&1
            echo "[$(date)] 可视化完成: $PKG_NAME"
        else
            echo "[$(date)] 警告: $PKL_FILE 不存在, 跳过可视化 $PKG_NAME"
        fi
    fi
    echo "[$(date)] 全部完成: $PKG_NAME"
}

# 并发控制：同时运行 MAX_JOBS 个后台任务
for PKG_NAME in "${PKG_LIST[@]}"; do
    # 如果当前后台任务数 >= MAX_JOBS，则等待一个任务结束
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
    # 后台启动当前包的处理
    process_package "$PKG_NAME" &
done

# 等待所有后台任务完成
wait

echo "[$(date)] 所有数据包处理完毕。"