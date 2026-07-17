#!/bin/bash
# 快速运行 observation 分析

set -e

echo "========================================================================"
echo "Observation Sensitivity and Sim2Real Gap Analysis"
echo "========================================================================"
echo ""

# 配置路径
REAL_TRACE="src/pingpong_moz1/pingpong_controller/outputs/rl_sim/simulation_replay/policy_obs_trace.csv"
SIM_TRACE="src/pingpong_moz1/pingpong_controller/data/rl_sim/validate_obs_trace/validate_obs_trace.csv"
OUTPUT_DIR="obs_analysis_results"

# 检查文件是否存在
if [ ! -f "$REAL_TRACE" ]; then
    echo "错误: 找不到 real trace 文件: $REAL_TRACE"
    echo ""
    echo "请先运行: ./RUN_POLICY_ANALYSIS.sh"
    exit 1
fi

if [ ! -f "$SIM_TRACE" ]; then
    echo "错误: 找不到 sim trace 文件: $SIM_TRACE"
    exit 1
fi

echo "输入文件:"
echo "  Real trace: $REAL_TRACE"
echo "  Sim trace:  $SIM_TRACE"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "开始分析..."
echo ""

# 运行分析
python3 analyze_obs_impact_simple.py "$REAL_TRACE" "$SIM_TRACE" "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "分析完成！"
echo "========================================================================"
echo ""
echo "查看结果:"
echo "  文本报告: cat $OUTPUT_DIR/analysis_report.txt"
echo "  JSON数据:  cat $OUTPUT_DIR/analysis_results.json"
echo ""
echo "可视化:"
echo "  obs_impact_analysis.png  - 误差/相关性/贡献度分析"
echo "  top_obs_details.png      - Top观测维度时间序列"
echo "  temporal_evolution.png   - 贡献度时间演化热图"
echo ""

# 自动显示报告的前50行
if [ -f "$OUTPUT_DIR/analysis_report.txt" ]; then
    echo "========================================================================"
    echo "分析报告摘要:"
    echo "========================================================================"
    head -50 "$OUTPUT_DIR/analysis_report.txt"
    echo ""
    echo "完整报告: $OUTPUT_DIR/analysis_report.txt"
fi

