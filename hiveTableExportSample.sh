#!/bin/bash

# ===================================================================
# Configuration & Setup
# ===================================================================
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "用法：$0 <Hive表名> [可选的WHERE等条件]"
    exit 1
fi

PASSWORD="123456"

# --- 输入处理 ---
tableName=$(echo "$1" | tr -d ' ')
condition="${2:-}"
# 使用 rm -f 更合适
rm -f "./${tableName}.csv"

# --- 环境发现 ---
tsp_inst_file="/export/home/omc/etc/CBB/tsp/tsp_inst.cfg"
grep -q "Collector" "$tsp_inst_file" && KeyRole="Collector" || KeyRole="(Manager1|TS-TSP-Master1)"
hdfs_node=$(grep -E "$KeyRole" "$tsp_inst_file" | head -1 | awk '{print $6}' | awk -F'=' '{print $2}')
maintain_node=$(grep -E "$KeyRole" "$tsp_inst_file" | head -1 | awk '{print $2}' | awk -F'=' '{print $2}')
maeInst=$(find /opt/cloud -maxdepth 1 -name "MAE*" | head -1 | awk -F'/' '{print $NF}')

# --- SQL 构建 ---
if [[ -n "$condition" ]]; then
    sql="SELECT * FROM ${tableName} ${condition};"
else
    sql="SELECT * FROM ${tableName};"
fi
echo "将要执行的SQL: $sql"
echo ""

# ===================================================================
# 步骤 1: 构建将在远程服务器上执行的完整命令
# ===================================================================
# 使用 Heredoc 来构建命令, 所有变量都会被正确展开
REMOTE_COMMANDS=$(cat <<EOF
set -euo pipefail
echo "--- [远程] 执行开始 ---"

# 设置远程环境变量
echo "--- [远程] 正在设置环境变量..."
for f in /opt/cloud/${maeInst}/apps/HDSparkThriftService/envs/*.properties; do
    [[ -f "\$f" ]] && export \$(grep = "\$f" | xargs)
done

echo "--- [远程] 正在进入工作目录..."
cd /opt/cloud/${maeInst}/apps/HDSparkThriftService/rtsp/NdpSparkComponent/pkg/bin

echo "--- [远程] 正在执行 beeline 导出..."
./beeline \
  -u "jdbc:hive2://${hdfs_node}:22550/mbbdb;principal=spark2x/hadoop.hadoop.com@HADOOP.COM;saslQop=auth-conf;auth=KERBEROS;user.principal=spark2x/hadoop.hadoop.com@HADOOP.COM;user.keytab=/opt/cloud/${maeInst}/apps/HDSparkThriftService/etc/spark2x.keytab;ssl=true" \
  --outputformat=csv2 \
  -e "${sql}" \
> /tmp/${tableName}.csv

echo "--- [远程] 导出到 /tmp/${tableName}.csv 完成 ---"
EOF
)

# ===================================================================
# 步骤 2: 通过 SSH 在远程执行命令
# ===================================================================
echo "--> 正在远程服务器 ${maintain_node} 上执行导出..."
export maintain_node PASSWORD REMOTE_COMMANDS
expect << EOF
set timeout -1
spawn ssh "ossuser@\$env(maintain_node)" -- "\$env(REMOTE_COMMANDS)"
expect {
    -re "password.*:" { send "\$env(PASSWORD)\r"; exp_continue }
    eof
}
EOF
echo "远程执行完成."
echo ""

# ===================================================================
# 步骤 3: 将导出的文件从远程服务器复制回本地
# ===================================================================
echo "--> 正在从远程服务器复制结果文件..."
expect << EOF
set timeout -1
spawn scp "ossuser@${maintain_node}:/tmp/${tableName}.csv" ./
expect {
    -re "password.*:" { send "$PASSWORD\r"; exp_continue }
    eof
}
EOF

# ===================================================================
# 步骤 4: (可选但推荐) 清理远程服务器上的临时文件
# ===================================================================
echo "--> 正在清理远程服务器上的临时文件..."
expect << EOF
set timeout 60
spawn ssh "ossuser@${maintain_node}" "rm -f /tmp/${tableName}.csv"
expect {
    -re "password.*:" { send "$PASSWORD\r"; exp_continue }
    eof
}
EOF
echo "清理完成."
echo ""

echo "导出成功！文件已保存为: ./${tableName}.csv"
