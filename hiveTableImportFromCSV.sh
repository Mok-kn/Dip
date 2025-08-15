#!/bin/bash

# ===================================================================
# Configuration & Setup
# ===================================================================
set -euo pipefail
[[ $# -lt 3 ]] && { echo "用法: $0 <本地CSV文件> <目标数据库> <目标表>"; exit 1; }

# --- 用户输入 ---
LOCAL_CSV_FILE=$1
TARGET_DB=$2
TARGET_TBL=$3

# --- 安全凭证 (在生产环境中，建议从更安全的地方读取) ---
PASSWORD="123456"

# --- 自动发现环境配置 ---
echo "--> 正在自动发现环境配置..."
TSP_FILE="/export/home/omc/etc/CBB/tsp/tsp_inst.cfg"

if grep -q Collector "$TSP_FILE"; then
    KEY_ROLE="Collector"
else
    KEY_ROLE="(Manager1|TS-TSP-Master1)"
fi

HDFS_NODE=$(awk -v KR="$KEY_ROLE" '$0~KR{print $6}' "$TSP_FILE" | awk -F'=' '{print $2}' | head -1)
MAINT_NODE=$(awk -v KR="$KEY_ROLE" '$0~KR{print $2}' "$TSP_FILE" | awk -F'=' '{print $2}' | head -1)
MAE_INST=$(find /opt/cloud -maxdepth 1 -name "MAE*" | head -1 | awk -F'/' '{print $NF}')

# --- 定义远程路径和URL ---
REMOTE_CSV="/tmp/$(basename "$LOCAL_CSV_FILE")"
REMOTE_CSV_NOHDR="/tmp/$(basename "${LOCAL_CSV_FILE}" .csv)_nohdr.csv"
BEELINE_URL="jdbc:hive2://${HDFS_NODE}:22550/;principal=spark2x/hadoop.hadoop.com@HADOOP.COM;saslQop=auth-conf;auth=KERBEROS;user.principal=spark2x/hadoop.hadoop.com@HADOOP.COM;user.keytab=/opt/cloud/${MAE_INST}/apps/HDSparkThriftService/etc/spark2x.keytab;ssl=true"

echo "配置加载完毕:"
echo "  维护节点: ${MAINT_NODE}"
echo "  HDFS节点:  ${HDFS_NODE}"
echo "  远程CSV路径: ${REMOTE_CSV}"
echo ""


# ===================================================================
# Step 1: 将本地CSV文件安全地复制到远程服务器
# ===================================================================
echo "--> 步骤 1: 正在复制 '$LOCAL_CSV_FILE' 到远程服务器..."
expect << EOF
set timeout 300
spawn scp "$LOCAL_CSV_FILE" "ossuser@${MAINT_NODE}:${REMOTE_CSV}"
expect {
    -re "password.*:" { send "$PASSWORD\r"; exp_continue }
    eof
}
EOF
echo "文件复制成功."
echo ""


# ===================================================================
# Step 2: 准备要在远程服务器上执行的完整命令集
# ===================================================================
echo "--> 步骤 2: 正在准备远程执行命令..."
# 使用Heredoc将所有远程命令构建成一个单一的字符串变量
# 这使得下面的 expect 调用非常简洁
REMOTE_COMMANDS=$(cat <<EOF
# 在远程服务器上执行, 开启错误检查
set -euo pipefail
echo "--- [远程] 执行开始 ---"

# 设置远程环境 (与您的导出脚本逻辑相同)
echo "--- [远程] 正在设置环境变量..."
for f in /opt/cloud/${MAE_INST}/apps/HDSparkThriftService/envs/*.properties; do
    [[ -f "\$f" ]] && export \$(grep = "\$f" | xargs)
done
cd /opt/cloud/${MAE_INST}/apps/HDSparkThriftService/rtsp/NdpSparkComponent/pkg/bin

# 定义临时数据库和表的名称
TMP_DB="tmp_imp_\$(date +%s)"
TMP_TBL="\${TMP_DB}.${TARGET_TBL}_csv"

# 准备数据: 去掉CSV文件的表头
echo "--- [远程] 正在准备数据 (去除表头)..."
tail -n +2 "${REMOTE_CSV}" > "${REMOTE_CSV_NOHDR}"

# Beeline操作 1: 创建临时表
echo "--- [远程] 正在创建临时库和表..."
./beeline -u "${BEELINE_URL}" --outputformat=csv2 -e "
CREATE DATABASE IF NOT EXISTS \${TMP_DB};
DROP TABLE IF EXISTS \${TMP_TBL};
CREATE TABLE \${TMP_TBL} (
  TASKID        STRING,
  CGI           STRING,
  CarrierFreq   BIGINT,
  NCGI          STRING,
  NCARRIERFREQ  BIGINT,
  MRCOUNT       BIGINT,
  SRCCGI        STRING,
  SRCMRCOUNT    BIGINT,
  DATE_KEY      TIMESTAMP,
  TASKID_PART   STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE;"

# Beeline操作 2: 将数据加载到临时表
echo "--- [远程] 正在加载数据到临时表..."
./beeline -u "${BEELINE_URL}" --outputformat=csv2 -e "
LOAD DATA LOCAL INPATH '${REMOTE_CSV_NOHDR}' INTO TABLE \${TMP_TBL};"

# Beeline操作 3: 将数据从临时表插入到最终的目标表
echo "--- [远程] 正在将数据插入目标分区表..."
./beeline -u "${BEELINE_URL}" --outputformat=csv2 -e "
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
INSERT INTO TABLE ${TARGET_DB}.${TARGET_TBL} PARTITION (DATE_KEY, TASKID_PART)
SELECT * FROM \${TMP_TBL};"

# Beeline操作 4: 清理临时库和表
echo "--- [远程] 正在清理临时库和表..."
./beeline -u "${BEELINE_URL}" --outputformat=csv2 -e "
DROP TABLE IF EXISTS \${TMP_TBL};
DROP DATABASE IF EXISTS \${TMP_DB};"

# 清理远程服务器上的临时CSV文件
echo "--- [远程] 正在清理临时CSV文件..."
rm -f "${REMOTE_CSV}" "${REMOTE_CSV_NOHDR}"

echo "--- [远程] 执行成功结束 ---"
EOF
)
echo "远程命令准备完毕."
echo ""


# ===================================================================
# Step 3: 通过SSH执行远程命令集
# ===================================================================
echo "--> 步骤 3: 正在远程执行数据导入流程..."
# 使用与您的导出脚本完全相同的非交互式模式
expect << EOF
set timeout -1
# 将整个命令块作为单个参数传递给ssh
spawn ssh "ossuser@${MAINT_NODE}" "${REMOTE_COMMANDS}"
expect {
    -re "password.*:" { send "$PASSWORD\r"; exp_continue }
    eof
}
EOF
echo "远程执行完成."
echo ""

# ===================================================================
# Finalization
# ===================================================================
echo "数据导入任务成功完成！"


--> 步骤 3: 正在远程执行数据导入流程...
extra characters after close-quote
    while executing
"spawn ssh "ossuser@172.28.129.141" "# 在远程服务器上执行, 开启错误检查
set -euo pipefail
echo "-"
