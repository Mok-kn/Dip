#!/bin/bash

set -euo pipefail
[[ $# -lt 3 ]] && { echo "Usage: $0 <csv> <db> <table>"; exit 1; }

CSV=$1
DB=$2
TBL=$3

TSP_FILE="/export/home/omc/etc/CBB/tsp/tsp_inst.cfg"

if grep -q Collector "$TSP_FILE"; then
    KEY_ROLE="Collector"
else
    KEY_ROLE="(Manager1|TS-TSP-Master1)"
fi

HDFS_NODE=$(awk -v KR="$KEY_ROLE" '$0~KR{print $6}' "$TSP_FILE" | awk -F'=' '{print $2}' | head -1)
MAINT_NODE=$(awk -v KR="$KEY_ROLE" '$0~KR{print $2}' "$TSP_FILE" | awk -F'=' '{print $2}' | head -1)
MAE_INST=$(find /opt/cloud -maxdepth 1 -name "MAE*" | head -1 | awk -F'/' '{print $NF}')

REMOTE_CSV="/tmp/$(basename "$CSV")"
BEELINE_URL="jdbc:hive2://${HDFS_NODE}:22550/;principal=spark2x/hadoop.hadoop.com@HADOOP.COM;saslQop=auth-conf;auth=KERBEROS;user.principal=spark2x/hadoop.hadoop.com@HADOOP.COM;user.keytab=/opt/cloud/${MAE_INST}/apps/HDSparkThriftService/etc/spark2x.keytab;ssl=true"

# ####################################################################
# ## 关键修改部分开始
# ####################################################################

# 仍然硬编码密码，因为这是前提条件
PASSWORD="123456"

# 步骤1: 使用 expect 处理 scp (这部分和您原来的一样)
echo "Step 1: Copying CSV file to remote server..."
expect -c "
set timeout 120
spawn scp \"$CSV\" ossuser@${MAINT_NODE}:${REMOTE_CSV}
expect {
    -re \"password.*:\" { send \"$PASSWORD\\r\"; exp_continue }
    eof
}
wait
"

# 步骤2: 将原 heredoc 中的所有命令读入一个 Bash 变量
# 使用 `cat <<EOF` 可以完美保留变量展开规则 (本地的展开, 远程的\$f不展开)
REMOTE_COMMANDS=$(cat <<EOF
set -euo pipefail

# 远程环境设置
for f in /opt/cloud/${MAE_INST}/apps/HDSparkThriftService/envs/*.properties; do
    [[ -f "\$f" ]] && export \$(grep = "\$f" | xargs)
done
cd /opt/cloud/${MAE_INST}/apps/HDSparkThriftService/rtsp/NdpSparkComponent/pkg/bin

# 远程变量定义
TMP_DB="tmp_imp_\$(date +%s)"
TMP_TBL="\${TMP_DB}.${TBL}_csv"
REMOTE_CSV_NOHDR="/tmp/\$(basename ${REMOTE_CSV} .csv)_nohdr.csv"

# 执行数据处理
tail -n +2 "${REMOTE_CSV}" > "\${REMOTE_CSV_NOHDR}"

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

./beeline -u "${BEELINE_URL}" --outputformat=csv2 -e "
LOAD DATA LOCAL INPATH '\${REMOTE_CSV_NOHDR}' INTO TABLE \${TMP_TBL};"

./beeline -u "${BEELINE_URL}" --outputformat=csv2 -e "
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
INSERT INTO TABLE ${DB}.${TBL} PARTITION (DATE_KEY, TASKID_PART)
SELECT
  TASKID,
  CGI,
  CarrierFreq,
  NCGI,
  NCARRIERFREQ,
  MRCOUNT,
  SRCCGI,
  SRCMRCOUNT,
  DATE_KEY,
  TASKID_PART
FROM \${TMP_TBL};"

./beeline -u "${BEELINE_URL}" --outputformat=csv2 -e "
DROP TABLE \${TMP_TBL};
DROP DATABASE \${TMP_DB};"

# 清理远程文件
rm -f "${REMOTE_CSV}" "\${REMOTE_CSV_NOHDR}"
EOF
)


# 步骤3: 使用新的 expect 块来处理 ssh 登录和命令执行
echo "Step 2: Executing commands on remote server..."
echo "Step 2: Executing commands on remote server..."
expect -c "
set timeout -1
set password \"$PASSWORD\"
# 注意这里的 { ... }，它能防止 bash 变量和 expect 变量的混淆
set remote_commands {${REMOTE_COMMANDS}}

spawn ssh -T ossuser@${MAINT_NODE}

# 使用一个循环块来处理多种可能的输出
expect {
    timeout {
        # 如果等待超过默认时间（这里因为 set timeout -1 所以不会触发，但这是好习惯）
        # 或者在特定 expect 中设置了超时，则会执行这里
        send_user \"Error: Timed out waiting for prompt.\\n\"
        exit 1
    }
    -re \"password.*:\" {
        send \"\$password\\r\"
        exp_continue
    }
    # 这是关键的修改：使用更通用的正则表达式匹配 shell 提示符
    -re {[$#>] $} {
        # 成功匹配到提示符，expect 块将正常结束
    }
}

# 只有在成功等到 shell 提示符后，代码才会执行到这里
send \"\$remote_commands\\r\"

# 发送 exit 命令来关闭 session
send \"exit\\r\"

# 等待进程结束
expect eof
wait
"

# ####################################################################
# ## 关键修改部分结束
# ####################################################################

echo "Import finished"

