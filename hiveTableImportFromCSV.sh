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

PASSWORD="Huawei_123"
expect -c "
set timeout -1
spawn scp \"$CSV\" ossuser@${MAINT_NODE}:${REMOTE_CSV}
expect {
    -re \"password.*:\" { send \"$PASSWORD\\r\"; exp_continue }
    eof
}
wait
"
REMOTE_CSV_NOHDR="/tmp/$(basename "${CSV}" .csv)_nohdr.csv"

ssh -T ossuser@"${MAINT_NODE}" <<EOF
set -euo pipefail

for f in /opt/cloud/${MAE_INST}/apps/HDSparkThriftService/envs/*.properties; do
    [[ -f "\$f" ]] && export \$(grep = "\$f" | xargs)
done
cd /opt/cloud/${MAE_INST}/apps/HDSparkThriftService/rtsp/NdpSparkComponent/pkg/bin

TMP_DB="tmp_imp_\$(date +%s)"
TMP_TBL="\${TMP_DB}.${TBL}_csv"


REMOTE_CSV_NOHDR="/tmp/\$(basename ${REMOTE_CSV} .csv)_nohdr.csv"
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
LOAD DATA LOCAL INPATH '${REMOTE_CSV_NOHDR}' INTO TABLE \${TMP_TBL};"

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

rm -f "${REMOTE_CSV}"
EOF

echo "Import finished"