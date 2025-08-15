#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "用法：$0 <Hive表名> [条件语句]"
    exit 1
fi

PASSWORD="Huawei_123"

tableName=$(echo "$1" | tr -d ' ')
condition="${2:-}"
rm -rf "./${tableName}.csv"

tsp_inst_file="/export/home/omc/etc/CBB/tsp/tsp_inst.cfg"
grep -q "Collector" "$tsp_inst_file" && KeyRole="Collector" || KeyRole="(Manager1|TS-TSP-Master1)"
hdfs_list=$(grep -E "$KeyRole" "$tsp_inst_file" | head -1 | awk '{print $6}' | awk -F'=' '{print $2}')
maintain_list=$(grep -E "$KeyRole" "$tsp_inst_file" | head -1 | awk '{print $2}' | awk -F'=' '{print $2}')
maintain_node=$(echo "$maintain_list" | awk '{print $1}')
hdfs_node=$(echo "$hdfs_list" | awk '{print $1}')

maeInst=$(find /opt/cloud -maxdepth 1 -name "MAE*" | head -1 | awk -F'/' '{print $NF}')

if [[ -n "$condition" ]]; then
    sql="SELECT * FROM $tableName $condition;"
else
    sql="SELECT * FROM $tableName;"
fi
echo "SQL: $sql"

expect <<EOF
set timeout -1
spawn ssh ossuser@$maintain_node -t \
"for i in \$(cat /opt/cloud/${maeInst}/apps/HDSparkThriftService/envs/HDSparkThriftService.properties); do export \\$i; done;
 for i in \$(cat /opt/cloud/${maeInst}/apps/HDSparkThriftService/envs/hdsparkthriftservice.properties); do export \\$i; done;
 cd /opt/cloud/${maeInst}/apps/HDSparkThriftService/rtsp/NdpSparkComponent/pkg/bin;
 ./beeline \
   -u \"jdbc:hive2://${hdfs_node}:22550/mbbdb;principal=spark2x/hadoop.hadoop.com@HADOOP.COM;saslQop=auth-conf;auth=KERBEROS;user.principal=spark2x/hadoop.hadoop.com@HADOOP.COM;user.keytab=/opt/cloud/${maeInst}/apps/HDSparkThriftService/etc/spark2x.keytab;ssl=true\" \
   --outputformat=csv2 \
   -e \"$sql\" \
 > /tmp/${tableName}.csv"

expect {
    -re "password.*:" { send "$PASSWORD\r"; exp_continue }
    eof
}
EOF

expect <<EOF
set timeout -1
spawn scp ossuser@$maintain_node:/tmp/${tableName}.csv ./
expect {
    -re "password.*:" { send "$PASSWORD\r"; exp_continue }
    eof
}
EOF

echo "Export finished：./${tableName}.csv"