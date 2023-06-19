
set -euxo pipefail

# 先根据task_compose/uie/requirements.txt安装python 依赖 
# python 版本为3.7.4,高版本需要自行测试python 依赖包

# export PROJECT_PATH="/home/nlpbigdata/net_disk_project/zhubin/sousuo/query_correct/NlpTaskSpace-Torch"

export PROJECT_PATH=E:\\NlpProgram\\NlpTaskSpace-Torch

cd ${PROJECT_PATH}/scripts/uie_data_process

# config 为数据集配置，可以参考具体config 文件
python uie_convert.py  -config=data_config/entity_zh/zh_query.yaml
