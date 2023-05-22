cd $(dirname $0)

#第一次运行需要先安装环境依赖requirements.txt
# python3 -m pip install -r ../task_compose/bert_qa/requirements.txt


# 文件预处理
set -euxo pipefail
#进入脚本所在路径
export PROJECT_PATH="/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_code_repository/NlpTaskSpace-Torch"
# 设置项目root路径为PYTHONPATH
export PYTHONPATH=${PROJECT_PATH}

# 进入工程目录
cd ${PROJECT_PATH}

# 本地测试脚本

# python3 task_compose/bert_qa/task_bert_qa_train.py \
# --bert_config_dir=
# --train_data=../data/product_extraction/train_demo.txt \
# --batch_size=32 \
# --max_epochs=2 \
# --loss_type=bce


python3 task_compose/bert_qa/task_bert_qa_predict.py \
--accelerator=gpu \
--devices=1 \
--bert_config_dir=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model \
--train_data=/home/nlpbigdata/net_disk_project/zhubin/sousuo/train_data/query_correct_data/product_extract_train.txt \
--batch_size=16 \
--max_epochs=1 \
--loss_type=bce
