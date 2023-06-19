cd $(dirname $0)

#第一次运行需要先安装环境依赖requirements.txt
# python3 -m pip install -r ../task_compose/bert_qa/requirements.txt

# 文件预处理
set -euxo pipefail
#进入脚本所在路径
export PROJECT_PATH="/home/nlpbigdata/local_disk/NlpTaskSpace-Torch"
# 设置项目root路径为PYTHONPATH
export PYTHONPATH=${PROJECT_PATH}

# 进入工程目录
cd ${PROJECT_PATH}

export TOKENIZERS_PARALLELISM=false
# 本地测试脚本


python3 task_compose/bert_qa/task_bert_qa_train_fast.py \
--output_dir=/home/nlpbigdata/local_disk/output \
--accelerator=gpu \
--devices=4 \
--bert_config_dir=/home/nlpbigdata/net_disk_project/zhubin/nlpprogram_data_repository/bert_resource/resource/pretrain_models/bert_model \
--train_data=/home/nlpbigdata/net_disk_project/zhubin/sousuo/train_data/query_understand_data/train_data_damo_v2.txt \
--dev_data=/home/nlpbigdata/net_disk_project/zhubin/sousuo/train_data/query_understand_data/humanlabel_eval_v1.txt \
--batch_size=8 \
--max_epochs=2 \
--num_sanity_val_steps=0 \
--loss_type=bce
