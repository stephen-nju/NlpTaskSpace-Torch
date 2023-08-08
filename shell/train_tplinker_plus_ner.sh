cd $(dirname $0)

#第一次运行需要先安装环境依赖requirements.txt
# python3 -m pip install -r ../task_compose/bert_qa/requirements.txt

# 文件预处理
set -euxo pipefail
#进入脚本所在路径
export PROJECT_PATH="/home/zhubin/code/NlpTaskSpace-Torch"
# 设置项目root路径为PYTHONPATH
export PYTHONPATH=${PROJECT_PATH}

# 进入工程目录
cd ${PROJECT_PATH}

export TOKENIZERS_PARALLELISM=false
# 本地测试脚本

CUDA_VISIABLE_DEVICES=0
# 模型训练收敛极其慢，该数据集上loss的量级要达到10-3级才开始收敛
CUDA_VISIABLE_DEVICES=0 python3 task_compose/tplinker_plus_ner/task_tplinker_plus_ner_train.py \
--output_dir=/home/zhubin/output\
--accelerator=gpu \
--devices=1 \
--bert_model=/home/zhubin/model/bert_model/ \
--train_data=/home/zhubin/train_data/query_understand/humanlabel_eval_v1.txt \
--dev_data=/home/zhubin/train_data/query_understand/humanlabel_eval_v1.txt \
--batch_size=8 \
--num_labels=3 \
--max_epochs=2 \
--max_length=64 \
--lr_scheduler=onecycle \
--num_sanity_val_steps=0 \
--loss_type=bce
