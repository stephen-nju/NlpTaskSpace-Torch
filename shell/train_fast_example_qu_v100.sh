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

CUDA_VISIABLE_DEVICES=0,1 python3 task_compose/bert_qa/task_bert_qa_train_fast.py \
--output_dir=/home/zhubin/output_dir \
--accelerator=gpu \
--devices=2 \
--bert_config_dir=/home/zhubin/model/bert_model \
--train_data=/home/zhubin/train_data/query_understand/train_data_damo_v2_humnan_optimize_0810_v4.txt \
--dev_data=/home/zhubin/train_data/query_understand/humanlabel_eval_v1.txt \
--batch_size=32 \
--max_epochs=10 \
--max_seq_length=128 \
--n_best_size=1 \
--max_answer_length=10 \
--num_sanity_val_steps=0 \
--loss_type=bce
