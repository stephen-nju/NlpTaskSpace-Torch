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

# 模型训练收敛极其慢，该数据集上loss的量级要达到10-3级才开始收敛
CUDA_VISIABLE_DEVICES=3 python3 task_compose/global_pointer/task_global_pointer_ner_train.py \
fit \
--trainer.accelerator=gpu \
--trainer.devices=1 \
--trainer.precision=32 \
--trainer.max_epochs=30 \
--trainer.num_sanity_val_steps=0 \
--data.train_data=/home/zhubin/train_data/query_understand/humanlabel_eval_v1.txt \
--data.dev_data=/home/zhubin/train_data/query_understand/humanlabel_eval_v1.txt \
--data.batch_size=16 \
--data.max_length=64 \
--data.workers=4 \
--data.num_labels=3 \
--model.model_path=/home/zhubin/model/bert_model/ \
--model.loss_type=bce \
--model.lr=2e-5 \
--model.optimizer=adamw \
--model.lr_scheduler=cawr \
--model.rewarm_epoch_num=2 \
--model.weight_decay=1e-2 \
--model.warmup_proportion=0.1 \
--model.adam_epsilon=1e-8 \

