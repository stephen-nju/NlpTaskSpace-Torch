# -*- coding:utf-8 -*-
export CUDA_VISIBLE_DEVICES="0"

export PROJECT_PATH="E:\\NlpProgram\\NlpTaskSpace-Torch"
export batch_size="4"
export model_name=E:\\Resources\\uie-char-small\\uie-char-small
export task_name="meta"
export decoding_format='spotasoc'
export lr=1e-4
export seed="42"
export lr_scheduler=linear
export label_smoothing="0"
export epoch=30
export eval_steps=0
export warmup_ratio=0
export constraint_decoding=''
export verbose=false
export fp16=''
export negative=-1
export positive=1
export ordered_prompt=True
export max_source_length=256
export spot_noise=0
export asoc_noise=0
export map_config=offset_map/closest_offset_en.yaml
export data_folder=E:\\NlpProgram\\NlpTaskSpace-Torch\\data\\text2spotasoc\\entity_zh\\zh_weibo
export PYTHONPATH=${PROJECT_PATH}
export output_dir=output

if [[ ${eval_steps} == 0 ]]
then
  evaluation_strategy='epoch'
else
  evaluation_strategy='steps'
fi
# source scripts/function_code.bash

# for index in $(seq 1 ${run_time}); do
#   output_dir=${model_folder}_run${index}

if [[ ${verbose} == true ]]
then
    stdout_file=/dev/stdout
    stderr_file=/dev/stderr
    disable_tqdm=False
else
    stdout_file=${output_dir}.log
    stderr_file=${output_dir}.err
    disable_tqdm=True
fi

cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} /d/ProgramData/miniconda3/envs/uie/python task_compose/uie/task_uie_finetune.py \
    --do_train --do_eval --do_predict ${constraint_decoding} ${fp16} \
    --use_fast_tokenizer=True \
    --overwrite_output_dir=True \
    --ddp_find_unused_parameters=False \
    --predict_with_generate \
    --evaluation_strategy=${evaluation_strategy} \
    --save_strategy=${evaluation_strategy} \
    --metric_for_best_model eval_overall-F1 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --max_source_length=${max_source_length:-"256"} \
    --max_prefix_length=${max_prefix_length:-"-1"} \
    --max_target_length=${max_target_length:-"192"} \
    --num_train_epochs=${epoch} \
    --task=${task_name} \
    --train_file=${data_folder}/train.json \
    --validation_file=${data_folder}/val.json \
    --test_file=${data_folder}/test.json \
    --record_schema=${data_folder}/record.schema \
    --per_device_train_batch_size=${batch_size} \
    --per_device_eval_batch_size=$((batch_size * 4)) \
    --output_dir=${output_dir} \
    --logging_dir=${output_dir}_log \
    --model_name_or_path=${model_name} \
    --learning_rate=${lr} \
    --source_prefix="${task_name}: " \
    --lr_scheduler_type=${lr_scheduler} \
    --label_smoothing_factor=${label_smoothing} \
    --eval_steps ${eval_steps} \
    --decoding_format ${decoding_format} \
    --warmup_ratio ${warmup_ratio} \
    --preprocessing_num_workers=4 \
    --dataloader_num_workers=0 \
    --meta_negative=${negative} \
    --meta_positive_rate=${positive} \
    --skip_memory_metrics \
    --no_remove_unused_columns \
    --ordered_prompt=${ordered_prompt} \
    --save_better_checkpoint=False \
    --start_eval_step=${start_eval_step:-"0"} \
    --spot_noise=${spot_noise} \
    --asoc_noise=${asoc_noise} \
    --seed=${seed}${index} --disable_tqdm=${disable_tqdm}

  if [[ ${verbose} != true ]]
  then
    tail -n 200 ${stderr_file}
  fi

  echo "Map Config" ${map_config}
  # python3 scripts/sel2record.py -p ${output_dir} -g ${data_folder} -v -d ${decoding_format} -c ${map_config}
  # python3 scripts/eval_extraction.py -p ${output_dir} -g ${data_folder} -w -m ${eval_match_mode:-"normal"}

