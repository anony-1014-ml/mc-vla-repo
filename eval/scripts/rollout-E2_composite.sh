#!/bin/bash

workers=10
max_frames=1800
temperature=1.0
history_num=4
action_chunk_len=1
instruction_type="normal"

tasks=(
    # mine x craft
    "mid/craft_crafting_table"
    "mid/craft_stick"
    "mid/craft_furnace"
    "mid/craft_stone_pickaxe"
    "mid/craft_stone_sword"

    # mine x smelt
    "mid/craft_baked_potato"
    "mid/craft_charcoal"
    "mid/craft_glass"
    "mid/craft_iron_ingot"

    # combat x smelt
    "mid/craft_cooked_beef"
    "mid/craft_cooked_chicken"
    "mid/craft_cooked_mutton"
    "mid/craft_cooked_porkchop"

    # craft x craft
    "mid/craft_chest"
    "mid/craft_torch"
    "mid/craft_white_bed"
    "mid/craft_wooden_pickaxe"
    "mid/craft_wooden_shovel"
)

out_dir=$1

echo "tasks [${#tasks[@]}] : ${tasks[@]}, workers : $workers"
for top_p in 0.99 ; do 
    echo "temperature : $temperature, top_p : $top_p"

    model_name_or_path="${PRISMATIC_MODEL_DIR}/prism-paligemma-3b-pt-224_cab"

    for task in "${tasks[@]}"; do
        env_config=$task

        # Evaluate
        num_iterations=$(($workers / 5 + 1))
        for ((i = 0; i < num_iterations; i++)); do
            python ./evaluate_e2_composite.py \
                --workers $workers \
                --env-config $env_config \
                --max-frames $max_frames \
                --temperature $temperature \
                --top-p $top_p \
                --checkpoints $model_name_or_path \
                --video-main-fold $out_dir  \
                --history-num $history_num \
                --instruction-type $instruction_type \
                --action-chunk-len $action_chunk_len \
                #--verbos True \
            # 如果 Python 脚本执行成功，则退出循环
            if [[ $? -eq 0 ]]; then
                break
            fi
            if [[ $i -lt $((num_iterations - 1)) ]]; then
                sleep 10
            fi
        done
    done 
done 