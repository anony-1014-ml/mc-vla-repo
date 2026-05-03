#!/bin/bash

workers=10
max_frames=600
temperature=1.0
history_num=4
action_chunk_len=1
instruction_type="normal"

tasks=(
    "kill/kill_zombine"
    "kill/kill_spider"
    "kill/kill_skeleton"
    "kill/kill_sheep"
    "kill/kill_pig"
    "kill/kill_creeper"
    "kill/kill_cow"
    "kill/kill_chicken"

    "mine/mine_oak_log"
    "mine/mine_stone"
    "mine/mine_coal_ore"
    "mine/mine_iron_ore"
    "mine/mine_diamond_ore"
    # "mine/mine_gold_ore"
    "mine/mine_dirt"
    "mine/mine_sand"
    # "mine/mine_dark_oak_log"
    # "mine/mine_redstone_ore"
    "mine/mine_obsidian"    

    "craft/craft_bread"
    "craft/craft_crafting_table"
    "craft/craft_furnace"
    "craft/craft_stick"
    "craft/craft_iron_pickaxe"
    "craft/craft_iron_sword"
    "craft/craft_diamond_chestplate"
    "craft/craft_diamond_boots"
    # "craft/craft_oak_planks"

    "smelt/smelt_iron_ingot"
    "smelt/smelt_gold_ingot"
    "smelt/smelt_coal_from_smelting"
    "smelt/smelt_charcoal"
    "smelt/smelt_glass"
    "smelt/smelt_baked_potato"
    "smelt/smelt_cooked_beef"
    "smelt/smelt_cooked_chicken"
    # "smelt/smelt_cooked_mutton"
    # "smelt/smelt_cooked_porkchop"    
)

out_dir=$1

echo "tasks [${#tasks[@]}] : ${tasks[@]}, workers : $workers"
for top_p in 0.99 ; do 
    echo "temperature : $temperature, top_p : $top_p"
# for top_p in 0.1 0.3 0.5 0.7 0.9 0.99 ; do 
#     echo "temperature : $temperature, top_p : $top_p"

    model_name_or_path="${PRISMATIC_MODEL_DIR}/prism-paligemma-3b-pt-224_cab"

    for task in "${tasks[@]}"; do
        env_config=$task

        # Evaluate
        num_iterations=$(($workers / 5 + 1))
        for ((i = 0; i < num_iterations; i++)); do
            python ./evaluate_e2_single.py \
                --workers $workers \
                --env-config $env_config \
                --max-frames $max_frames \
                --temperature $temperature \
                --top-p $top_p \
                --checkpoints $model_name_or_path \
                --video-main-fold $out_dir \
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