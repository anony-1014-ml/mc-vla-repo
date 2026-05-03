import argparse
from rich import print,console
from pathlib import Path
import os
import random
import numpy as np
import hydra
import json
# import ray

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.callbacks import (
    MinecraftCallback,
    SpeedTestCallback, 
    RecordCallback, 
    RewardsCallback, 
    TaskCallback, 
    FastResetCallback, 
    FastResetwoTPCallback,
    InitInventoryCallback,
    SummonMobsCallback,
    CommandsCallback,
    # TeleportCallback,
)

from env_helper.craft_agent import CraftWorker
from env_helper.smelt_agent import SmeltWorker

import draw_utils
import file_utils

import agent_wrapper
from file_utils import load_json_file
from minestudio.models import VPTPolicy, SteveOnePolicy, CaBVLAPolicy

import torch
from torchcodec.decoders import VideoDecoder
torch.set_printoptions(linewidth=10000)

#----------------------------------------------------------------------------------------
from minestudio.models.cab_vla.body import convert_text_to_intention_for_inference

#----------------------------------------------------------------------------------------
#       
def convert_text_to_intention_steve_1(text):
    # #
    text = text.replace("minecraft.", "")
    if "craft_item:" in text:
        prompt = text.replace("craft_item:",  "").replace("_"," ")
        prompt = f"make {prompt}, craft {prompt}"
    elif "kill_entity:" in text:
        prompt = text.replace("kill_entity:", "").replace("_"," ")
        prompt = f"kill a {prompt}"
    elif "mine_block:" in text:
        prompt = text.replace("mine_block:",  "").replace("_"," ")
        prompt = f"mine {prompt}, go mining, get {prompt}"
    else:
        prompt = text

    # #
    return prompt

#----------------------------------------------------------------------------------------
#
def load_rollout_data(video_file):

    # set path #
    video_file  = Path(video_file)
    action_file = video_file.with_suffix('.action.json')
    reward_file = video_file.with_suffix('.reward.json')

    # load data #
    # observation
    video = VideoDecoder(video_file, device="cpu")
    base_frame_indices = np.array( [ i for i in range(video.metadata.num_frames) ] )
    observations = video.get_frames_at(base_frame_indices).data
    # action
    with open(action_file, "r") as f:
        actions = json.load(f)
    # reward
    with open(reward_file, "r") as f:
        rewards = json.load(f)

    # return #
    return observations, actions, rewards

#
def evaluate(results,member_ids,video_path,rollout_path,checkpoints,environment_config:dict,model_config:dict,device="cuda:0",base_url=""):

    # set cfg #
    hydra.core.global_hydra.GlobalHydra.instance().clear() # 清理 Hydra 的全局实例
    config_path = Path(f"{environment_config['env_config']}.yaml")
    config_name = config_path.stem
    config_path = os.path.join("./config",config_path.parent)
    hydra.initialize(config_path=config_path, version_base='1.3')
    cfg = hydra.compose(config_name=config_name)
    
    # create agent #
    # #
    if base_url != "":
        # agent #
        agent = agent_wrapper.VLLM_AGENT(checkpoint_path=checkpoints,base_url=base_url,**model_config)
        agent_resolution = cfg.origin_resolution

        # camera_config #
        camera_cfg = CameraConfig(**cfg.camera_config)
    # #
    elif '_VPT' in checkpoints:
        # agent #
        agent = VPTPolicy.from_pretrained(checkpoints).to("cuda")
        agent.eval()
        agent_resolution = [128, 128]

        # camera_config #
        camera_cfg = CameraConfig()
    # #
    elif '_STEVE-1' in checkpoints:
        # agent #
        agent = SteveOnePolicy.from_pretrained(checkpoints).to("cuda")
        agent.mineclip.clip_model.language_model = str(Path(checkpoints).parent.parent / "clip-vit-base-patch16")
        agent.eval()
        agent_resolution = [128, 128]

        # camera_config #
        camera_cfg = CameraConfig()
    # #
    elif 'prism-paligemma' in checkpoints:
        # agent #
        agent = CaBVLAPolicy(
            checkpoints,
            temperature=model_config["temperature"],
            nucleus_prob=model_config["top_p"], # 0.99
            ).to("cuda")
        agent.eval()
        agent_resolution = [224, 224] #[640, 360]

        # camera_config #
        camera_cfg = CameraConfig()
    # #
    else:
        raise NotImplementedError

    # #
    env_refresh_interval = 1
    for trial_num, member_id in enumerate(member_ids):     

        # load rollout_data #
        rollout_obss, rollout_actions, rollout_rewards = load_rollout_data( os.path.join( rollout_path, f"rollout_episode_{member_id:06}.mp4" ) )

        # # check env_refresh_interval #
        # if ( trial_num % env_refresh_interval ) == 0:
        #     # create callbacks #
        #     record_callback = RecordCallback(record_path=Path(video_path), fps=20, show_actions=True)  
        #     init_inventory_callback = InitInventoryCallback(
        #         cfg.init_inventory,
        #         # inventory_distraction_level=cfg.inventory_distraction_level,
        #         # equip_distraction_level="normal",
        #         distraction_level=cfg.inventory_distraction_level,
        #     ) if len(cfg.init_inventory) > 0 else MinecraftCallback()
        #     callbacks = [
        #         # FastResetwoTPCallback(start_time=cfg.start_time,),
        #         SpeedTestCallback(50), 
        #         TaskCallback(getattr(cfg,"task_conf",None)),
        #         RewardsCallback(getattr(cfg,"reward_conf",None)),
        #         init_inventory_callback,
        #         CommandsCallback(getattr(cfg,"command",[]),),
        #         record_callback,
        #     ]
        #     if save_rollout_flag:
        #         rollout_record_callback = RecordCallback(
        #             record_path=Path(video_path), fps=20, frame_type='obs', show_actions=False, record_actions=True, record_rewards=True, record_infos=False,#True,
        #             prefix="rollout")
        #         callbacks.append(rollout_record_callback)
        #     #if hasattr(cfg,"teleport"):
        #     #    callbacks.append(TeleportCallback(x=cfg.teleport.x, y=cfg.teleport.y, z=cfg.teleport.z,))
        #     if cfg.mobs:
        #         callbacks.append(SummonMobsCallback(cfg.mobs))

        #     # create env #
        #     env = MinecraftSim(
        #         action_type="env",
        #         seed=cfg.seed,
        #         obs_size=agent_resolution,
        #         render_size=cfg.resize_resolution,
        #         camera_config=camera_cfg,
        #         preferred_spawn_biome=getattr(cfg,"preferred_spawn_biome",None),
        #         callbacks = callbacks
        #     )
        #     #
        #     if 'prism-paligemma' in checkpoints:
        #         env.action_mapper = agent.action_proc.action_mapper
        #         env.action_transformer = agent.action_proc.action_transformer

        # # reset env #
        # try:
        #     obs, info = env.reset()
        # except Exception as e:
        #     print(f"error : {e}")
        #     os._exit(1)

        # # prepare env with pre_agent #
        # # change action_type : agent -> env
        # env.action_type = "env" 

        # # create pre_agent
        # pre_agent = None
        # worker_type =  getattr(cfg,"worker", None)
        # if   worker_type == "craft":
        #     pre_agent = CraftWorker(env,if_discrete=True)
        # elif worker_type == "smelt":
        #     pre_agent = SmeltWorker(env,if_discrete=True)
        # elif worker_type == "mine":
        #     pre_agent = CraftWorker(env,if_discrete=True)

        # # prepare gui
        # need_crafting_table = False
        # if getattr(cfg, "need_gui", False):
        #     need_crafting_table= getattr(cfg,"need_crafting_table", False)
        #     need_furnace = getattr(cfg,"need_furnace", False)
        #     if need_crafting_table:
        #         try:
        #             frames,_,_ = pre_agent.open_crating_table_wo_recipe()
        #         except AssertionError as e:
        #             env.close()
        #             console.Console().log(f"error: {e}")
        #             return False,-1

        #     elif need_furnace:
        #         try:
        #             frames,_,_ = pre_agent.open_furnace_wo_recipe()
        #         except AssertionError as e:
        #             env.close()
        #             console.Console().log(f"error: {e}")
        #             return False,-1

        #     else:
        #         pre_agent._null_action(1)
        #         if not pre_agent.info['isGuiOpen']:
        #             pre_agent._call_func('inventory')

        # # wait for spawn mobs
        # if cfg.mobs:
        #     for _ in range(20*1):
        #         env.step(env.noop_action())

        # # change action_type : env -> agent
        # env.action_type = "agent"  

        # # set callbacks #
        # # record_callback
        # record_callback.forget()
        # record_callback.episode_id = member_id
        # # rollout_record_callback
        # if save_rollout_flag:
        #     rollout_record_callback.forget()
        #     rollout_record_callback.episode_id = member_id

        # init agent #
        # common #
        state_in = None
        # #
        if 'prism-paligemma' in checkpoints:
            # clear agent #
            agent.clear_agent()        

        # set intstructions #
        # #
        if base_url != "":
            # instructions #
            instructions = [item["text"] for item in cfg.task_conf]
        # #
        elif '_VPT' in checkpoints:
            pass
        # #
        elif '_STEVE-1' in checkpoints:
            # instructions #
            instructions = [item["text"] for item in cfg.task_conf]
            # intention = instructions[0].replace("_"," ").replace(":"," ")
            intention = convert_text_to_intention_steve_1(instructions[0])     
            print(f"intention : {intention}")       

            # prepare condition #
            condition = agent.prepare_condition(
                {
                    'cond_scale': 4.0,
                    'text': intention
                }
            )
        # #
        elif 'prism-paligemma' in checkpoints:
            # instructions #
            instructions = [item["text"] for item in cfg.task_conf]
            intention_high = "survival skill."
            intention_fine = convert_text_to_intention_for_inference(instructions[0])
            print(f"intention_high : {intention_high}, intention_fine : {intention_fine}")

            # set intention #
            agent.set_intention(intention=[ intention_high, intention_fine ])
        # #
        else:
            raise NotImplementedError

        # Rollout #
        rollout_frame_idx = 0
        obs, reward = { 'image' : rollout_obss[rollout_frame_idx].permute(1, 2, 0) }, rollout_rewards[rollout_frame_idx]
        #
        success = (False,environment_config["max_frames"])
        for i in range(environment_config["max_frames"]):
            # #
            if base_url != "":
                action = agent.forward([info["pov"]],instructions,verbos=environment_config["verbos"],need_crafting_table = need_crafting_table)
            elif '_VPT' in checkpoints:
                action, state_in = agent.get_action(input=obs, state_in=state_in, input_shape='*')
            elif '_STEVE-1' in checkpoints:
                action, state_in = agent.get_action(input={'image' : obs['image'], 'condition' : condition }, state_in=state_in, input_shape='*')
            elif 'prism-paligemma' in checkpoints:            
                # action, state_in = agent.get_action(input={'image' : obs['image'], 'intention' : [ intention_high, intention_fine ] }, state_in=state_in, input_shape='*', deterministic=False)
                action, state_in = agent.get_action(input={'image' : obs['image'] }, state_in=state_in, input_shape='*', deterministic=False)

            # #
            if environment_config["verbos"]:
                console.Console().log(action)
            # obs, reward, terminated, truncated, info = env.step(action)
            rollout_frame_idx += 1
            if rollout_frame_idx >= len(rollout_obss): break
            obs, reward = { 'image' : rollout_obss[rollout_frame_idx].permute(1, 2, 0) }, rollout_rewards[rollout_frame_idx]

            # #
            if reward>0:
                success = (True,i)
                break   
            
        # sample another 50 steps if success
        if success[0]:
            for i in range(50):
                # #
                if base_url != "":
                    action = agent.forward([info["pov"]],instructions,verbos=environment_config["verbos"],need_crafting_table = need_crafting_table)
                elif '_VPT' in checkpoints:
                    action, state_in = agent.get_action(input=obs, state_in=state_in, input_shape='*')
                elif '_STEVE-1' in checkpoints:
                    action, state_in = agent.get_action(input={'image' : obs['image'], 'condition' : condition }, state_in=state_in, input_shape='*')
                elif 'prism-paligemma' in checkpoints:            
                    # action, state_in = agent.get_action(input={'image' : obs['image'], 'intention' : [ intention_high, intention_fine ] }, state_in=state_in, input_shape='*', deterministic=False)
                    action, state_in = agent.get_action(input={'image' : obs['image'] }, state_in=state_in, input_shape='*', deterministic=False)

                # #
                # obs, reward, terminated, truncated, info = env.step(action)
                rollout_frame_idx += 1
                if rollout_frame_idx >= len(rollout_obss): break
                obs, reward = { 'image' : rollout_obss[rollout_frame_idx].permute(1, 2, 0) }, rollout_rewards[rollout_frame_idx]

        # #
        result = (success[0],success[1],member_id)
        results.append(result)
        print(f"Done : {result}")

        # dump results #
        file_utils.dump_json_file(results, os.path.join(video_path,"end.json"))

        # dump intention_history #
        if 'prism-paligemma' in checkpoints:
            file_utils.dump_json_file(agent.intention_dict, os.path.join(video_path, f"intention_dict_{member_id:06}.json"), if_backup=False)
            file_utils.dump_json_file(agent.intention_history, os.path.join(video_path, f"intention_history_{member_id:06}.json"), if_backup=False)
            # file_utils.dump_json_file(agent.intention_history, os.path.join(video_path, f"intention_history_{member_id:06}.json.gz"), if_backup=False)

        # # set callbacks #
        # # record_callback
        # record_callback._save_episode()
        # # rollout_record_callback        
        # if save_rollout_flag: rollout_record_callback._save_episode()

        # # check env_refresh_interval #
        # if ( trial_num % env_refresh_interval ) == ( env_refresh_interval - 1 ):
        #     # close env #
        #     env.close()

    # close agent #
    # #
    if 'prism-paligemma' in checkpoints:
        agent.stop()

    # #
    return True

def multi_evaluate(args):
    import os
    from pathlib import Path
    
    model_ref_name = args.checkpoints.split('/')[-1]
    if "checkpoint" in model_ref_name:
        checkpoint_num = model_ref_name.split("-")[-1]
        model_base_name = args.checkpoints.split('/')[-2]
        model_ref_name = f"{model_base_name}-{checkpoint_num}"
    
    # video_fold  = os.path.join(args.video_main_fold, f"{model_ref_name}-{args.env_config.split('/')[-1]}") 
    video_fold  = os.path.join(args.video_main_fold, f"{model_ref_name}-{args.env_config.split('/')[-1]}-t={args.temperature:.2f}-p={args.top_p:.2f}") 
    if not os.path.exists(video_fold):
        Path(video_fold).mkdir(parents=True,exist_ok=True)

    rollout_fold = os.path.join( args.rollout_main_fold, "-" + args.env_config.split('/')[-1] ) if args.rollout_main_fold != "" else ""
    
    model_config = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        history_num = args.history_num,
        instruction_type = args.instruction_type,
        action_chunk_len = args.action_chunk_len,
    )
    environment_config = dict(
        env_config = args.env_config,
        max_frames = args.max_frames,
        verbos = args.verbos,
    )
    
    video_log_path = os.path.join(video_fold,"end.json") 
    results = file_utils.load_json_file(video_log_path,data_type="list")

    total_ids = [i for i in range(args.workers)]
    done_ids = [result[2] for result in results]
    undone_ids = [id for id in total_ids if id not in done_ids]
    if not undone_ids:
        return
    print(f"undone_ids : {undone_ids}")

    # #
    evaluate(
        results=results,
        member_ids=undone_ids,
        video_path=video_fold,
        rollout_path=rollout_fold,
        checkpoints=args.checkpoints,
        environment_config=environment_config,
        base_url=args.base_url,
        model_config=model_config
        )

    # #
    draw_utils.show_success_rate(results,os.path.join(video_fold,"image.png") )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1) 
    # parser.add_argument('--split-number', type=int, default=6) 
    parser.add_argument('--env-config',"-e", type=str, default='craft/craft_bread') #vpt/test_vpt
    parser.add_argument('--max-frames', type=int, default=200) #vpt/test_vpt
    parser.add_argument('--verbos', type=bool, default=False)
    parser.add_argument('--checkpoints', type=str, default="/public/models/qwen2-vl-7b-instruct/")
    parser.add_argument('--device',type=str,default="cuda:1")
    
    parser.add_argument('--base-url',type=str, default="")
    parser.add_argument('--video-main-fold',type=str)
    parser.add_argument('--rollout-main-fold',type=str, default="")
    
    parser.add_argument('--instruction-type',type=str,default='normal')
    parser.add_argument('--temperature','-t',type=float,default=0.7)
    parser.add_argument('--top-p','-p',type=float,default=0.99)
    parser.add_argument('--history-num',type=int,default=0)
    parser.add_argument('--action-chunk-len',type=int,default=1)

    args = parser.parse_args()
    
    # #
    multi_evaluate(args)