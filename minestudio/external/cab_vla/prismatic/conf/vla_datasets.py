"""
datasets.py

Draccus Dataclass Definition for a DatasetConfig object, with various registered subclasses for each dataset variant
and processing scheme. A given dataset variant (e.g., `llava-lightning`) configures the following attributes:
    - Dataset Variant (Identifier) --> e.g., "llava-v15"
    - Align Stage Dataset Components (annotations, images)
    - Finetune Stage Dataset Components (annotations, images)
    - Dataset Root Directory (Path)
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class VLA_DatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                 # Unique ID that fully specifies a dataset variant
    dataset_sampling_weight: float
    dataset_labels_weight: float
    dataset_labels_type: int

    # Video Sampling Config
    sample_n_frames: int
    sample_stride: int
    sample_size: Tuple[int, int]

    # Path to dataset root directory; others paths are relative to root
    dataset_root_dir: Path                          
    # fmt: on

#--------------------------------------------------------------------------------------------
# Contractor_Action_Config
@dataclass
class Contractor_Action_Config(VLA_DatasetConfig):
    dataset_id: str = "contractor_action"
    dataset_sampling_weight: float = 1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 0

    dataset_components: Tuple[Path, Path, Path, Path, Path] = (
        Path('minestudio-data/minestudio-data-6xx-v110' ), 
        Path('minestudio-data/minestudio-data-7xx-v110' ), 
        Path('minestudio-data/minestudio-data-8xx-v110' ), 
        Path('minestudio-data/minestudio-data-9xx-v110' ), 
        Path('minestudio-data/minestudio-data-10xx-v110'), 
    )
    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        ('minecraft.kill_entity:.*',                                                    -1),
        ('minecraft.craft_item:.*',                                                     -1),
        ('minecraft.mine_block:.*(_log|_wood|cobblestone$|blackstone$|iron|diamond)', 1000),
        ('minecraft.mine_block:.*',                                                    250),
    )
    min_within: int                 = -1
    win_len: int                    = 216
    win_bias: int                   =  54
    
    sample_n_frames: int            = 10 # 10
    sample_stride: int              =  1 #  6
    sample_size: Tuple[int, int]    = (224, 224)#(360,640)

    video_start_idx: int    =  0
    video_end_idx:int       =  4 # 8
    action_start_idx: int   =  ( 4-1) * 1 # ( 4-1) * 1
    action_end_idx:int      =  (10-1) * 1 # (10-1) * 6
    action_stride: int      =  1

    dataset_root_dir: Path = Path("/db")

# Contractor_Combat_Action_Config
@dataclass
class Contractor_Combat_Action_Config(Contractor_Action_Config):
    dataset_id: str = "contractor_combat_action"
    dataset_sampling_weight: float = 1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 0

    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        ('minecraft.kill_entity:minecraft.(zombie$|spider$|skeleton$|creeper$|cow$|sheep$|chicken$|pig$)', 2000),
        # ('minecraft.kill_entity:.*', 2000),
    )
    min_within: int                 = 100 #500

# Contractor_Mine_Action_Config
@dataclass
class Contractor_Mine_Action_Config(Contractor_Action_Config):
    dataset_id: str = "contractor_mine_action"
    dataset_sampling_weight: float = 1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 0

    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        ('minecraft.mine_block:minecraft.(oak_log$|dark_oak_log$|birch_log$|spruce_log$|stone$|cobblestone$|coal_ore$|iron_ore$|gold_ore$|diamond_ore$|redstone_ore$|dirt$|sand$|obsidian$)', 2000),
        # ('minecraft.mine_block:.*', 2000),
    )    
    min_within: int                 = 100 #500

# Contractor_Craft_Action_Config
@dataclass
class Contractor_Craft_Action_Config(Contractor_Action_Config):
    dataset_id: str = "contractor_craft_action"
    dataset_sampling_weight: float = 1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 0

    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        # ('minecraft.craft_item:.*', 2000),
        ("minecraft.craft_item:minecraft.(\
oak_planks|spruce_planks|birch_planks|dark_oak_planks|\
crafting_table|furnace|bucket|chest|flint_and_steel|ladder|shears|torch|white_bed|\
stick|bone_meal|paper|wheat|\
iron_ingot|gold_ingot|charcoal|glass|stone|\
wooden_hoe|wooden_pickaxe|wooden_shovel|wooden_sword|wooden_axe|\
stone_hoe|stone_pickaxe|stone_shovel|stone_sword|stone_axe|\
iron_hoe|iron_pickaxe|iron_shovel|iron_sword|iron_axe|\
iron_helmet|iron_chestplate|iron_leggings|iron_boots|\
bread|baked_potato|cooked_beef|cooked_chicken|cooked_mutton|cooked_porkchop|cooked_cod|cooked_salmon)$",
        2000),
    )
    min_within: int                 = 100 #500
    win_len: int                    = 108 # 54
    win_bias: int                   =  27 # 13

# Contractor_Use_Action_Config
@dataclass
class Contractor_Use_Action_Config(Contractor_Action_Config):
    dataset_id: str = "contractor_use_action"
    dataset_sampling_weight: float = 0.5 #1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 0

    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        # ('minecraft.use_item:.*', 3000),
        ('minecraft.use_item:minecraft.(crafting_table|furnace|bucket|chest|ladder|shears|torch|white_bed|bone_meal|wheat_seeds)$', 3000),
    )
    min_within: int                 = 100 #500
    win_len: int                    = 108 # 54
    win_bias: int                   =  27 # 13

#--------------------------------------------------------------------------------------------
# Contractor_Prediction_Config
@dataclass
class Contractor_Prediction_Config(VLA_DatasetConfig):
    dataset_id: str = "contractor_prediction"
    dataset_sampling_weight: float = 1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 1

    dataset_components: Tuple[Path, Path, Path, Path, Path] = (
        Path('minestudio-data/minestudio-data-6xx-v110' ), 
        Path('minestudio-data/minestudio-data-7xx-v110' ), 
        Path('minestudio-data/minestudio-data-8xx-v110' ), 
        Path('minestudio-data/minestudio-data-9xx-v110' ), 
        Path('minestudio-data/minestudio-data-10xx-v110'), 
    )
    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        ('minecraft.kill_entity:.*',                                                    -1),
        ('minecraft.craft_item:.*',                                                     -1),
        ('minecraft.mine_block:.*(_log|_wood|cobblestone$|blackstone$|iron|diamond)', 1000),
        ('minecraft.mine_block:.*',                                                    250),
    )
    min_within: int                 = -1
    win_len: int                    = 216
    win_bias: int                   =   0
    
    sample_n_frames: int            =  4
    sample_stride: int              = 10
    sample_size: Tuple[int, int]    = (224, 224)#(360,640)

    video_start_idx: int    =  0
    video_end_idx:int       =  4
    action_start_idx: int   =  0 * 10
    action_end_idx:int      =  4 * 10
    action_stride: int      =  10

    dataset_root_dir: Path = Path("/db")

# Contractor_Combat_Prediction_Config
@dataclass
class Contractor_Combat_Prediction_Config(Contractor_Prediction_Config):
    dataset_id: str = "contractor_combat_prediction"
    dataset_sampling_weight: float = 0.5 #1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 1

    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        ('minecraft.kill_entity:minecraft.(zombie$|spider$|skeleton$|creeper$|cow$|sheep$|chicken$|pig$)', 2000),
        # ('minecraft.kill_entity:.*', 2000),
    )
    min_within: int                 = 100 #500

# Contractor_Mine_Prediction_Config
@dataclass
class Contractor_Mine_Prediction_Config(Contractor_Prediction_Config):
    dataset_id: str = "contractor_mine_prediction"
    dataset_sampling_weight: float = 0.5 #1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 1

    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        ('minecraft.mine_block:minecraft.(oak_log$|dark_oak_log$|birch_log$|spruce_log$|stone$|cobblestone$|coal_ore$|iron_ore$|gold_ore$|diamond_ore$|redstone_ore$|dirt$|sand$|obsidian$)', 2000),
        # ('minecraft.mine_block:.*', 2000),
    )    
    min_within: int                 = 100 #500

# Contractor_Craft_Prediction_Config
@dataclass
class Contractor_Craft_Prediction_Config(Contractor_Prediction_Config):
    dataset_id: str = "contractor_craft_prediction"
    dataset_sampling_weight: float = 0.5 #1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 1

    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        # ('minecraft.craft_item:.*', 2000),
        ("minecraft.craft_item:minecraft.(\
oak_planks|spruce_planks|birch_planks|dark_oak_planks|\
crafting_table|furnace|bucket|chest|flint_and_steel|ladder|shears|torch|white_bed|\
stick|bone_meal|paper|wheat|\
iron_ingot|gold_ingot|charcoal|glass|stone|\
wooden_hoe|wooden_pickaxe|wooden_shovel|wooden_sword|wooden_axe|\
stone_hoe|stone_pickaxe|stone_shovel|stone_sword|stone_axe|\
iron_hoe|iron_pickaxe|iron_shovel|iron_sword|iron_axe|\
iron_helmet|iron_chestplate|iron_leggings|iron_boots|\
bread|baked_potato|cooked_beef|cooked_chicken|cooked_mutton|cooked_porkchop|cooked_cod|cooked_salmon)$",
        2000),
    )
    min_within: int                 = 100 #500
    win_len: int                    = 108 # 54
    win_bias: int                   =   0

# Contractor_Use_Prediction_Config
@dataclass
class Contractor_Use_Prediction_Config(Contractor_Prediction_Config):
    dataset_id: str = "contractor_use_prediction"
    dataset_sampling_weight: float = 0.5 #1.0
    dataset_labels_weight: float = 1.0
    dataset_labels_type: int = 1

    event_info_list: Tuple[Tuple[str,int], Tuple[str,int], Tuple[str,int], Tuple[str,int]] = (
        # ('minecraft.use_item:.*', 3000),
        ('minecraft.use_item:minecraft.(crafting_table|furnace|bucket|chest|ladder|shears|torch|white_bed|bone_meal|wheat_seeds)$', 3000),
    )
    min_within: int                 = 100 #500
    win_len: int                    = 108 # 54
    win_bias: int                   =   0

#--------------------------------------------------------------------------------------------
# === Define a Dataset Registry Enum for Reference & Validation =>> all *new* datasets must be added here! ===
@unique
class VLA_DatasetRegistry(Enum):
    
    CONTRACTOR_ACTION           = Contractor_Action_Config
    CONTRACTOR_COMBAT_ACTION    = Contractor_Combat_Action_Config
    CONTRACTOR_MINE_ACTION      = Contractor_Mine_Action_Config
    CONTRACTOR_CRAFT_ACTION     = Contractor_Craft_Action_Config
    CONTRACTOR_USE_ACTION       = Contractor_Use_Action_Config

    CONTRACTOR_PREDICTION        = Contractor_Prediction_Config
    CONTRACTOR_COMBAT_PREDICTION = Contractor_Combat_Prediction_Config
    CONTRACTOR_MINE_PREDICTION   = Contractor_Mine_Prediction_Config
    CONTRACTOR_CRAFT_PREDICTION  = Contractor_Craft_Prediction_Config
    CONTRACTOR_USE_PREDICTION    = Contractor_Use_Prediction_Config

    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# Register Datasets in Choice Registry
for dataset_variant in VLA_DatasetRegistry:
    VLA_DatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)
