from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import gymnasium

from prismatic.action.action_mapping import CameraHierarchicalMapping, CameraParallelMapping, CameraXYParallelMapping
from prismatic.action.actions import ActionTransformer
from prismatic.action.action_head import make_action_head

#-------------------------------------------------------------------------------------------------------------
#
@dataclass
class CameraConfig:
    """Configuration for camera quantization and binning settings.

    :param camera_binsize: The size of each bin for camera quantization, default is 2.
    :param camera_maxval: The maximum value for camera quantization, default is 10.
    :param camera_mu: The mu parameter for mu-law quantization, default is 10.0.
    :param camera_quantization_scheme: The quantization scheme to use, either "mu_law" or "linear", default is "mu_law".
    """
    camera_binsize: int = 2
    camera_maxval: int = 10
    camera_mu: float = 10.0
    camera_quantization_scheme: str = "mu_law"

    def __post_init__(self):
        if self.camera_quantization_scheme not in ["mu_law", "linear"]:
            raise ValueError("camera_quantization_scheme must be 'mu_law' or 'linear'")
        
    @property
    def n_camera_bins(self):
        """The bin number of the setting.
        
        :returns: The number of camera bins.
        """
        return 2 * self.camera_maxval // self.camera_binsize + 1
    
    @property
    def action_transformer_kwargs(self):
        """Dictionary of camera settings used by an action transformer."""
        return {
            'camera_binsize': self.camera_binsize,
            'camera_maxval': self.camera_maxval,
            'camera_mu': self.camera_mu,
            'camera_quantization_scheme': self.camera_quantization_scheme,
        }

#-------------------------------------------------------------------------------------------------------------
#
class ActionProc:
    #
    def __init__(
        self,
        method: int,
        temperature: float,
        nucleus_prob: float,
    ):
        # #
        self.method = method
        print(f"ActionProc : method = {method}, temperature = {temperature}, nucleus_prob = {nucleus_prob}")

        # action_mapper #
        if   self.method == 0:
            camera_config = CameraConfig()
            self.action_mapper = CameraHierarchicalMapping(n_camera_bins = camera_config.n_camera_bins)
        elif self.method == 1:
            camera_config = CameraConfig( camera_binsize=1, camera_maxval=10, camera_mu=20, camera_quantization_scheme="mu_law" )
            self.action_mapper = CameraHierarchicalMapping(n_camera_bins = camera_config.n_camera_bins)
        elif self.method == 2:
            camera_config = CameraConfig()
            self.action_mapper = CameraParallelMapping(n_camera_bins = camera_config.n_camera_bins)
        elif self.method == 3:
            camera_config = CameraConfig( camera_binsize=1, camera_maxval=10, camera_mu=20, camera_quantization_scheme="mu_law" )
            self.action_mapper = CameraParallelMapping(n_camera_bins = camera_config.n_camera_bins)
        elif self.method == 4:
            camera_config = CameraConfig()
            self.action_mapper = CameraXYParallelMapping(n_camera_bins = camera_config.n_camera_bins)
        elif self.method == 5:
            camera_config = CameraConfig( camera_binsize=1, camera_maxval=10, camera_mu=20, camera_quantization_scheme="mu_law" )
            self.action_mapper = CameraXYParallelMapping(n_camera_bins = camera_config.n_camera_bins)
        #
        self.buttons_dim   = len(self.action_mapper.BUTTONS_COMBINATIONS )
        self.camera_dim    = len(self.action_mapper.camera_combinations  ) if hasattr(self.action_mapper, "camera_combinations"  ) else None
        self.camera_x_dim  = len(self.action_mapper.camera_x_combinations) if hasattr(self.action_mapper, "camera_x_combinations") else None
        self.camera_y_dim  = len(self.action_mapper.camera_y_combinations) if hasattr(self.action_mapper, "camera_y_combinations") else None
        print(f"buttons_dim : {self.buttons_dim}, camera_dim : {self.camera_dim}, camera_x_dim : {self.camera_x_dim}, camera_y_dim : {self.camera_y_dim}")
        
        # action_transformer #
        self.action_transformer = ActionTransformer(**camera_config.action_transformer_kwargs)

        # pi_head #
        if   self.method <= 3:
            action_space = gymnasium.spaces.Dict({
                "camera" : gymnasium.spaces.MultiDiscrete([self.camera_dim ]), 
                "buttons": gymnasium.spaces.MultiDiscrete([self.buttons_dim]),
            })
        elif self.method <= 5:
            action_space = gymnasium.spaces.Dict({
                "camera_x" : gymnasium.spaces.MultiDiscrete([self.camera_x_dim]), 
                "camera_y" : gymnasium.spaces.MultiDiscrete([self.camera_y_dim]), 
                "buttons"  : gymnasium.spaces.MultiDiscrete([self.buttons_dim ]),
            })
        self.pi_head = make_action_head(action_space, 1, temperature=temperature, nucleus_prob=nucleus_prob)

    # #
    def action_env_to_policy(self, action):
        action = self.action_transformer.env2policy(action)
        action = self.action_mapper.from_factored(action)
        return action

    # #
    def action_policy_to_env(self, action):
        action = self.action_mapper.to_factored(action)
        action = self.action_transformer.policy2env(action)
        return action

    # #
    def get_actions_from_action_logits_list(self, action_logits_list, deterministic=False):
        # #
        if   self.method <= 3:
            pi_logits = { "buttons" : action_logits_list[0], "camera" : action_logits_list[1] }
        elif self.method <= 5:
            pi_logits = { "buttons" : action_logits_list[0], "camera_x" : action_logits_list[1], "camera_y" : action_logits_list[2] }

        # #
        action_policy = self.pi_head.sample(pi_logits, deterministic=deterministic)

        # #
        if   self.method <= 3:
            actions = torch.cat( [ action_policy['buttons'].unsqueeze(-1), action_policy['camera' ].unsqueeze(-1) ], axis=-1 )
        elif self.method <= 5:
            actions = torch.cat( [ action_policy['buttons'].unsqueeze(-1), action_policy['camera_x'].unsqueeze(-1), action_policy['camera_y'].unsqueeze(-1) ], axis=-1 )

        # #
        return actions
