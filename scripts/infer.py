import hydra
import torch
import numpy as np
import time
from collections import deque
from omegaconf import OmegaConf

from torchrl.data.tensor_specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    BoundedTensorSpec,
    TensorSpec,
)

import torch.nn as nn
from einops.layers.torch import Rearrange
from resources.learning.ppo.ppo import PPOConfig, make_mlp, Actor, IndependentNormal
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule, TensorDictModuleBase
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry

from resources.NeuFlow_v2.infer_lidar import init_neuflow


class PPOPolicy(TensorDictModuleBase):
    def __init__(self, cfg: PPOConfig, observation_spec: CompositeSpec, action_spec: CompositeSpec, reward_spec: TensorSpec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.action_dim = 3

        fake_input = observation_spec.zero()

        cnn = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128)
        )
        mlp = make_mlp([256, 256])

        self.encoder = TensorDictSequential(
            TensorDictModule(cnn, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
            TensorDictModule(mlp, ["_feature"], ["_feature"]),
        ).to(self.device)

        self.actor = ProbabilisticActor(
            TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=False
        ).to(self.device)

        self.encoder(fake_input)
        self.actor(fake_input)

    def __call__(self, tensordict: TensorDict):
        self.encoder(tensordict)
        self.actor(tensordict)
        tensordict.exclude("loc", "scale", "_feature", inplace=True)
        return tensordict



class Infer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.odom = Odometry()
        self.zpose = 2.0
        self.target_pos_np = None
        self.target_pos = None
        
        self.starttime = time.time()
        
        self.velox = 0
        self.veloy = 0
        self.veloz = 0
        self.posx = 0
        self.posy = -15
        self.posz = self.zpose
        
        self.flow_gap = 25
        self.flow_slide_window = 5
        self.vel_ref = 5.
        self.acc_ref = 10.
        
        self.auto_test = True
        self.toggle_left_right = False
        self.reach_goal_time = None

        self.dismap_image_queue = None
        self.dismap_flow_queue = None
        self.dismap_flow_size = None
        self.flow_est_model = None
        
        self.init_params()
        self.init_policy()
        self.init_neuflow()
        
        self.odom_pub = rospy.Publisher("/sim/odom", Odometry, queue_size=10)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        
        self.rayhits_sub = rospy.Subscriber('/ray2array_hits', Float32MultiArray, self.lidar_callback)   
        self.odom_sub = rospy.Subscriber("/sim/odom", Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)

    def init_params(self):
        self.lidar_range = self.cfg.task.lidar_range
        self.lidar_h_res = self.cfg.task.lidar_h_res
        self.lidar_v_res = self.cfg.task.lidar_v_res
        self.lidar_h_sample = self.cfg.task.lidar_h_sample
        self.lidar_v_sample = self.cfg.task.lidar_v_sample
        self.bound_h = self.cfg.task.bound_h
        self.lidar_resolution = (self.lidar_h_res, self.lidar_v_res)
        self.flow_gap = self.cfg.task.flow_gap
        self.flow_slide_window = self.cfg.task.flow_slide_window
        self.num_envs = torch.tensor(1).to('cuda:0')

    def init_policy(self):
        batch_size = 1
        
        observation_spec_ = CompositeSpec(
            state=UnboundedContinuousTensorSpec(
                shape=(batch_size, 9), device="cuda:0", dtype=torch.float32
            ),
            lidar=UnboundedContinuousTensorSpec(
                shape=(batch_size, 3, 36, 6), device="cuda:0", dtype=torch.float32
            ),
            device="cuda:0",
        )

        intrinsics_spec = CompositeSpec(
            mass=UnboundedContinuousTensorSpec(
                shape=(batch_size, 1), device="cuda:0", dtype=torch.float32
            ),
            inertia=UnboundedContinuousTensorSpec(
                shape=(batch_size, 3), device="cuda:0", dtype=torch.float32
            ),
            com=UnboundedContinuousTensorSpec(
                shape=(batch_size, 3), device="cuda:0", dtype=torch.float32
            ),
            KF=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4), device="cuda:0", dtype=torch.float32
            ),
            KM=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4), device="cuda:0", dtype=torch.float32
            ),
            tau_up=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4), device="cuda:0", dtype=torch.float32
            ),
            tau_down=UnboundedContinuousTensorSpec(
                shape=(batch_size, 4), device="cuda:0", dtype=torch.float32
            ),
            drag_coef=UnboundedContinuousTensorSpec(
                shape=(batch_size, 1), device="cuda:0", dtype=torch.float32
            ),
            device="cuda:0",
        )

        stats_spec = CompositeSpec(
            **{
                "return": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device="cuda:0", dtype=torch.float32
                ),
                "episode_len": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device="cuda:0", dtype=torch.float32
                ),
                "action_smoothness": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device="cuda:0", dtype=torch.float32
                ),
                "safety": UnboundedContinuousTensorSpec(
                    shape=(batch_size, 1), device="cuda:0", dtype=torch.float32
                ),
            },
            device="cuda:0",
        )
        
        self.drone_intrinsics_spec_ = intrinsics_spec.zero().to("cuda:0")
        
        base_env_observation_spec = CompositeSpec(
            agents=CompositeSpec(
                observation=observation_spec_,
                intrinsics=intrinsics_spec,
                device="cuda:0",
            ),
            stats=stats_spec,
            device="cuda:0",
        )

        action_shape = torch.Size([128, 3])
        low_bound = torch.full(action_shape, -1.0, device="cuda:0", dtype=torch.float32)
        high_bound = torch.full(action_shape, 1.0, device="cuda:0", dtype=torch.float32)

        action_spec = BoundedTensorSpec(
            shape=action_shape,
            minimum=low_bound,
            maximum=high_bound,
            device="cuda:0",
            dtype=torch.float32,
            domain="continuous"
        )
        reward_shape = torch.Size([128, 1])
        reward_spec = UnboundedContinuousTensorSpec(
            shape=reward_shape,
            device="cuda:0",
            dtype=torch.float32,
            domain="continuous"
        )
        
        try:
            OmegaConf.register_new_resolver("eval", eval)
        except Exception:
            pass
        OmegaConf.resolve(self.cfg)
        OmegaConf.set_struct(self.cfg, False)
        
        self.stats = stats_spec.zero()
        self.observation_spec = base_env_observation_spec
        
        self.policy = PPOPolicy(
            self.cfg.algo,
            self.observation_spec,
            action_spec,
            reward_spec,
            device=self.device
        )
        
        ckt_path = "../models/p2m_default.pt"
        checkpoint = torch.load(ckt_path)
        
        filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                                if not any(prefix in k for prefix in ['critic.', 'gae.', 'value_norm.'])}
        
        self.policy.load_state_dict(filtered_checkpoint, strict=False)
        self.policy.eval()
        
        self.tensordict = self.observation_spec.zero()
        with torch.no_grad():
            self.policy.encoder(self.tensordict)
            self.policy.actor(self.tensordict)

    def init_neuflow(self):
        self.dismap_flow_size = (96, 16)
        self.flow_est_model = init_neuflow(1, self.dismap_flow_size)

    def goal_callback(self, msg):
        z = self.zpose
        if self.auto_test:
            if self.odom.pose.pose.position.y <= 0:
                self.target_pos_np = [0, 15, z]
            else:
                self.target_pos_np = [0, -15, z]
        else:
            self.target_pos_np = [msg.pose.position.x, msg.pose.position.y, z]

    def odom_callback(self, msg):
        self.odom= msg
        
        if self.auto_test:
            target_pos_left = np.array([0, -15, self.zpose])
            target_pos_right = np.array([0, 15, self.zpose])
            current_pos = np.array([self.odom.pose.pose.position.x,
                                    self.odom.pose.pose.position.y,
                                    self.odom.pose.pose.position.z])
            left_dis = np.linalg.norm(current_pos - target_pos_left)
            right_dis = np.linalg.norm(current_pos - target_pos_right)
            reach_goal_dis = 1.0
            wait_goal_time = 3.
            
            if (((right_dis <= reach_goal_dis and (not self.toggle_left_right)) |
                (left_dis <= reach_goal_dis and self.toggle_left_right))
                and (self.reach_goal_time is None)):
                self.reach_goal_time = rospy.Time.now().to_sec()
                
            if self.reach_goal_time is not None:
                now = rospy.Time.now().to_sec()
                if now - self.reach_goal_time >= wait_goal_time:
                    self.toggle_left_right = not self.toggle_left_right
                    auto_test_goal_msg = PoseStamped()
                    self.goal_pub.publish(auto_test_goal_msg)
                    self.reach_goal_time = None

    def lidar_callback(self, msg):
        if (self.target_pos_np is None):
            return
        
        tensordict = self.prepare_input(msg)
        
        with torch.no_grad():
            self.policy.encoder(tensordict)
            self.policy.actor(tensordict)
            actions = tensordict[("agents", "action")]
        self.acccmd_2_odom(actions)

    def prepare_input(self, msg):
        data = np.array(msg.data)
        reshaped_data = data.reshape(1, self.lidar_h_res*self.lidar_v_res*self.lidar_h_sample*self.lidar_v_sample, 3)
        ray_hits_w = torch.tensor(reshaped_data,dtype=torch.float32).to('cuda:0')
        
        drone_state = [self.odom.pose.pose.position.x,
                       self.odom.pose.pose.position.y,
                       self.odom.pose.pose.position.z,
                       self.odom.twist.twist.linear.x,
                       self.odom.twist.twist.linear.y,
                       self.odom.twist.twist.linear.z]
        drone_state = torch.tensor(np.array(drone_state).reshape(1, 6),dtype=torch.float32).unsqueeze(1).to('cuda:0')
        
        self.target_pos_np = np.array(self.target_pos_np).reshape(1, 3)
        self.target_pos = torch.tensor(self.target_pos_np,dtype=torch.float32).unsqueeze(1).to('cuda:0')
        
        pos_w_z = drone_state[..., 2].squeeze(1)
        ray_hits_w_z = ray_hits_w[:, :, 2] 
        
        z_in_range = (ray_hits_w_z >= (pos_w_z - self.bound_h)) & (ray_hits_w_z <= (pos_w_z + 2*self.bound_h))
        distances = (ray_hits_w - drone_state[..., :3]).norm(dim=-1)
        lidar_dis = torch.where(z_in_range, distances, torch.full_like(distances, self.lidar_range))
        lidar_dis = lidar_dis.clamp(0, self.lidar_range)
        
        lidar_dis_unfold = lidar_dis.reshape(
            self.num_envs, self.lidar_h_res * self.lidar_h_sample, self.lidar_v_res * self.lidar_v_sample
        ).unfold(1, self.lidar_h_sample, self.lidar_v_sample).unfold(2, self.lidar_h_sample, self.lidar_v_sample) 
        lidar_scan_raw, _ = lidar_dis_unfold.reshape(
            self.num_envs, 1, self.lidar_h_res * self.lidar_v_res, self.lidar_h_sample * self.lidar_v_sample
        ).min(dim=-1)
        lidar_scan = self.lidar_range - (
            lidar_scan_raw
            .clamp_max(self.lidar_range)
            .reshape(1, 1, *self.lidar_resolution)
        )
        
        rpos = self.target_pos - drone_state[..., :3]
        target_dir = rpos / rpos.norm(dim=-1, keepdim=True).clamp(1e-6)
        vel_fb = (drone_state[..., 3:])
        vel_input = vel_fb/self.vel_ref
        
        actions = self.tensordict[("agents", "action")]
        if actions is not None:
            acc_fb = actions.unsqueeze(1)
        else:
            acc_fb = torch.zeros_like(vel_fb)
        acc_input = acc_fb/self.acc_ref
        
        scan4flow = (self.lidar_range - lidar_dis.unsqueeze(1)).reshape(
                            1, 1, 
                            self.lidar_h_res * self.lidar_h_sample,  
                            self.lidar_v_res * self.lidar_v_sample
                            )/self.lidar_range
        
        scan4flow_scaled = torch.nn.functional.interpolate(scan4flow.half() * 255., 
                                                            self.dismap_flow_size, 
                                                            mode='bilinear', 
                                                            align_corners=False)

        if self.dismap_image_queue is None:
            self.dismap_image_queue = deque([scan4flow_scaled] * int(self.flow_gap + 3), maxlen=int(self.flow_gap + 3))
        else:
            self.dismap_image_queue.append(scan4flow_scaled)

        dismap_tensor = list(self.dismap_image_queue)
        dismap_image0 = torch.cat(dismap_tensor[:3], dim=1)
        dismap_image1 = torch.cat(dismap_tensor[-3:], dim=1)

        with torch.no_grad():
            dismap_image0 = dismap_image0.half()
            dismap_image1 = dismap_image1.half()
            dismap_flow = self.flow_est_model(dismap_image0, dismap_image1)[-1]
        
        if self.dismap_flow_queue is None:
            self.flow_slide_window = int(self.flow_slide_window)
            self.dismap_flow_queue = deque([dismap_flow] * self.flow_slide_window, maxlen=self.flow_slide_window)
        else:
            self.dismap_flow_queue.append(dismap_flow)
        dismap_flow_mean = torch.mean(torch.stack(list(self.dismap_flow_queue)), dim=0)
        dismap_flow_scaled = torch.nn.functional.interpolate(dismap_flow_mean.float(), 
                                                                    self.lidar_resolution, 
                                                                    mode='bilinear', 
                                                                    align_corners=False)

        flow_normalized = torch.cat([dismap_flow_scaled[:, 0, :, :].unsqueeze(1)/3.6,
                                     dismap_flow_scaled[:, 1, :, :].unsqueeze(1)/0.6,], dim=1)
        
        scan_normalized = lidar_scan/self.lidar_range
        dismap_stack = torch.cat([scan_normalized, flow_normalized], dim=1)

        obs = {
            "state": torch.cat([target_dir, vel_input, acc_input], dim=-1).squeeze(1),
            "lidar": dismap_stack
        }
        
        self.tensordict = TensorDict(
                {
                    "agents": TensorDict(
                        {
                            "observation": obs,
                            "intrinsics": self.drone_intrinsics_spec_,
                        },
                        [self.num_envs],
                    ),
                    "stats": self.stats.clone(),
                },
                1,
            )
        return self.tensordict

    def acccmd_2_odom(self, actions):
        dt = 0.02
        self.velox = self.velox + actions[0][0]*dt
        self.veloy = self.veloy + actions[0][1]*dt
        self.veloz = self.veloz + actions[0][2]*dt
        
        if (time.time() - self.starttime >= 1):
            self.posx = self.posx + self.velox*dt
            self.posy = self.posy + self.veloy*dt
            self.posz = self.posz + self.veloz*dt
        
        odom = Odometry()
        odom.twist.twist.linear.x = self.velox
        odom.twist.twist.linear.y = self.veloy
        odom.twist.twist.linear.z = self.veloz
        odom.pose.pose.position.x = self.posx
        odom.pose.pose.position.y = self.posy
        odom.pose.pose.position.z = self.posz
        odom.pose.pose.orientation.x = 0
        odom.pose.pose.orientation.y = 0
        odom.pose.pose.orientation.z = 0
        odom.pose.pose.orientation.w = 1
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "/world"
        odom.child_frame_id  = "/quadrotor"
        self.odom_pub.publish(odom)


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(cfg):
    rospy.init_node('infer')
    node = Infer(cfg)
    rospy.spin()

if __name__ == "__main__":
    main()