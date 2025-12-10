import torch
import torch.distributions as D
import einops
from collections import deque
from . import env_utils

from resources.envs.isaac_env import AgentSpec, IsaacEnv
from resources.robots.drone import MultirotorBase
from resources.utils.torch import euler_to_quaternion
from tensordict.tensordict import TensorDict, TensorDictBase
from resources.NeuFlow_v2.infer_lidar import init_neuflow
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.lab.assets import RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg


class Env(IsaacEnv):
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.bound_h = cfg.task.bound_h

        super().__init__(cfg, headless)

        self.lidar._initialize_impl()

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        self.safety_dis = 0.3
        self.vel_min = 3.5
        self.vel_max = 5.
        self.acc_min = 0.
        self.acc_max = 10.
        self.virtual_ground = 0.5
        self.virtual_ceiling = 3.5
        self.height_bound = 0.5

        self.start_pos = None
        self.target_pos = None
        self.dismap_flow_size = None
        self.actions = None
        self.last_dis2goal = None
        self.last_acc = None
        self.dismap_image_queue = None
        self.dismap_flow_queue = None
        self.set_dobs_state = None
        self.set_wall_state = None
        self.ray_hits_dir = None
        self.input_dir = None
        self.error_tolerance = None
        self.flow_est_model = None
        self.dismap_flow = None
        self.trace_prob = None
        self.virtual_x_bound = None
        self.reward_dobs_max = None

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.assets import AssetBaseCfg
        from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
        from omni.isaac.lab.terrains import (
            TerrainImporterCfg,
            TerrainImporter,
            TerrainGeneratorCfg,
            HfDiscreteObstaclesTerrainCfg,
        )
        # from omni.isaac.lab.utils.assets import NVIDIA_NUCLEUS_DIR

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        rot = euler_to_quaternion(torch.tensor([0., 0.1, 0.1]))
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos, rot)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        self.seed = 10
        self.static_obs_num_per_gird = self.cfg.task.static_obs_num_per_gird
        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=self.seed,
                size=(6.0, 6.0),
                border_width=20.0,
                num_rows=6,
                num_cols=6,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        size=(6.0, 6.0),
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.static_obs_num_per_gird,
                        obstacle_height_mode="fixed",
                        obstacle_width_range=(0.5, 0.9),
                        obstacle_height_range=(3.5, 4.0),
                        platform_width=1.5,
                    )
                },
            ),
            max_init_terrain_level=5,
            collision_group=-1,
            debug_vis=False,
        )
        terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)

        self.dynamic_obs_num = self.cfg.task.dynamic_obs_num
        self.dobs_pos_x_range = (-18.0, 18.0)
        self.dobs_pos_y_range = (-18.0, 18.0)
        self.dobs_vel_range = (1.0, 5.0)
        self.dobs_rad_range = (0.25, 0.45)
        self.dobs_height = 4.
        self.dobs_states = env_utils.generate_obstacle_tensor(self.dynamic_obs_num, 
                                                         self.dobs_pos_x_range,
                                                         self.dobs_pos_y_range, 
                                                         self.dobs_vel_range, 
                                                         self.dobs_rad_range,
                                                         self.seed)
        self.dobs_origins = self.dobs_states[:, 0]
        self.dobs_rad = self.dobs_states[:, 2][:, 0]
        dobs_cfg_dict = {}
        for i, origin in enumerate(self.dobs_origins):
            cylinder_cfg = RigidObjectCfg(
                prim_path=f"/World/moving_obs{i}/Cylinder",
                spawn=sim_utils.CylinderCfg(
                    radius=self.dobs_rad[i],
                    height=self.dobs_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False),
                ),
                init_state=RigidObjectCfg.InitialStateCfg()
            )
            dobs_cfg_dict[f"Cylinder_{i}"] = cylinder_cfg
        cylinder_collection_cfg = RigidObjectCollectionCfg(rigid_objects = dobs_cfg_dict)
        self.dobs = RigidObjectCollection(cfg = cylinder_collection_cfg)
        self.dobs_states = torch.tensor(self.dobs_states, device=self.device)
        self.dobs_origins = torch.tensor(self.dobs_origins, device=self.device)

        self.wall_num = 1
        self.wall_width = 20
        self.wall_height = 0.4
        self.fly_height = 2.0
        self.wall_states = env_utils.generate_wall_tensor(self.wall_num, self.wall_width, 
                                                          self.wall_height, self.fly_height)
        self.wall_origins = self.wall_states[..., :3]
        self.wall_sizes = self.wall_states[..., 3:]
        wall_cfg_dict = {}
        for i, origin in enumerate(self.wall_origins):
            for j in range(4):
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Wall{i}/Cuboid{j}",
                    spawn=sim_utils.CuboidCfg(
                        size=self.wall_sizes[i, j, :],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            collision_enabled=False),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg()
                )
                wall_cfg_dict[f"Wall_{4*i+j}"] = cuboid_cfg
        wall_collection_cfg = RigidObjectCollectionCfg(rigid_objects = wall_cfg_dict)
        self.wall = RigidObjectCollection(cfg = wall_collection_cfg)
        self.wall_states = torch.tensor(self.wall_states, device=self.device)
        self.wall_origins = torch.tensor(self.wall_origins, device=self.device)
        self.wall_sizes = torch.tensor(self.wall_sizes, device=self.device)

        self.lidar_hfov = self.cfg.task.lidar_hfov
        self.lidar_vfov = (
            max(-89., self.cfg.task.lidar_vfov[0]),
            min(89., self.cfg.task.lidar_vfov[1])
        )
        self.lidar_range = self.cfg.task.lidar_range
        self.lidar_h_res = self.cfg.task.lidar_h_res
        self.lidar_v_res = self.cfg.task.lidar_v_res
        self.lidar_h_sample = self.cfg.task.lidar_h_sample
        self.lidar_v_sample = self.cfg.task.lidar_v_sample

        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res = self.lidar_hfov / (self.lidar_h_res * self.lidar_h_sample),
                vertical_ray_angles = torch.linspace(*self.lidar_vfov, self.lidar_v_res * self.lidar_v_sample)
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"]
        )
        self.lidar: RayCaster = ray_caster_cfg.class_type(ray_caster_cfg)
        return ["/World/ground"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 9
        self.lidar_resolution = (self.lidar_h_res, self.lidar_v_res)

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
                    "lidar": UnboundedContinuousTensorSpec((3, self.lidar_resolution[0], self.lidar_resolution[1]), device=self.device),
                }),
                "intrinsics": self.drone.intrinsics_spec.to(self.device)
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec,
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "reward_velocity": UnboundedContinuousTensorSpec(1),
            "reward_acceleration": UnboundedContinuousTensorSpec(1),
            "reward_jerk": UnboundedContinuousTensorSpec(1),
            "reward_height": UnboundedContinuousTensorSpec(1),
            "reward_goal": UnboundedContinuousTensorSpec(1),
            "reward_safety": UnboundedContinuousTensorSpec(1),
            "reward_dobs": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec 
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

        if (self.start_pos is None) & (self.target_pos is None):
            drones_per_side = self.cfg.env.num_envs // 4
            in_max = 20
            out_max = 44
            offset = 12
            left_vals = torch.linspace(-0.5, 0, int(drones_per_side/2), device=self.device) * in_max
            right_vals = torch.linspace(0, 0.5, int(drones_per_side/2), device=self.device) * in_max
            vals = torch.cat([right_vals, left_vals], dim=0)

            self.start_pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            self.start_pos[:drones_per_side, 0, 0] = vals
            self.start_pos[:drones_per_side, 0, 1] = - out_max/2
            self.start_pos[drones_per_side:2*drones_per_side, 0, 0] = out_max/2
            self.start_pos[drones_per_side:2*drones_per_side, 0, 1] = vals
            self.start_pos[2*drones_per_side:3*drones_per_side, 0, 0] = - out_max/2
            self.start_pos[2*drones_per_side:3*drones_per_side, 0, 1] = vals
            self.start_pos[3*drones_per_side:, 0, 0] = vals
            self.start_pos[3*drones_per_side:, 0, 1] = out_max/2
            self.start_pos[:, 0, 2] = self.fly_height

            self.target_pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            self.target_pos[:drones_per_side, 0, 0] = - vals  
            self.target_pos[:drones_per_side, 0, 1] = out_max/2 - offset
            self.target_pos[drones_per_side:2*drones_per_side, 0, 0] = - out_max/2 + offset  
            self.target_pos[drones_per_side:2*drones_per_side, 0, 1] = - vals
            self.target_pos[2*drones_per_side:3*drones_per_side, 0, 0] = out_max/2 - offset  
            self.target_pos[2*drones_per_side:3*drones_per_side, 0, 1] = - vals 
            self.target_pos[3*drones_per_side:, 0, 0] = - vals
            self.target_pos[3*drones_per_side:, 0, 1] = - out_max/2 + offset
            self.target_pos[:, 0, 2] = self.fly_height

        pos = self.start_pos[env_ids]
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos, rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        if self.set_dobs_state is None:
            self.set_dobs_state = self.dobs.data.default_object_state.clone()
            self.set_dobs_state[..., :2] = self.dobs_origins
            self.set_dobs_state[..., 2] = self.dobs_height/2
        else:
            if self.trace_prob is None:
                self.trace_prob = self.cfg.task.trace_prob
            self.dobs_states[:, 1] = env_utils.update_dobs_vel(self.device, self.dobs_states, 
                                                               self.dobs_pos_x_range, self.dobs_pos_y_range, 
                                                               self.drone, self.trace_prob)
            set_dobs_vel = self.dobs_states[:, 1]
            self.set_dobs_state[..., :2] = self.dobs_states[:, 0] + (set_dobs_vel * self.dt)
            self.set_dobs_state[..., 2] = self.dobs_height/2
            self.dobs_states[:, 0] = self.set_dobs_state[..., :2]
        self.dobs.write_object_link_pose_to_sim(self.set_dobs_state[..., :7])

        if self.set_wall_state is None:
            self.set_wall_state = self.wall.data.default_object_state.clone()
            self.set_wall_state[..., :3] = self.wall_origins.reshape(self.wall_num * 4, 3)
            self.wall.write_object_link_pose_to_sim(self.set_wall_state[..., :7])

        self.actions = tensordict[("agents", "action")]
        ego_drone_state = self.drone.get_state(env_frame=False)[..., :13].squeeze(0)
        unit_thrust = self.controller(ego_drone_state, self.actions.unsqueeze(1), None, False)
        self.effort = self.drone.apply_action(unit_thrust)

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.lidar.update(self.dt)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        self.rpos = self.target_pos - self.drone_state[..., :3]

        if (self.ray_hits_dir is None) & (self.input_dir is None):
            self.ray_hits_dir = env_utils.compute_rayhitsdir(self.device, self.num_envs, self.lidar_hfov, self.lidar_vfov,
                                                         self.lidar_h_res * self.lidar_h_sample,
                                                         self.lidar_v_res * self.lidar_v_sample)
            self.input_dir = env_utils.compute_rayhitsdir(self.device, self.num_envs, self.lidar_hfov, self.lidar_vfov,
                                                      self.lidar_h_res, self.lidar_v_res)

        self.dobs_hits_w = env_utils.dobs_lidar_hits(
            self.lidar_range,
            self.dobs_height,
            self.dobs_states,
            self.lidar.data.pos_w,
            self.ray_hits_dir,
            error_tolerance=0.33
        )

        self.wall_hits_w = env_utils.wall_lidar_hits(
            self.lidar_range,
            self.fly_height,
            self.wall_sizes,
            self.lidar.data.pos_w,
            self.ray_hits_dir
        )

        self.merged_hits = env_utils.merge_hit_points(
            self.dobs_hits_w,
            self.wall_hits_w, 
            self.lidar.data.ray_hits_w,
            self.lidar.data.pos_w
        )

        pos_w_z = self.lidar.data.pos_w[:, 2].unsqueeze(1)
        ray_hits_w_z = self.merged_hits[:, :, 2]
        z_in_range = (ray_hits_w_z >= (pos_w_z - self.bound_h)) & (ray_hits_w_z <= (pos_w_z + 2*self.bound_h))
        distances = (self.merged_hits - self.lidar.data.pos_w.unsqueeze(1)).norm(dim=-1)
        lidar_dis = torch.where(
            z_in_range, distances, torch.full_like(distances, self.lidar_range)).clamp(0, self.lidar_range)

        lidar_dis_unfold = lidar_dis.reshape(
            self.num_envs, self.lidar_h_res * self.lidar_h_sample, self.lidar_v_res * self.lidar_v_sample
        ).unfold(1, self.lidar_h_sample, self.lidar_v_sample).unfold(2, self.lidar_h_sample, self.lidar_v_sample) 
        self.lidar_scan_raw = lidar_dis_unfold.reshape(
            self.num_envs, 1, self.lidar_h_res * self.lidar_v_res, self.lidar_h_sample * self.lidar_v_sample
        ).min(dim=-1)[0]

        self.lidar_scan = self.lidar_range - (
            self.lidar_scan_raw
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        )

        distance = self.rpos.norm(dim=-1, keepdim=True)
        target_dir = self.rpos / distance.clamp(1e-6)
        vel_fb = (self.drone_state[..., 7:])[..., :3]
        vel_input = vel_fb/self.vel_max
        if self.actions is not None:
            acc_fb = self.actions.unsqueeze(1)
        else:
            acc_fb = torch.zeros_like(vel_fb)
        acc_input = acc_fb/self.acc_max

        if self.dismap_flow_size is None:
            self.dismap_flow_size = (96, 16)
        if self.flow_est_model is None:
            self.flow_est_model = init_neuflow(self.num_envs, 
                                               self.dismap_flow_size)
        self.scan4flow = (
            self.lidar_range - lidar_dis.unsqueeze(1)).reshape(
                self.num_envs, 1, 
                self.lidar_h_res * self.lidar_h_sample,  
                self.lidar_v_res * self.lidar_v_sample
            ) / self.lidar_range
        scan4flow_scaled = torch.nn.functional.interpolate(self.scan4flow.half() * 255., 
                                                            self.dismap_flow_size, 
                                                            mode='bilinear', 
                                                            align_corners=False)
        if self.dismap_image_queue is None:
            self.dismap_image_queue = deque([scan4flow_scaled] * int(self.cfg.task.flow_gap + 3), 
                                            maxlen=int(self.cfg.task.flow_gap + 3))
        else:
            self.dismap_image_queue.append(scan4flow_scaled)
        dismap_tensor = list(self.dismap_image_queue)
        dismap_image0 = torch.cat(dismap_tensor[:3], dim=1)
        dismap_image1 = torch.cat(dismap_tensor[-3:], dim=1)
        with torch.no_grad():
            self.dismap_flow = self.flow_est_model(dismap_image0, dismap_image1)[-1]
        if self.dismap_flow_queue is None:
            flow_slide_window = int(self.cfg.task.flow_slide_window)
            self.dismap_flow_queue = deque([self.dismap_flow] * flow_slide_window, maxlen=flow_slide_window)
        else:
            self.dismap_flow_queue.append(self.dismap_flow)
        self.dismap_flow_mean = torch.mean(torch.stack(list(self.dismap_flow_queue)), dim=0)
        self.dismap_flow_zoom = torch.nn.functional.interpolate(self.dismap_flow_mean.float(), 
                                                                  self.lidar_resolution, 
                                                                  mode='bilinear', 
                                                                  align_corners=False)
        flow_scaled = torch.cat([self.dismap_flow_zoom[:, 0, :, :].unsqueeze(1)/3.6,
                                 self.dismap_flow_zoom[:, 1, :, :].unsqueeze(1)/0.6], 
                                 dim=1)
        scan_normalized = self.lidar_scan/self.lidar_range

        self.dismap_stack = torch.cat([scan_normalized, flow_scaled], dim=1)

        obs = {
            "state": torch.cat([target_dir, vel_input, acc_input], dim=-1).squeeze(1),
            "lidar": self.dismap_stack
        }         

        if (self._should_render(0)) & (self.cfg.task.vis_lidar):
            ray_dis = (self.lidar_range - self.lidar_scan).reshape(self.num_envs, self.lidar_h_res * self.lidar_v_res)
            rayhits = self.lidar.data.pos_w.unsqueeze(1) + ray_dis.unsqueeze(-1) * self.input_dir  
            self.debug_draw.clear()
            x = self.lidar.data.pos_w[0]
            set_camera_view(
                eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )
            v = (rayhits[0] - x).reshape(*self.lidar_resolution, 3)
            self.debug_draw.vector(x.expand_as(v[:, 1]), v[:, 1])

        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs,
                        "intrinsics": self.drone.intrinsics,
                    },
                    [self.num_envs],
                ),
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        distance = self.rpos.norm(dim=-1, keepdim=True)
        dis2goal = distance.squeeze(-1)
        height = self.drone_state[..., 2]
        touch_goal_mask = dis2goal <= 3.
        vel_direction = self.rpos / distance.clamp_min(1e-6)
        vel_magnitude = self.drone.vel_w[..., :3].norm(dim=-1)
        acc = self.actions
        acc_magnitude = acc.norm(dim=-1, keepdim=True)
        if self.last_acc is None:
            self.last_acc = acc
        if self.last_dis2goal is None:
            self.last_dis2goal = dis2goal
        else:
            new_rollout_mask = ((self.last_dis2goal - dis2goal) > 0.1) | ((self.last_dis2goal - dis2goal) < 0)
            self.last_dis2goal[new_rollout_mask] = dis2goal[new_rollout_mask]

        [virtual_ground, virtual_ceiling] = [self.virtual_ground, self.virtual_ceiling]
        [beta_vel, beta_acc, vel_limit, acc_limit] = [2., 5., 1.2 * self.vel_max, 1.5 * self.acc_max]
        [beta_hei, hei_set_min, hei_set_max] = [2., self.fly_height - self.height_bound/2, self.fly_height + self.height_bound/2]
        [vel_set_min, vel_set_max, acc_set_min, acc_set_max] = [self.vel_min, self.vel_max, self.acc_min, self.acc_max]
        [k_v, k_a, k_j, k_h, k_g, k_s, k_d] = [1.2, 0.6, 0.2, 0.3, 0.8, 1.0, 0.6]

        reward_vel, reward_acc, reward_jerk, reward_height = self._compute_state_reward(
            beta_vel, vel_set_min, vel_set_max, vel_magnitude, beta_acc, acc_set_min, acc_set_max, acc_magnitude,
            beta_hei, hei_set_min, hei_set_max, height, acc, self.last_acc, touch_goal_mask)
        reward_goal = self._compute_goal_reward(self.drone.vel_w[..., :3], vel_direction,
                                                self.last_dis2goal, dis2goal,
                                                touch_goal_mask)
        reward_safety = self._compute_safety_reward(self.lidar_scan)
        reward_dobs = self._compute_dobs_reward(self.dobs_states, 
                                                self.drone_state[..., :2].squeeze(1),
                                                self.drone.vel_w[..., :2].squeeze(1))
        reward = (k_v * reward_vel +
                  k_a * reward_acc + 
                  k_j * reward_jerk + 
                  k_h * reward_height +
                  k_g * reward_goal +
                  k_s * reward_safety +
                  k_d * reward_dobs)
        
        print(f"\r \r vel: {k_v * reward_vel[0].item():.3f}, "
              f"acc: {k_a * reward_acc[0].item():.3f}, "
              f"jerk: {k_j * reward_jerk[0].item():.3f}, "
              f"goal: {k_g * reward_goal[0].item():.3f}, "
              f"height: {k_h * reward_height[0].item():.3f}, "
              f"safety: {k_s * reward_safety[0].item():.3f}, "
              f"dobs: {k_d * reward_dobs[0].item():.3f}, "
              f"total: {reward[0].item():.3f}\r", end="", flush=True)

        bound_misbehave = env_utils.get_bound_misbehave(self.drone_state[..., :2].squeeze(1), 
                                                    self.start_pos[..., :2].squeeze(1),
                                                    self.target_pos[..., :2].squeeze(1))
        misbehave = (
            (self.drone.pos[..., 2] < virtual_ground)
            | (self.drone.pos[..., 2] > virtual_ceiling)
            | (bound_misbehave)
            | (vel_magnitude > vel_limit)
            | (acc_magnitude > acc_limit)
            | ((self.lidar_range - einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max")) < self.safety_dis)
        )

        hasnan = torch.isnan(self.drone_state).any(-1)
        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["reward_velocity"].add_(reward_vel)
        self.stats["reward_acceleration"].add_(reward_acc)
        self.stats["reward_jerk"].add_(reward_jerk)
        self.stats["reward_height"].add_(reward_height)
        self.stats["reward_goal"].add_(reward_goal)
        self.stats["reward_safety"].add_(reward_safety)
        self.stats["reward_dobs"].add_(reward_dobs)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.last_acc = acc
        self.last_dis2goal = dis2goal

        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )  
    
    def _compute_state_reward(self, beta_vel, vel_set_min, vel_set_max, vel_magnitude,
                              beta_acc, acc_set_min, acc_set_max, acc_magnitude,
                              beta_hei, hei_set_min, hei_set_max, height, 
                              acc, last_acc, touch_goal_mask):
        reward_vel = torch.log(torch.exp(- beta_vel * (torch.clamp(vel_set_min - vel_magnitude, min = 0.)
                                         + torch.clamp(vel_magnitude - vel_set_max, min = 0.))) + 1.)
        reward_vel[touch_goal_mask] = torch.log(torch.exp(- beta_vel * torch.clamp(
                                         vel_magnitude[touch_goal_mask] - vel_set_max, min = 0.)) + 1.)
        reward_acc = torch.log(torch.exp(- beta_acc * (torch.clamp(acc_set_min - acc_magnitude, min = 0.)
                                         + torch.clamp(acc_magnitude - acc_set_max, min = 0.))) + 1.)
        reward_jerk = 1. / (1. + torch.norm(acc - last_acc, dim=-1, keepdim=True))
        reward_height = torch.log(torch.exp(- beta_hei * (torch.clamp(hei_set_min - height, min = 0.)
                                         + torch.clamp(height - hei_set_max, min = 0.))) + 1.)

        return reward_vel, reward_acc, reward_jerk, reward_height
    
    def _compute_goal_reward(self, vel_vector, vel_direction, last_dis2goal, dis2goal, touch_goal_mask):
        reward_goal_dir = (vel_vector * vel_direction).sum(-1).clip(max=2.0)
        reward_goal_dis = (torch.exp(last_dis2goal - dis2goal) - 1.) * 10.
        reward_goal_dis[touch_goal_mask] = 0.
        reward_goal = reward_goal_dir + reward_goal_dis

        return reward_goal

    def _compute_safety_reward(self, lidar_scan):
        lidar_values = self.lidar_range - lidar_scan
        lidar_values_merged = lidar_values.reshape(
                                lidar_values.size(0), lidar_values.size(1), -1).squeeze(1)
        lidar_values_clip = torch.clamp(lidar_values_merged - self.safety_dis, min = 0.)
        obs_mask = lidar_values_merged <= (self.lidar_range/10)
        obs_count = obs_mask.sum(dim=1)
        obs_dist = torch.where(
            obs_count != 0,
            (lidar_values_clip * obs_mask).sum(dim=1) / obs_count, 
            lidar_values_clip.min(dim=1)[0])
        reward_safety = torch.log(obs_dist).clamp_min(-5.)
        reward_safety = reward_safety.reshape(self.num_envs, 1)

        return reward_safety

    def _compute_dobs_reward(self, obstacle_tensor, drone_pos, drone_vel):
        num_env = drone_pos.shape[0]
        n = obstacle_tensor.shape[0]
        pos = obstacle_tensor[:, 0]  
        vel = obstacle_tensor[:, 1]
        rad = obstacle_tensor[:, 2, 0]
        drone_pos_expanded = drone_pos.unsqueeze(1).expand(num_env, n, 2)
        drone_vel_expanded = drone_vel.unsqueeze(1).expand(num_env, n, 2)
        pos_expanded = pos.unsqueeze(0).expand(num_env, n, 2)
        obstacle_vel_drone_frame = vel - drone_vel_expanded
        if self.reward_dobs_max is None:
            self.reward_dobs_max = torch.full((num_env, 1), float('-inf'), device=self.device)
        
        r = pos - drone_pos_expanded
        dot_product = (r * obstacle_vel_drone_frame).sum(dim=2)
        r_norm = r.norm(dim=2)
        v_norm = obstacle_vel_drone_frame.norm(dim=2)
        cos_theta = dot_product / (r_norm * v_norm)
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)
        coll_mask = theta < (torch.pi / 2)
        vel_magnitude = torch.norm(vel, dim=1)
        dist = torch.norm(pos_expanded - drone_pos_expanded, dim=2) - rad
        unit_velocity = vel / (vel_magnitude.unsqueeze(1) + 1e-6)
        unit_velocity_expanded = unit_velocity.unsqueeze(0).expand(num_env, n, 2)
        v_x = unit_velocity_expanded[..., 0]
        v_y = unit_velocity_expanded[..., 1]
        x = pos_expanded[..., 0]
        y = pos_expanded[..., 1]
        x_d = drone_pos_expanded[..., 0]
        y_d = drone_pos_expanded[..., 1]
        speed_line_distance = torch.abs((x_d - x) * v_y - (y_d - y) * v_x) 
        fov_mask = dist <= (self.lidar_range * 0.75)
        obs_count = fov_mask.sum(dim=1).clamp(min=1)
        
        k_v = torch.norm(obstacle_vel_drone_frame, dim=2)
        k_theta = 1. - (2 * theta / torch.pi)
        k_d = torch.exp(1. / (1. + speed_line_distance))
        k_total = torch.where(coll_mask != 0,
                              1 + k_v * k_theta * k_d,
                              torch.ones_like(theta))
        
        r_d_zoom = (dist- self.safety_dis).clamp_min(0.) / (k_total + 1e-6)
        r_d = torch.log(r_d_zoom).clamp_min(-5.)
        reward_dobs = ((r_d * fov_mask).sum(dim=1) / obs_count).reshape(num_env, 1)
        self.reward_dobs_max = torch.max(self.reward_dobs_max, reward_dobs)
        reward_dobs = torch.where(reward_dobs == 0, self.reward_dobs_max, reward_dobs)

        return reward_dobs