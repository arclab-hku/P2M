import torch
from torch._tensor import Tensor
import torch.nn as nn
from .controller import ControllerBase

from resources.utils.torch import (
    quat_rotate_inverse,
    normalize,
    quaternion_to_rotation_matrix,
    quaternion_to_euler,
)
import yaml
import os.path as osp
# torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device("cuda:0")

def compute_parameters(
    rotor_config,
    inertia_matrix,
):
    rotor_angles = torch.as_tensor(rotor_config["rotor_angles"], device=device)
    arm_lengths = torch.as_tensor(rotor_config["arm_lengths"], device=device)
    force_constants = torch.as_tensor(rotor_config["force_constants"], device=device)
    moment_constants = torch.as_tensor(rotor_config["moment_constants"], device=device)
    directions = torch.as_tensor(rotor_config["directions"], device=device)
    max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"], device=device)
    A = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles),
        ]
    )
    mixer = A.T @ (A @ A.T).inverse() @ inertia_matrix

    return mixer

class LeeAccelerationController(ControllerBase):
    def __init__(
        self,
        g: float,
        uav_params,
    ) -> None:
        super().__init__()
        controller_param_path = osp.join(
            osp.dirname(__file__), "cfg", f"lee_controller_{uav_params['name']}.yaml"
        )
        with open(controller_param_path, "r") as f:
            controller_params = yaml.safe_load(f)

        self.mass = nn.Parameter(torch.tensor(uav_params["mass"], device=device))
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, g], device=device).abs())
        # print(f"g device is {self.g.device}")
        # self.g = self.g.to(device)

        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]

        force_constants = torch.as_tensor(rotor_config["force_constants"], device=device)
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"], device=device)

        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)

        I = torch.diag_embed(
            torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1], device=device)
        )
        self.mixer = nn.Parameter(compute_parameters(rotor_config, I))
        self.attitute_gain = nn.Parameter(
            torch.as_tensor(controller_params["attitude_gain"], device=device).float() @ I[:3, :3].inverse()
        )
        # print(f"attitude gain device is {self.attitute_gain.device}")
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"], device=device).float() @ I[:3, :3].inverse()
        )
        self.requires_grad_(False)

    def forward(
        self,
        root_state: torch.Tensor,
        target_acc: torch.Tensor=None,
        target_yaw: torch.Tensor=None,
        body_rate: bool=False
    ):
        # print("Using Custom Acceleration controller")
        batch_shape = root_state.shape[:-1]
        # print(f"root state device is {root_state.device}")
        device = root_state.device

        if target_acc is None:
            target_acc = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_acc = target_acc.expand(batch_shape+(3,))
        if target_yaw is None:
            target_yaw = quaternion_to_euler(root_state[..., 3:7])[..., -1]
            target_yaw = torch.full_like(target_yaw, torch.pi / 4)     # tf45
        else:
            if not target_yaw.shape[-1] == 1:
                target_yaw = target_yaw.unsqueeze(-1)
            target_yaw = target_yaw.expand(batch_shape+(1,))

        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_acc.reshape(-1, 3),
            target_yaw.reshape(-1, 1),
            body_rate
        )

        return cmd.reshape(*batch_shape, -1)

    def _compute(self, root_state, target_acc, target_yaw, body_rate):
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            # convert angular velocity from world frame to body frame
            ang_vel = quat_rotate_inverse(rot, ang_vel)
        
        # print(f"root state device is {root_state.device}")
        g = self.g.to(root_state.device)
        mass = self.mass.to(root_state.device)
        attitute_gain = self.attitute_gain.to(root_state.device)
        ang_rate_gain = self.ang_rate_gain.to(root_state.device)
        mixer = self.mixer.to(root_state.device)
        max_thrusts = self.max_thrusts.to(root_state.device)

        # print(f"g device is {g.device}")
        # print(f"target acc device is {target_acc.device}")
        acc = -(g + target_acc)
        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat([
            torch.cos(target_yaw),
            torch.sin(target_yaw),
            torch.zeros_like(target_yaw)
        ],dim=-1)

        b3_des = -normalize(acc)
        b2_des = normalize(torch.cross(b3_des, b1_des, 1))
        R_des = torch.stack([
            b2_des.cross(b3_des, 1),
            b2_des,
            b3_des
        ], dim=-1)
        ang_error_matrix = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), R)
            - torch.bmm(R.transpose(-2, -1), R_des)
        )
        ang_error = torch.stack([
            ang_error_matrix[:, 2, 1],
            ang_error_matrix[:, 0, 2],
            ang_error_matrix[:, 1, 0]
        ],dim=-1)
        ang_rate_err = ang_vel
        ang_acc = (
            - ang_error * attitute_gain
            - ang_rate_err * ang_rate_gain
            + torch.linalg.cross(ang_vel, ang_vel)
        )
        thrust = (-mass * (acc * R[:, :, 2]).sum(-1, True))
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1)
        cmd = (mixer @ ang_acc_thrust.T).T
        cmd = (cmd / max_thrusts) * 2 - 1
        return cmd

    def process_actions_yaw(self, actions) -> Tensor:
        target_acc, target_yaw = actions.split([3, 1], dim=-1)
        return target_acc, target_yaw * torch.pi

