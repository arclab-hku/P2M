import torch
import numpy as np
import random


def generate_obstacle_tensor(n, pos_x_range, pos_y_range, vel_range, rad_range, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    obstacles = np.zeros((n, 3, 2))
    posy_min, posy_max = pos_y_range
    posx_min, posx_max = pos_x_range
    vel_min, vel_max = vel_range
    rad_min, rad_max = rad_range

    for i in range(n):
        posx = random.uniform(posx_min, posx_max)
        posy = random.uniform(posy_min, posy_max)
        
        velocity_norm = random.uniform(vel_min, vel_max)
        angle = random.uniform(0, 2 * np.pi)
        velx = velocity_norm * np.cos(angle)
        vely = velocity_norm * np.sin(angle)
        rad = random.uniform(rad_min, rad_max)
        hit = 0

        obstacles[i] = [[posx, posy], [velx, vely], [rad, hit]]

    return obstacles

def generate_wall_tensor(n, width, height, fly_height=2.):
    walls = np.zeros((n, 4, 6))
    for i in range(n):
        wall_upper = [0, width/2, fly_height, width, 0.01, height]
        wall_lower = [0, -width/2, fly_height, width, 0.01, height]
        wall_left  = [-width/2, 0, fly_height, 0.01, width, height]
        wall_right = [width/2, 0, fly_height, 0.01, width, height]
        walls[i] = [wall_upper, wall_lower, wall_left, wall_right]

    return walls

def get_bound_misbehave(drone_pos, start_pos, target_pos):
    A = target_pos[:, 1] - start_pos[:, 1]
    B = start_pos[:, 0] - target_pos[:, 0] 
    C = target_pos[:, 0] * start_pos[:, 1] - start_pos[:, 0] * target_pos[:, 1]
    distance = torch.abs(A * drone_pos[:, 0] + B * drone_pos[:, 1] + C) / torch.sqrt(A**2 + B**2)
    mask = distance > 8.0
    bound_misbehave = mask.unsqueeze(1)

    return bound_misbehave

def compute_rayhitsdir(device, num_envs, h_fov, v_fov, h_num, v_num):
    if h_fov == 360:
        horizontal_angles = torch.linspace(0, h_fov, h_num + 1, device=device)
        horizontal_angles = horizontal_angles[:h_num]
    else:
        horizontal_angles = torch.linspace(0, h_fov, h_num, device=device)
    vertical_angles = torch.linspace(v_fov[0], v_fov[1], v_num, device=device) 
    horizontal_radians = horizontal_angles * torch.pi/180 
    vertical_radians = vertical_angles * torch.pi/180
    horizontal_grid, vertical_grid = torch.meshgrid(horizontal_radians, vertical_radians)
    directions = torch.stack((
        torch.cos(vertical_grid) * torch.cos(horizontal_grid),
        torch.cos(vertical_grid) * torch.sin(horizontal_grid),
        torch.sin(vertical_grid) 
    ), dim=-1) 
    ray_hits_dir = directions.reshape(h_num * v_num, -1).unsqueeze(0).expand(num_envs, -1, -1)

    return ray_hits_dir

def update_dobs_vel(device, dobs_states, dobs_pos_x_range, dobs_pos_y_range, drone, trace_prob):
    touch_bound_x_lower = dobs_states[:, 0][:, 0] < dobs_pos_x_range[0]
    touch_bound_x_upper = dobs_states[:, 0][:, 0] > dobs_pos_x_range[1]  
    touch_bound_y_lower = dobs_states[:, 0][:, 1] < dobs_pos_y_range[0]  
    touch_bound_y_upper = dobs_states[:, 0][:, 1] > dobs_pos_y_range[1]  
    touch_bound = (
        touch_bound_x_lower | touch_bound_x_upper | touch_bound_y_lower | touch_bound_y_upper).to(device)

    if not touch_bound.any():
        return dobs_states[:, 1, :]
    
    obs_pos = dobs_states[:, 0, :]
    obs_vel = dobs_states[:, 1, :] 
    drone_pos_xy = drone.get_state(env_frame=False)[..., :2].squeeze(1)
    x_in_center = (drone_pos_xy[:, 0] > -10) & (drone_pos_xy[:, 0] < 10)
    y_in_center = (drone_pos_xy[:, 1] > -10) & (drone_pos_xy[:, 1] < 10)
    valid_drone_mask = (x_in_center & y_in_center)
    valid_drone_indices = torch.nonzero(valid_drone_mask, as_tuple=False).squeeze(1)
    has_valid_drones = valid_drone_indices.numel() > 0  

    if has_valid_drones:
        valid_drone_pos = drone_pos_xy[valid_drone_indices]  
        random_drone_indices = torch.randint(
            0, valid_drone_pos.shape[0], (len(touch_bound),), device=device) 
        random_drone_pos = valid_drone_pos[random_drone_indices]
        direction_to_drone = random_drone_pos - obs_pos  
        direction_norm = torch.norm(direction_to_drone, dim=1, keepdim=True)  
        direction_unit = direction_to_drone / (direction_norm + 1e-8)  
        vel_norm = torch.norm(obs_vel, dim=1, keepdim=True) 
        new_dobs_vel = direction_unit * vel_norm  
    else:
        new_dobs_vel = None

    random_probs = torch.rand(len(touch_bound), device=device)  
    is_trace = random_probs >= trace_prob  
    updated_vel = torch.clone(obs_vel)
    touch_bound_indices = torch.nonzero(touch_bound).squeeze(1)
    if touch_bound_indices.numel() > 0:
        if has_valid_drones:
            new_velocities = torch.where(
                is_trace[touch_bound_indices].unsqueeze(1),
                new_dobs_vel[touch_bound_indices], 
                -obs_vel[touch_bound_indices]
            )
        else:
            new_velocities = -obs_vel[touch_bound_indices]
        vel_x = new_velocities[:, 0]
        vel_y = new_velocities[:, 1]
        reflect_x = (touch_bound_x_lower[touch_bound_indices] & (vel_x <= 0)) | (touch_bound_x_upper[touch_bound_indices] & (vel_x >= 0))
        reflect_y = (touch_bound_y_lower[touch_bound_indices] & (vel_y <= 0)) | (touch_bound_y_upper[touch_bound_indices] & (vel_y >= 0))
        needs_reflect = reflect_x | reflect_y
        new_velocities[needs_reflect] = -obs_vel[touch_bound_indices][needs_reflect]
        updated_vel[touch_bound_indices] = new_velocities
        
    return updated_vel

def dobs_lidar_hits(lidar_range, dobs_height, cylinders, pos_w, ray_hits_dir, error_tolerance=0.33):
    cylinders = cylinders.float()
    pos_w = pos_w.float()
    ray_hits_dir = ray_hits_dir.float()
    cyl_pos = cylinders[:, 0, :2]
    cyl_radius = cylinders[:, 2, 0]
    pos_xy = pos_w[:, :2]
    ray_dir_xy = ray_hits_dir[:, :, :2]

    pos_xy = pos_xy.unsqueeze(1).unsqueeze(2)  
    ray_dir_xy = ray_dir_xy.unsqueeze(-2)  
    cyl_pos = cyl_pos.unsqueeze(0).unsqueeze(0)  
    cyl_radius = cyl_radius.unsqueeze(0).unsqueeze(0) 

    d_horizontal = torch.norm(pos_xy - cyl_pos, dim=-1)
    d_hit = d_horizontal - cyl_radius
    cos_theta = torch.norm(ray_dir_xy, dim=-1)
    t = d_hit / cos_theta
    t = t.unsqueeze(-1)
    intersection_points = pos_w.unsqueeze(1).unsqueeze(2) + t * ray_hits_dir.unsqueeze(-2)  

    z_coords = intersection_points[..., 2]
    valid_z = (z_coords >= 0) & (z_coords <= dobs_height)
    hit_xy = intersection_points[..., :2]
    cyl_xy = cyl_pos
    xy_dist = torch.norm(hit_xy - cyl_xy, dim=-1)
    valid_xy = (xy_dist <= (cyl_radius + error_tolerance))
    valid_hits = valid_z & valid_xy
    t = torch.where(valid_hits, t.squeeze(-1), torch.full_like(t.squeeze(-1), lidar_range))  
    hit_points = pos_w.unsqueeze(1).unsqueeze(2) + t.unsqueeze(-1) * ray_hits_dir.unsqueeze(-2)
    min_t_indices = torch.argmin(t, dim=-1)
    index = min_t_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    hits = torch.gather(hit_points, dim=-2, index=index).squeeze(-2)

    return hits

def wall_lidar_hits(lidar_range, fly_height, wall_size, pos_w, ray_hits_dir):
    wall_size = wall_size.float()
    pos_w = pos_w.float()
    ray_hits_dir = ray_hits_dir.float()
    wall_width = wall_size[:, 0, 0]  
    wall_height = wall_size[:, 0, 2]  
    pos_xy = pos_w[:, :2]   
    ray_dir_xy = ray_hits_dir[:, :, :2]

    wall_width = wall_width.unsqueeze(0).unsqueeze(0)
    pos_xy_expanded = pos_xy.unsqueeze(1).unsqueeze(2)  
    pos_xy_expanded = pos_xy_expanded.expand(-1, ray_hits_dir.shape[1], wall_size.shape[0], -1)  
    pos_w_expanded = pos_w.unsqueeze(1).unsqueeze(2).expand(-1, ray_hits_dir.shape[1], wall_size.shape[0], -1)
    ray_dir_xy_expanded = ray_dir_xy.unsqueeze(2)  
    ray_dir_xy_expanded = ray_dir_xy_expanded.expand(-1, -1, wall_size.shape[0], -1)  
    ray_dir_expanded = ray_hits_dir.unsqueeze(2).expand(-1, -1, wall_size.shape[0], -1)
    left_x = -wall_width / 2  
    right_x = wall_width / 2  
    top_y = wall_width / 2    
    bottom_y = -wall_width / 2  

    t_left   = (left_x   - pos_xy_expanded[:, :, :, 0]) / ray_dir_xy_expanded[:, :, :, 0]  
    t_right  = (right_x  - pos_xy_expanded[:, :, :, 0]) / ray_dir_xy_expanded[:, :, :, 0]  
    t_top    = (top_y    - pos_xy_expanded[:, :, :, 1]) / ray_dir_xy_expanded[:, :, :, 1]  
    t_bottom = (bottom_y - pos_xy_expanded[:, :, :, 1]) / ray_dir_xy_expanded[:, :, :, 1]  
    hit_left   = (t_left > 0)   & (
        pos_xy_expanded[:, :, :, 1] + t_left   * ray_dir_xy_expanded[:, :, :, 1] >= bottom_y) & (
            pos_xy_expanded[:, :, :, 1] + t_left   * ray_dir_xy_expanded[:, :, :, 1] <= top_y)
    hit_right  = (t_right > 0)  & (
        pos_xy_expanded[:, :, :, 1] + t_right  * ray_dir_xy_expanded[:, :, :, 1] >= bottom_y) & (
            pos_xy_expanded[:, :, :, 1] + t_right  * ray_dir_xy_expanded[:, :, :, 1] <= top_y)
    hit_top    = (t_top > 0)    & (
        pos_xy_expanded[:, :, :, 0] + t_top    * ray_dir_xy_expanded[:, :, :, 0] >= left_x)   & (
            pos_xy_expanded[:, :, :, 0] + t_top    * ray_dir_xy_expanded[:, :, :, 0] <= right_x)
    hit_bottom = (t_bottom > 0) & (
        pos_xy_expanded[:, :, :, 0] + t_bottom * ray_dir_xy_expanded[:, :, :, 0] >= left_x)   & (
            pos_xy_expanded[:, :, :, 0] + t_bottom * ray_dir_xy_expanded[:, :, :, 0] <= right_x)
    valid_t_left = torch.full_like(t_left, lidar_range, dtype=torch.float)  
    valid_t_right = torch.full_like(t_right, lidar_range, dtype=torch.float)
    valid_t_top = torch.full_like(t_top, lidar_range, dtype=torch.float)
    valid_t_bottom = torch.full_like(t_bottom, lidar_range, dtype=torch.float)
    valid_t_left[hit_left] = t_left[hit_left]
    valid_t_right[hit_right] = t_right[hit_right]
    valid_t_top[hit_top] = t_top[hit_top]
    valid_t_bottom[hit_bottom] = t_bottom[hit_bottom]
    
    t = torch.cat([valid_t_left.unsqueeze(-1), 
                    valid_t_right.unsqueeze(-1), 
                    valid_t_top.unsqueeze(-1), 
                    valid_t_bottom.unsqueeze(-1)], dim=-1).min(dim=-1)[0]
    t = t.unsqueeze(-1)
    intersection_points = pos_w_expanded + t * ray_dir_expanded
    z_coords = intersection_points[..., 2]
    valid_z = (z_coords >= (fly_height - wall_height/2)) & (z_coords <= (fly_height + wall_height/2))
    t = torch.where(valid_z, t.squeeze(-1), torch.full_like(t.squeeze(-1), lidar_range))
    hit_points = pos_w_expanded + t.unsqueeze(-1) * ray_dir_expanded
    min_t_indices = torch.argmin(t, dim=-1)
    index = min_t_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    hits_wall = torch.gather(hit_points, dim=-2, index=index).squeeze(-2)

    return hits_wall

def merge_hit_points(tensor1, tensor2, tensor3, drone_pos):
    dist1 = torch.norm(tensor1 - drone_pos[:, None, :], dim=2)
    dist2 = torch.norm(tensor2 - drone_pos[:, None, :], dim=2)
    mask_12 = dist1 < dist2
    tensor4 = torch.where(mask_12[:, :, None], tensor1, tensor2)
    dist3 = torch.norm(tensor3 - drone_pos[:, None, :], dim=2)
    dist4 = torch.norm(tensor4 - drone_pos[:, None, :], dim=2)
    mask_34 = dist3 < dist4
    merged_tensor = torch.where(mask_34[:, :, None], tensor3, tensor4)

    return merged_tensor