import copy
import time
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
import math
from scipy.spatial.transform import Rotation as Rot
import cv2
import logging
# from utils_vis import *

logger = logging.getLogger("utils")

class MultiCameraRegistration(object):
    def __init__(self, rgb_intrinsic, relative_mat):
        self.project_idx_x = None
        self.project_idx_y = None
        self.rgb_intrinsic = rgb_intrinsic
        self.relative_mat = relative_mat
        self.eps = 1e-10

    def point_projection_vector(self, point):
        cam_fx = self.rgb_intrinsic[0, 0]
        cam_fy = self.rgb_intrinsic[1, 1]
        cam_cy = self.rgb_intrinsic[1, 2]
        cam_cx = self.rgb_intrinsic[0, 2]
        point_x = point[:, 0] / (point[:, 2] + self.eps)
        point_y = point[:, 1] / (point[:, 2] + self.eps)
        point_x = point_x * cam_fx + cam_cx
        point_y = point_y * cam_fy + cam_cy
        return point_x.astype(np.int32), point_y.astype(np.int32)

    def calculate_project_index(self, pc_array):
        pc_array = np.concatenate([pc_array, np.ones((pc_array.shape[0], 1))], axis=1)
        pc_rgb_array = np.linalg.inv(self.relative_mat).dot(pc_array.T).T[:, :3]
        self.project_idx_x, self.project_idx_y = self.point_projection_vector(pc_rgb_array)

    def get_rbgd_pointcloud(self, pc_array, rgb_img):
        self.calculate_project_index(pc_array)
        rgb_pc = rgb_img[np.clip(self.project_idx_y, 0, rgb_img.shape[0] - 1),
                         np.clip(self.project_idx_x, 0, rgb_img.shape[1] - 1), :]
        rgbd_pcd = o3d.geometry.PointCloud()
        rgbd_pcd.points = o3d.utility.Vector3dVector(pc_array)
        rgbd_pcd.colors = o3d.utility.Vector3dVector(rgb_pc[:, ::-1]/255.0)
        return rgbd_pcd

    def get_img_to_3d_mapping(self, pc_array, rgb_img):
        """
        map each (x,y) in rgb_img to real world x,y,z
        """
        self.calculate_project_index(pc_array)
        maxy, maxx = rgb_img.shape[:2]
        depth_img = np.zeros([maxy, maxx, 3], dtype=float)
        depth_img[:] = np.finfo(float).max
        for (point, rgbx, rgby) in zip(pc_array, self.project_idx_x, self.project_idx_y):
            if not ((0 <= rgbx < maxx) and (0 <= rgby < maxy)): continue
            if point[2] < depth_img[rgby, rgbx, 2]: depth_img[rgby, rgbx] = point
        return depth_img

    def imgXYs_to_3d_mapping(self, imgXYs, pc_array):
        """
        map (x,y) in rgb_img to real world x,y,z
        """
        self.calculate_project_index(pc_array)
        output = []
        for imgX, imgY in imgXYs:
            xy_diff = abs(imgX - self.project_idx_x) + abs(imgY - self.project_idx_y)
            output.append(pc_array[np.argmin(xy_diff)])
        return output

    def get_project_index(self):
        return self.project_idx_x, self.project_idx_y

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area

def proj_bin_depth_map(pcd_array, depth_map_height, bin_scale):
    depth_map = np.zeros((depth_map_height, depth_map_height)).astype(np.uint8)
    x_ = np.round(pcd_array[:, 0] / bin_scale * depth_map_height).astype(int) + depth_map_height // 2
    y_ = np.round(pcd_array[:, 1] / bin_scale * depth_map_height).astype(int) + depth_map_height // 2
    depth_map[y_, x_] = 1
    return depth_map

def de_proj_bin_xy(points, depth_map_height, bin_scale):
    points -= depth_map_height // 2
    pcd_xy = points * bin_scale / depth_map_height
    return pcd_xy 

def locate_box_corners(pc_array, 
                       bottom2camera = 1.09, top2camera = 1.06,
                       box_shape = [0.32, 0.45], height_dir = [0.13751492, 0.00769821, 0.99046978], depth_map_height = 720, 
                       bin_scale = 0.8, thickness = 0.02, show_result = False):
    '''
    box (default params assumes container is blue)
    Args:
        pcd: 3d point cloud (m), numpy array
        img: rgb input numpy array
    Returns:
        a list of 4 corner positions (numpy array: x,y,z)
    '''
    height_dir = np.array(height_dir) / np.linalg.norm(height_dir)
    rot_matrix = create_center_axis([0, 0, -1], height_dir)[:3, :3]
    projected_pc = np.dot(pc_array, rot_matrix)
    box_shape = np.array(box_shape)
    indx = np.logical_and(projected_pc[:,2]<bottom2camera, projected_pc[:,2]>top2camera)
    projected_pc = projected_pc[indx]
    projected_center = np.mean(projected_pc, axis=0).copy()
    projected_pc[:, :2] -= projected_center[:2]

    depth_map = proj_bin_depth_map(projected_pc, depth_map_height, bin_scale)

    # plt.figure()
    # plt.imshow(depth_map)
    # plt.show()

    dilate_kernel = np.ones((3, 3), np.uint8)
    dmask = cv2.dilate(depth_map.astype('uint8'), dilate_kernel, 5)

    mask = dmask
    mask = mask.astype(np.uint8)*255
    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours=list(contours)
    if len(contours)==0:
        logger.warning("Box Detection fail. Abnomal input.")
        return [], [], []
    contours.sort(key = cnt_area, reverse=True)

    rect = cv2.minAreaRect(contours[0])
    box_points = cv2.boxPoints(rect)
    box_center = np.mean(box_points, axis=0)

    bin_top = np.median(projected_pc[:, 2])
    box_xyz = np.ones((4, 3)) * bin_top
    box_center_xyz = np.ones(3) * bin_top
    box_xyz[:, :2] = de_proj_bin_xy(box_points, depth_map_height, bin_scale)
    box_center_xyz[:2] = de_proj_bin_xy(box_center, depth_map_height, bin_scale)

    dir1 = (box_xyz[1] - box_xyz[0]) / np.linalg.norm(box_xyz[1] - box_xyz[0])
    dir2 = (box_xyz[3] - box_xyz[0]) / np.linalg.norm(box_xyz[3] - box_xyz[0])
    dir3 = (box_xyz[2] - box_xyz[1]) / np.linalg.norm(box_xyz[2] - box_xyz[1])
    inner_xyz0 = box_xyz[0] + (dir1 + dir2) * thickness
    inner_xyz1 = box_xyz[1] + (-dir1 + dir3) * thickness
    inner_xyz2 = 2 * box_center_xyz - inner_xyz0
    inner_xyz3 = 2 * box_center_xyz - inner_xyz1
    inner_xyz = np.array([inner_xyz0, inner_xyz1, inner_xyz2, inner_xyz3]) 
    inner_xyz[:, :2] += projected_center[:2]
    corners3D = np.dot(inner_xyz, np.linalg.inv(rot_matrix))
    corners3D = [corners3D[x] for x in range(4)]

    nextshortcorner_id = np.argmin(np.sum((corners3D[1:]-corners3D[0])**2, axis=1)) + 1
    diagcorner_id = np.argmax(np.sum((corners3D[1:]-corners3D[0])**2, axis=1)) + 1
    bin_inner_width = np.sqrt(np.sum((corners3D[6 - diagcorner_id - nextshortcorner_id] - corners3D[0])**2))
    axis_x = corners3D[6 - diagcorner_id - nextshortcorner_id] - corners3D[0]
    axis_y = corners3D[nextshortcorner_id] - corners3D[0]
    axis_x /= np.linalg.norm(axis_x)
    axis_y /= np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    if axis_z[-1]>0:
        axis_y *= -1
        axis_z *= -1
    bin_center_matrix = np.c_[axis_x, np.c_[axis_y, axis_z]]
    bin_center_pose = np.eye(4)
    bin_center_pose[:3, :3] = bin_center_matrix
    bin_center_pose[:3, 3] = np.mean(corners3D, axis=0)
         
    #check the result
    lines=[]
    for i in range(-1,3): 
        lines.append(np.sqrt(np.sum((corners3D[i][:2]-corners3D[i+1][:2])**2)))
    if abs(sum(lines)-sum(box_shape)*2)>(0.1*sum(box_shape)):
        logger.warning("Box Detection fail. Detect wrong shape.")
        return [], [], []

    if show_result:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_array)    
        # draw result on the pointcloud
        pcd.paint_uniform_color([0,0,1]) 
        corner_pcd = o3d.geometry.LineSet()
        line_box = np.array([[0,1],[1,2],[2,3],[3,0]])
        corner_pcd.points = o3d.utility.Vector3dVector(corners3D)  
        corner_pcd.lines = o3d.utility.Vector2iVector(line_box) 
        corner_pcd.paint_uniform_color([255,0,0])
        bin_pose_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        bin_pose_vis.transform(bin_center_pose)
        o3d.visualization.draw_geometries([corner_pcd,pcd, bin_pose_vis]) 
    return corners3D, bin_center_pose, bin_inner_width

def get_neighbor_points(pc_tree, query_point, radius):
    [k, idx, _] = pc_tree.search_radius_vector_3d(query_point, radius)
    return idx

def get_nearest_points(pc_tree, query_point, nearest_num):
    [k, idx, _] = pc_tree.search_knn_vector_3d(query_point, nearest_num)
    return idx

def viewpoint_params_to_matrix(towards, angle):
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]])
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2.dot(R1)
    return matrix.astype(np.float32)

def viewpoint_params_to_matrix1(towards, angle):
    # enable fixed angle rotation for picking
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    initial_angle = math.atan(-axis_x[0]/(axis_x[1]+1e-20))
    angle += initial_angle + np.pi/2
    R1 = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]])
    R2 = np.c_[axis_z, np.c_[axis_y, -axis_x]]
    matrix = R2.dot(R1)
    return matrix.astype(np.float32)

# def get_smoothness_score(grasp_pose, grasp_surface_template_array, sampled_point_neighbor_pcd):
#     R = grasp_pose[:3, :3]
#     t = grasp_pose[:3, 3]
#     grasp_surface_template_array = np.asarray(grasp_surface_template_array)[:, [2, 1, 0]]
#     # grasp_surface_template_array[:, 0] += 0.0005 / 2
#     grasp_surface_point_array = np.dot(R, grasp_surface_template_array.T).T + t
#
#     grasp_surface_point_pcd = o3d.geometry.PointCloud()
#     grasp_surface_point_pcd.points = o3d.utility.Vector3dVector(grasp_surface_point_array)
#     grasp_surface_point_pcd.paint_uniform_color([0, 0.5, 0.5])
#     smoothness_score = compute_grasp_surface_intersection_ratio(grasp_surface_point_pcd, sampled_point_neighbor_pcd)
#     # o3d.visualization.draw_geometries([sampled_point_neighbor_pcd, grasp_surface_point_pcd])
#     return smoothness_score, grasp_surface_point_pcd

def get_smoothness_score1(grasp_pose, grasp_surface_template_array, sampled_point_neighbor_pcd, smooth_voxel_size=0.003):
    R = grasp_pose[:3, :3]
    t = grasp_pose[:3, 3]
    grasp_surface_template_array = np.asarray(grasp_surface_template_array)
    grasp_surface_point_array = np.dot(R, grasp_surface_template_array.T).T + t

    grasp_surface_point_pcd = o3d.geometry.PointCloud()
    grasp_surface_point_pcd.points = o3d.utility.Vector3dVector(grasp_surface_point_array)
    smoothness_score = compute_grasp_surface_intersection_ratio(grasp_surface_point_pcd,
                                                                sampled_point_neighbor_pcd, smooth_voxel_size)
    return smoothness_score

def get_smoothness_score2(grasp_pose, cup_radius, downsample_voxelsize, sampled_point_neighbor_array, smooth_voxel_size=0.003):
    ## fast smoothness computation
    R = -grasp_pose[:3, :3]
    t = grasp_pose[:3, 3]
    # grasp_surface_point_pcd = o3d.geometry.PointCloud()
    # grasp_surface_point_pcd.points = o3d.utility.Vector3dVector(sampled_point_neighbor_array-t)
    # O = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
    # trans_matrix = np.eye(4)
    # trans_matrix[:3, :3] = R
    # O.transform(trans_matrix)
    # O1 = o3d.geometry.TriangleMesh.create_coordinate_frame(0.02)
    # o3d.visualization.draw_geometries([grasp_surface_point_pcd, O, O1])
    sampled_point_in_cup_depth = np.sum(R[:3, 2] * (sampled_point_neighbor_array-t), axis=1)
    cup_point_num = np.pi*cup_radius**2/downsample_voxelsize**2 + 2*np.pi*cup_radius/downsample_voxelsize
    smoothness_score = np.sum(sampled_point_in_cup_depth <= smooth_voxel_size) / cup_point_num
    return smoothness_score  

def get_smoothness_score_by_plate(smooth_radius, downsample_voxelsize, sampled_point_neighbor_pcd, smooth_voxel_size=0.001):
    cup_point_num = (np.pi*smooth_radius**2)/(downsample_voxelsize**2) + 2*np.pi*smooth_radius/downsample_voxelsize
    try:
        plane_model, inliers = sampled_point_neighbor_pcd.segment_plane(distance_threshold=smooth_voxel_size,
                                                ransac_n=10,
                                                num_iterations=10)
        norm_dir = plane_model[0:3]   
        if norm_dir[2]<0:
            norm_dir *= -1                                        
        smoothness_score = len(inliers)/cup_point_num
        return smoothness_score, norm_dir
    except:
        logger.warning("Find no smooth plane!")
        return 0, [0, 0, 1]  

def compute_grasp_surface_intersection_ratio(gripper_pcd, inst_pcd, voxel_size):  # 0.0015
    source_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(inst_pcd, voxel_size=voxel_size)
    gripper_pcd = gripper_pcd.voxel_down_sample(voxel_size)
    queries = np.asarray(gripper_pcd.points)
    intersection_result = source_voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    intersected_pc_idx = np.array(intersection_result).nonzero()[0]

    # gripper_pcd.paint_uniform_color([0, 0.5, 0.5])
    # o3d.visualization.draw_geometries([inst_pcd, gripper_pcd])

    return len(intersected_pc_idx) / len(intersection_result)

def get_vector_angle(x1, x2):
    if x1.shape != x2.shape:
        return np.rad2deg(np.arccos(x1.dot(x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))))
    else:
        return np.rad2deg(np.arccos(x1.dot(x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))))

def evaluate_cup_area(inst_pcd, inst_pcd_tree, cup_radius, offset, center_pose, grasp_surface_template,
                      minimum_search_neighbor_pc, smooth_voxel_size):
    offset_matrix = np.identity(4)
    offset_matrix[:3, 3] = np.array([[0, offset, 0]])
    sampled_pose = center_pose.dot(offset_matrix)
    sampled_pose_trans = sampled_pose[:3, 3]
    sampled_pose_idx = get_neighbor_points(inst_pcd_tree, sampled_pose_trans, cup_radius)
    if len(sampled_pose_idx) <= minimum_search_neighbor_pc:
        logger.warning("no enough point for evaluating cup area")
        return [], -1
    sampled_pose_neighbors_pcd = inst_pcd.select_by_index(sampled_pose_idx)
    sampled_pose[:3, 3] = np.mean(np.asarray(sampled_pose_neighbors_pcd.points), axis=0)
    sampled_pose_smoothness_score = get_smoothness_score1(sampled_pose,
                                                        grasp_surface_template,
                                                        sampled_pose_neighbors_pcd,
                                                          smooth_voxel_size)
    return sampled_pose, sampled_pose_smoothness_score

def get_grasp_quat(sampled_pose):
    ### get_grasp_quat
    x_180_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    grasp_rot = np.dot(sampled_pose[:3, :3], x_180_R)
    q = Rot.from_matrix(grasp_rot).as_quat()
    q = q / np.linalg.norm(q)
    t = sampled_pose[:3, 3]
    pose_quat = [t[0], t[1], t[2], q[0], q[1], q[2], q[3]]
    return pose_quat

def cal_bin_rect_pose_inv(corners3D):
    ## calculate transition matrix for collision detection with bin
    dists = np.sum((corners3D[0] - corners3D[1:]) ** 2, axis=1)
    i = np.argmin(dists) + 1
    j = np.argmax(dists) + 1
    k = 6 - i - j
    axis_x = corners3D[i] - corners3D[0]
    axis_y = corners3D[k] - corners3D[0]
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    bin_rect_matrix = np.c_[axis_x, np.c_[axis_y, axis_z]]
    bin_rect_pose = np.eye(4)
    bin_rect_pose[:3, :3] = bin_rect_matrix
    bin_rect_pose[:3, 3] = corners3D[0]
    bin_rect_pose_inv = np.linalg.inv(bin_rect_pose)
    return bin_rect_pose_inv

def collide_bin(inst_pose, bin_plane_normal, corners3D, bin_rect_pose_inv=[], cup_radius=0.015):
    proj_length = np.dot(bin_plane_normal, inst_pose[:3, 3])
    m = np.abs(np.dot(bin_plane_normal, corners3D[0]) / proj_length)
    pose_z_dir = inst_pose[:3, 2].copy()
    if pose_z_dir[2] < 0:  # orient normals along camera z axis
        pose_z_dir *= -1
    cos_alpha = np.dot(bin_plane_normal, pose_z_dir)
    inst_point_intersect = inst_pose[:3, 3] - (1 - m) * proj_length / cos_alpha * pose_z_dir
    inst_point_proj = inst_pose[:3, 3] - (1 - m) * proj_length * bin_plane_normal

    ##### visualization #####
    # bin_points = [o3d.geometry.TriangleMesh.create_sphere(radius=0.005) for x in range(4)]
    # for x in range(4): 
    #     bin_points[x].paint_uniform_color([1, 0, 0]) 
    #     bin_points[x].vertices = o3d.utility.Vector3dVector(np.asarray(bin_points[x].vertices + corners3D[x]))
    # cylinder_height = 0.25
    # pose_quat = get_grasp_quat(inst_pose)
    # R = Rot.from_quat(pose_quat[3:]).as_matrix()
    # t = pose_quat[:3]
    # mesh_arrow = o3d.geometry.TriangleMesh.create_cylinder(0.015, cylinder_height)
    # vertices = np.asarray(mesh_arrow.vertices)
    # vertices[:, 2] -= cylinder_height / 2
    # vertices = np.dot(R, vertices.T).T + t
    # mesh_arrow.vertices = o3d.utility.Vector3dVector(vertices)
    # mesh_arrow.paint_uniform_color([0.5, 0.5, 0.8])
    # grasp_pose = np.eye(4)
    # grasp_pose[:3, :3] = R
    # grasp_pose[:3, 3] = t
    # grasp_pose_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # grasp_pose_vis.transform(grasp_pose)
    # O = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    # o3d.visualization.draw_geometries([mesh_arrow, grasp_pose_vis] + bin_points)

    dists = np.sum((corners3D[0] - corners3D[1:]) ** 2, axis=1)
    j = np.argmax(dists) + 1
    if len(bin_rect_pose_inv)==0:
        bin_rect_pose_inv = cal_bin_rect_pose_inv(corners3D)

    diag_point_h = np.ones((4, 1))
    diag_point_h[:3, 0] = corners3D[j]
    diag_point_trans = np.matmul(bin_rect_pose_inv, diag_point_h)
    inst_point_intersect_h = np.ones((4, 1))
    inst_point_intersect_h[:3, 0] = inst_point_intersect
    inst_point_intersect_trans = np.matmul(bin_rect_pose_inv, inst_point_intersect_h)
    inst_point_proj_h = np.ones((4, 1))
    inst_point_proj_h[:3, 0] = inst_point_proj
    inst_point_proj_trans = np.matmul(bin_rect_pose_inv, inst_point_proj_h)
    ## calculate ellipse parameters https://cloud.tencent.com/developer/article/2067487
    a, b = cup_radius / cos_alpha, cup_radius # major_radius, minor_radius
    ellip_dir = inst_point_intersect_trans[:2, 0] - inst_point_proj_trans[:2, 0]
    rot_angle = np.arctan(ellip_dir[1] / (ellip_dir[0] + 1e-20))
    sin_theta = np.sin(rot_angle)
    cos_theta = np.cos(rot_angle)
    A = a**2 * sin_theta**2 + b**2 * cos_theta**2
    B = 2 * (a**2 - b**2) * sin_theta * cos_theta
    C = a**2 * cos_theta**2 + b**2 * sin_theta**2
    D = -a**2 * b**2
    x = np.sqrt(4*C*D / (B**2 - 4*C*A))
    x1, x2 = inst_point_intersect_trans[0, 0] - np.abs(x), inst_point_intersect_trans[0, 0] + np.abs(x) # elllipse bounding box x
    y = np.sqrt(4*A*D / (B**2 - 4*A*C))
    y1, y2 = inst_point_intersect_trans[1, 0] - np.abs(y), inst_point_intersect_trans[1, 0] + np.abs(y) # elllipse bounding box y

    ## visualization
    # plt.figure()
    # plt.axis('equal')
    # theta = np.linspace(0, 2*np.pi, 100)
    # plot_x = a * np.cos(theta)
    # plot_y = b * np.sin(theta)
    # plot_x1 = plot_x * np.cos(rot_angle) - plot_y * np.sin(rot_angle)
    # plot_y1 = plot_x * np.sin(rot_angle) + plot_y * np.cos(rot_angle)
    # plot_x1 += inst_point_intersect_trans[0, 0]
    # plot_y1 += inst_point_intersect_trans[1, 0]
    # plt.plot(plot_x1, plot_y1)
    # plt.plot([x1, x2], [y1, y2], 'o')
    # plt.plot([inst_point_intersect_trans[0, 0], inst_point_proj_trans[0, 0]], [inst_point_intersect_trans[1, 0], inst_point_proj_trans[1, 0]], 'x')
    # plt.show()
    space = 3e-3
    diag_point_trans -= space # give space
    if x1>space and x2>space\
        and y1>space and y2>space\
        and x1 < diag_point_trans[0, 0] and x2 < diag_point_trans[0, 0]\
        and y1 < diag_point_trans[1, 0] and y2 < diag_point_trans[1, 0]:
        return False
    else:
        return True

def gene_suction_point(inst_pcd, center_pose, grasp_surface_template,
                         cup_radius, SKU_herizontal_offset, SKU_smoothness_threshold,
                         minimum_search_neighbor_pc, smooth_voxel_size, down_offset, height_dir, corners3D):

    inst_pcd_tree = o3d.geometry.KDTreeFlann(inst_pcd)
    sampled_pose, sampled_pose_smoothness_score = evaluate_cup_area(inst_pcd, inst_pcd_tree,
                                                                    cup_radius, 0, center_pose,
                                                                    grasp_surface_template,
                                                                    minimum_search_neighbor_pc,
                                                                    smooth_voxel_size)
    if sampled_pose_smoothness_score==-1:
        return [], [], [], []
    smoothness_ok = True
    if sampled_pose_smoothness_score < SKU_smoothness_threshold:
        logger.warning(f"no enough smoothness for current pose: {sampled_pose_smoothness_score}")
        smoothness_ok = False
    if not smoothness_ok:
        for offset in [-SKU_herizontal_offset, SKU_herizontal_offset]:
            if offset==0:
                continue
            sampled_pose1, sampled_pose_smoothness_score1, = evaluate_cup_area(inst_pcd, inst_pcd_tree,
                                                                               cup_radius, offset, center_pose,
                                                                               grasp_surface_template,
                                                                               minimum_search_neighbor_pc,
                                                                               smooth_voxel_size)
            if sampled_pose_smoothness_score1 >= SKU_smoothness_threshold:
                smoothness_ok = True
                sampled_pose = sampled_pose1
                sampled_pose_smoothness_score = sampled_pose_smoothness_score1
                logger.info(f"Find a new smooth scution pose: {sampled_pose_smoothness_score1}")
                break
            if sampled_pose_smoothness_score1 > sampled_pose_smoothness_score:
                sampled_pose_smoothness_score = sampled_pose_smoothness_score1
                sampled_pose = sampled_pose1
    if smoothness_ok:
        smoothness_final_score = 1
    else:
        smoothness_final_score = sampled_pose_smoothness_score
    # filter abnomal normal direction
    if len(corners3D)!=0:
        if not collide_bin(sampled_pose, height_dir, corners3D):
            normal_score = 1
        else:
            logger.warning('Warning!!! Impossible Angle!!!!!!!')
            return [], [], [], []
    else:
        normal_score = 1
    if down_offset!=0:
        offset_matrix = np.identity(4)
        offset_matrix[:3, 3] = np.array([[0, 0, -down_offset]])
        sampled_pose = sampled_pose.dot(offset_matrix)
    pose_quat = get_grasp_quat(sampled_pose)
    return sampled_pose, pose_quat, smoothness_final_score, normal_score

def rank_suction(sampled_points_list, smoothness_score_list, normal_score_list, height_dir=None):
    sampled_points_list = np.asarray(sampled_points_list)
    smoothness_scores = np.asarray(smoothness_score_list)
    normal_scores = np.asarray(normal_score_list)
    if len(sampled_points_list) > 1:
        if height_dir is not None:
            depths = np.dot(sampled_points_list, height_dir)
        else:
            depths = sampled_points_list[:, 2]
        d_min = np.min(depths)
        height_scores = d_min / depths
    else:
        height_scores = np.array([1.0], dtype=np.float32)
    final_scores = height_scores + smoothness_scores + normal_scores
    sorted_idxs = np.argsort(final_scores)
    # add instance-level randomness top2
    # if np.random.rand() < 0.8:
        # np.random.shuffle(sorted_idxs[-2:])
    return sorted_idxs

def crop_center_pcd_deformable(scene_pcd, _inst_mask, project_relation, minimum_inst_pc):
    minAreaRect, _ = find_min_rect(_inst_mask)
    cx, cy = minAreaRect[0]
    crop_range_pixels = 0.25 * max(minAreaRect[1])
    _inst_2d_center = np.round([cy, cx]).astype(np.int64)
    project_index_x, project_index_y = project_relation

    mask_index = np.where(_inst_mask==1)
    x_min, x_max, y_min, y_max = np.min(mask_index[0]), np.max(mask_index[0]), np.min(mask_index[1]), np.max(mask_index[1])
    Y, X = np.ogrid[x_min:x_max+1, y_min:y_max+1]
    dist_from_center = np.sqrt((X - _inst_2d_center[1])**2 + (Y-_inst_2d_center[0])**2)
    center_mask = dist_from_center <= crop_range_pixels
    inst_center_mask = _inst_mask.copy()
    inst_center_mask[x_min:x_max+1, y_min:y_max+1] = inst_center_mask[x_min:x_max+1, y_min:y_max+1] * center_mask
    inst_center_mask = inst_center_mask[np.clip(project_index_y, 0, inst_center_mask.shape[0] - 1),
                                        np.clip(project_index_x, 0, inst_center_mask.shape[1] - 1)]
    inst_center_index = np.flatnonzero(inst_center_mask)
    # inst_center_index = np.where(np.logical_and(np.abs(project_index_y - _inst_2d_center[0]) < crop_range_pixels,
    #                                             np.abs(project_index_x - _inst_2d_center[1]) < crop_range_pixels))[0]
    inst_center_pcd = scene_pcd.select_by_index(inst_center_index)
    inst_center_array = np.asarray(inst_center_pcd.points)

    # inst_center_pcd.paint_uniform_color([1,0,0])
    # draw_geometries_with_view_ctr1([-0.2,-0.1,1.74], [0,-1,0], [0,0,-1], 0.5, [inst_center_pcd, scene_pcd])

    # check if enough point cloud
    if len(inst_center_array) < minimum_inst_pc:
        logger.warning("no enough instance point cloud captured")
        return [], []
    return inst_center_pcd, inst_center_array

def crop_inst_pcd(scene_pcd, _inst_mask, project_relation, minimum_inst_pc):
    project_index_x, project_index_y = project_relation
    inst_mask = _inst_mask[np.clip(project_index_y, 0, _inst_mask.shape[0] - 1),
                           np.clip(project_index_x, 0, _inst_mask.shape[1] - 1)]
    inst_pcd_index = np.flatnonzero(inst_mask)
    inst_pcd = scene_pcd.select_by_index(inst_pcd_index)
    inst_pcd_array = np.asarray(inst_pcd.points)

    # inst_pcd.paint_uniform_color([1,0,0])
    # draw_geometries_with_view_ctr1([-0.2,-0.1,1.74], [0,-1,0], [0,0,-1], 0.5, [inst_pcd, scene_pcd])

    # check if enough point cloud
    if len(inst_pcd_array) < minimum_inst_pc:
        logger.warning("no enough instance point cloud captured")
        return [], []
    return inst_pcd, inst_pcd_array

def proj_depth_map(inst_pcd_array, depth_map_height):
    # project to depth map along camera z direction
    depth_map = np.zeros((depth_map_height, depth_map_height)).astype(np.uint8)
    SKU_scale = 2.1 * max(np.max(inst_pcd_array[:, 0]), -np.min(inst_pcd_array[:, 0]),
                          np.max(inst_pcd_array[:, 1]), -np.min(inst_pcd_array[:, 1]))
    SKU_scale = max(0.4, SKU_scale)
    x_ = np.round(inst_pcd_array[:, 0] / SKU_scale * depth_map_height).astype(np.int64) + depth_map_height // 2
    y_ = np.round(inst_pcd_array[:, 1] / SKU_scale * depth_map_height).astype(np.int64) + depth_map_height // 2
    depth_map[y_, x_] = 1
    if depth_map_height > 50:
        kernel = np.ones((3,3))
        erosion = cv2.erode(depth_map, kernel, iterations=1)
        depth_map = cv2.dilate(erosion, kernel, iterations=2)
    return depth_map, SKU_scale

def crop_center_pointcloud(scene_pcd, inst_pcd_array, _inst_mask, project_relation, fx, center_range, offset=[]):
    project_index_x, project_index_y = project_relation
    # project cup size to pixel space
    rough_distance = np.median(inst_pcd_array[:, 2])
    center_range_pixels = fx * center_range / rough_distance / 1.414
    minAreaRect, _ = find_min_rect(_inst_mask)
    cx, cy = minAreaRect[0]
    if offset!=[]:
        h = np.max(minAreaRect[1])
        w = np.min(minAreaRect[1])
        rec_bpoints = cv2.boxPoints(minAreaRect)
        dists = np.sum((rec_bpoints[0] - rec_bpoints[1:]) ** 2, axis=1)
        i = np.argmin(dists) + 1
        w_theta = math.atan((rec_bpoints[i, 1] - rec_bpoints[0, 1]) /
                            (rec_bpoints[i, 0] - rec_bpoints[0, 0] + 1e-20))
        _inst_2d_center = np.round([cy + 0.25*h*math.cos(w_theta)*offset[0] + 0.25*w*math.sin(w_theta)*offset[1], 
                                    cx + 0.25*h*math.sin(w_theta)*offset[0] + 0.25*w*math.cos(w_theta)*offset[1]]).astype(np.int64)
    else:
        # _inst_2d_center = np.round([cy, cx]).astype(np.int64)
        ## add randomness
        rand_delta_xy = 2 * (np.random.rand(2) - 0.5) * center_range_pixels * 1.5 # 1.5 times of center_range_pixels for sampling range
        _inst_2d_center = np.round([cy + rand_delta_xy[0], cx + rand_delta_xy[1]]).astype(np.int64)
    inst_2d_center_index = np.where((project_index_y - _inst_2d_center[0])**2 +
                                    (project_index_x - _inst_2d_center[1])**2 < center_range_pixels**2)[0]
    inst_center_pcd = scene_pcd.select_by_index(inst_2d_center_index)
    inst_center_array = np.asarray(inst_center_pcd.points)

    # inst_center_pcd.paint_uniform_color([1,0,0])
    # inst_pcd = o3d.geometry.PointCloud()
    # inst_pcd.points = o3d.utility.Vector3dVector(inst_pcd_array)
    # inst_pcd.paint_uniform_color([0,0,0])
    # o3d.visualization.draw_geometries([inst_center_pcd, inst_pcd])

    # check if point cloud center is empty
    if len(inst_center_array) < 5:
        logger.warning("not enough center points")
        return [], []
    return inst_center_array, inst_center_pcd

def crop_center_missing_pointcloud(inst_pcd_array, depth_map_height, center_range, view_point):
    view_project = np.dot(inst_pcd_array, view_point)
    inst_center_z = np.mean(view_project)
    inst_pruned_array = inst_pcd_array[view_project < inst_center_z]
    inst_center = np.mean(inst_pcd_array, axis=0)
    inst_pruned_array -= inst_center
    depth_map, SKU_scale = proj_depth_map(inst_pruned_array, depth_map_height)
    if np.sum(depth_map)==0:
        logger.warning("not enough center points")
        return []
    cx, cy, _ = find_max_circle(depth_map, depth_map_height)
    if cx==[]:
        logger.warning("not enough center points")
        return []
    rand_delta_xy = 2 * (np.random.rand(2) - 0.5) * center_range * 1.5 # 1.5 times of center_range for sampling range
    cx_ = (cx - depth_map_height // 2) / depth_map_height * SKU_scale + inst_center[0] + rand_delta_xy[0]
    cy_ = (cy - depth_map_height // 2) / depth_map_height * SKU_scale + inst_center[1] + rand_delta_xy[1]
    center_dis = (inst_pcd_array[:, 0] - cx_)**2 + (inst_pcd_array[:, 1] - cy_)**2
    inst_3d_center_index = np.where(center_dis < center_range**2)[0]
    inst_center_array = inst_pcd_array[inst_3d_center_index]

    # plt.imshow(depth_map)
    # plt.plot(cx, cy, 'g+')
    # plt.show()
    # inst_center_pcd = o3d.geometry.PointCloud()
    # inst_center_pcd.points = o3d.utility.Vector3dVector(inst_center_array)
    # inst_pcd = o3d.geometry.PointCloud()
    # inst_pcd.points = o3d.utility.Vector3dVector(inst_pcd_array)
    # inst_center_pcd.paint_uniform_color([1,0,0])
    # inst_pcd.paint_uniform_color([0,0,0])
    # o3d.visualization.draw_geometries([inst_pcd, inst_center_pcd])

    # check if point cloud center is empty
    if len(inst_center_array) < 5:
        logger.warning("not enough center points")
        return []
    return inst_center_array

def create_center_axis(inst_center_point, inst_center_normal):
    if np.dot(inst_center_normal, inst_center_point) < 0:  # orient normals along camera z axis
        inst_center_normal *= -1
    # calculate initial center pose rotation matrix
    axis_x = inst_center_normal
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    init_center_matrix = np.c_[axis_z, np.c_[axis_y, -axis_x]]
    init_center_pose = np.eye(4)
    init_center_pose[:3, :3] = init_center_matrix
    init_center_pose[:3, 3] = inst_center_point
    return init_center_pose

def find_init_center_pose(scene_pcd, inst_pcd_array, _inst_mask, project_relation, fx, center_range, offset=[]):
    inst_center_array, _ = crop_center_pointcloud(scene_pcd, inst_pcd_array, _inst_mask, project_relation, fx,
                                               center_range, offset)
    if inst_center_array == []:
        return [], [], 1e3
    # inst_center_point = np.mean(inst_center_array, axis=0)
    inst_center_point = inst_center_array[np.random.randint(0, len(inst_center_array))].copy() # add randomness
    inst_center_array -= inst_center_point
    S = np.cov(inst_center_array.transpose((1, 0)))
    eig_val, eig_vec = np.linalg.eig(S)
    inst_center_normal = eig_vec[:, np.argmin(eig_val)]
    avg_dist = np.sqrt(np.sum(np.sum(inst_center_normal * inst_center_array, axis=1) ** 2) / len(inst_center_array))
    return inst_center_point, inst_center_normal, avg_dist

def est_init_center_pose(scene_pcd, inst_pcd_array, _inst_mask, project_relation, fx, center_range):
    inst_center_point, inst_center_normal, avg_dist = find_init_center_pose(scene_pcd, inst_pcd_array,
                                                                            _inst_mask, project_relation, fx, center_range)
    if inst_center_point==[]:
        logger.warning("Find no init center pose") 
        return []                                                                       
    dis_threshold = 1e-3
    if avg_dist>dis_threshold:
        logger.info("Init pose shifted!")
        for offset in [[-1, 0], [1, 0]]:
            inst_center_point, inst_center_normal, avg_dist = find_init_center_pose(scene_pcd, inst_pcd_array,
                                                                                    _inst_mask, project_relation, fx, center_range, offset)
            if avg_dist<=dis_threshold: break
    init_center_pose = create_center_axis(inst_center_point, inst_center_normal)
    return init_center_pose

def est_multiple_init_center_pose(scene_pcd, inst_pcd_array, _inst_mask, project_relation, fx, center_range):
    init_center_poses = []                                                                   
    dis_threshold = 1e-3
    sampling_times = 5
    random_offsets = (np.random.uniform(0, 1, sampling_times*2)-0.5)*1.6
    for i in range(sampling_times):
        offset = [random_offsets[2*i], random_offsets[2*i+1]]
        inst_center_point, inst_center_normal, avg_dist = find_init_center_pose(scene_pcd, inst_pcd_array,
                                                                                    _inst_mask, project_relation, fx, center_range, offset)
        if avg_dist>dis_threshold: continue
        init_center_pose = create_center_axis(inst_center_point, inst_center_normal)
        init_center_poses.append(init_center_pose)
    if init_center_poses==[]:
        logger.warning("Find no init center pose")  
    return init_center_poses

def est_seed_pose(inst_pcd_array, inst_center_array, depth_threshold, depth_map_height, min_SKU_area, max_SKU_area):
    # inst_center_point = np.mean(inst_center_array, axis=0)
    inst_center_point = inst_center_array[np.random.randint(0, len(inst_center_array))].copy() # add randomness
    inst_center_array -= inst_center_point
    S = np.cov(inst_center_array.transpose((1, 0)))
    eig_val, eig_vec = np.linalg.eig(S)
    inst_center_normal = eig_vec[:, np.argmin(eig_val)]
    init_seed_pose = create_center_axis(inst_center_point, inst_center_normal)
    # trans_matrix = np.eye(4)
    # trans_matrix[:3, :3] = init_seed_pose[:3, :3]
    inst_pcd_array -= inst_center_point
    inst_pcd_array_cp = np.dot(inst_pcd_array, init_seed_pose[:3, :3])
    # project to depth map along initial center normal direction
    inst_pcd_array_cp[:, 2] = np.abs(inst_pcd_array_cp[:, 2])
    selected_array = inst_pcd_array_cp[inst_pcd_array_cp[:, 2] < depth_threshold]
    if len(selected_array) == 0:
        logger.warning("abnormal plane scales: 0!!!!")
        return []
    depth_map, SKU_scale = proj_depth_map(selected_array, depth_map_height)
    _, area = find_min_rect(depth_map)
    mask_area = area * (1 / depth_map_height * SKU_scale) ** 2
    if mask_area < min_SKU_area or mask_area > max_SKU_area:
        logger.warning(f"abnormal plane scales: {mask_area}!!!!")
        return []
    return init_seed_pose

def view_init_center_pose(scene_pcd, inst_pcd_array, _inst_mask, project_relation, fx, center_range):
    inst_center_array, inst_center_pcd = crop_center_pointcloud(scene_pcd, inst_pcd_array, _inst_mask, project_relation, fx,
                                               center_range)
    if inst_center_array == []:
        return [], []
    inst_center_point = inst_center_array[np.random.randint(0, len(inst_center_array))] # add randomness
    plane_model, _ = inst_center_pcd.segment_plane(distance_threshold=0.003, ransac_n=5, num_iterations=10)
    norm_dir = plane_model[0:3]
    if norm_dir[2]<0:
        norm_dir *= -1  
    init_center_pose = create_center_axis(inst_center_point, norm_dir)
    return init_center_pose, norm_dir

def missing_init_center_pose(inst_pcd_array, depth_map_height, center_range, view_point):
    inst_center_array = crop_center_missing_pointcloud(inst_pcd_array, depth_map_height, center_range, view_point)
    if inst_center_array == []:
        return [], []
    inst_center_point = inst_center_array[np.random.randint(0, len(inst_center_array))] # add randomness
    inst_center_pcd = o3d.geometry.PointCloud()
    inst_center_pcd.points = o3d.utility.Vector3dVector(inst_center_array)
    plane_model, _ = inst_center_pcd.segment_plane(distance_threshold=0.003, ransac_n=5, num_iterations=10)
    norm_dir = plane_model[0:3]
    if norm_dir[2]<0:
        norm_dir *= -1  
    init_center_pose = create_center_axis(inst_center_point, norm_dir)
    return init_center_pose, norm_dir

def find_min_rect(mask):
    mask = mask.astype(np.uint8)
    # find 2d contour in pixel space
    contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_area_list = [cv2.contourArea(x) for x in contours]
    if np.max(contours_area_list) == 0:
        return [], 0
    cont_points = contours[np.argmax(contours_area_list)].squeeze()
    minAreaRect = cv2.minAreaRect(cont_points)
    return minAreaRect, np.max(contours_area_list)

def find_max_circle(mask, depth_map_height):
    # find 2d contour in pixel space
    contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_area_list = [cv2.contourArea(x) for x in contours]
    if np.max(contours_area_list) == 0:
        return [], [], 0
    (center_point, box, _), _ = find_min_rect(mask)
    limit_bound = (max(box) / 5) ** 2
    cont_points = contours[np.argmax(contours_area_list)].squeeze()
    min_x, max_x = min(cont_points[:, 0]), max(cont_points[:, 0])
    min_y, max_y = min(cont_points[:, 1]), max(cont_points[:, 1])
    raw_dist = np.zeros((depth_map_height, depth_map_height))
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            if (i-center_point[0])**2 + (j-center_point[1])**2 > limit_bound:
                continue
            raw_dist[j, i] = cv2.pointPolygonTest(cont_points, (i, j), True)
    max_dis = np.max(raw_dist[:])
    center_tuple = np.where(raw_dist==max_dis)
    xy_coor = np.array([center_tuple[1], center_tuple[0]])
    mean_xy_coor = np.mean(xy_coor, axis=1)
    dis = (xy_coor[0] - mean_xy_coor[0])**2 + (xy_coor[1] - mean_xy_coor[1])**2
    # ind = np.argmin(dis)
    ind = np.random.choice(np.where(dis==np.min(dis))[0]) # add randomness
    return xy_coor[0][ind], xy_coor[1][ind], np.max(contours_area_list)

def rectify_center_pose(init_center_pose, inst_pcd_array_cp, depth_map_height, min_SKU_area, max_SKU_area, depth_threshold, proj_dir=[]):
    # trans_matrix = np.eye(4)
    # trans_matrix[:3, :3] = init_center_pose[:3, :3]
    inst_pcd_array_cp -= init_center_pose[:3, 3]
    if len(proj_dir)==0:    
        inst_pcd_array_cp = np.dot(inst_pcd_array_cp, init_center_pose[:3, :3])
        inst_pcd_array_z_cp = np.abs(inst_pcd_array_cp[:, 2])
    else:
        inst_pcd_array_z_cp = np.abs(np.dot(inst_pcd_array_cp, proj_dir))
        inst_pcd_array_cp = np.dot(inst_pcd_array_cp, init_center_pose[:3, :3])
    # project to depth map along initial center normal direction
    selected_array = inst_pcd_array_cp[inst_pcd_array_z_cp < depth_threshold]
    if len(selected_array)==0:
        logger.warning("abnormal plane scales: 0!!!!")
        return []
    depth_map, SKU_scale = proj_depth_map(selected_array, depth_map_height)
    if np.sum(depth_map)==0:
        logger.warning("abnormal plane scales: 0!!!!")
        return []

    minAreaRect, area = find_min_rect(depth_map)
    if minAreaRect==[]:
        logger.warning("abnormal plane scales: 0!!!!")
        return []
    rec_bpoints = cv2.boxPoints(minAreaRect)
    cx, cy = minAreaRect[0]
    # check if center is shifted
    if abs(cx - depth_map_height // 2) + abs(cy - depth_map_height // 2) > 0.003 / SKU_scale * depth_map_height:
        # rectify center location
        delta_x = (cx - depth_map_height // 2) / depth_map_height * SKU_scale
        delta_y = (cy - depth_map_height // 2) / depth_map_height * SKU_scale
        delta = np.dot(np.array([delta_x, delta_y, 0]), np.linalg.inv(init_center_pose[:3, :3]))
        init_center_pose[:3, 3] += delta

    # plt.imshow(depth_map)
    # plt.plot(rec_bpoints[:, 0], rec_bpoints[:, 1], 'g-')
    # plt.show()

    # screen planes with abnormal scales
    mask_area = area * (1 / depth_map_height * SKU_scale) ** 2
    if mask_area < min_SKU_area or mask_area > max_SKU_area:
        logger.warning(f"abnormal plane scales: {mask_area}!!!!")
        return []
    # detect principle direction
    dists = np.sum((rec_bpoints[0] - rec_bpoints[1:]) ** 2, axis=1)
    w = np.argmin(dists) + 1
    w_theta = math.atan((rec_bpoints[w, 1] - rec_bpoints[0, 1]) /
                        (rec_bpoints[w, 0] - rec_bpoints[0, 0] + 1e-20))
    R1 = np.array([[np.cos(w_theta), -np.sin(w_theta), 0],
                   [np.sin(w_theta), np.cos(w_theta), 0],
                   [0, 0, 1]])
    inst_center_normal_mat = init_center_pose[:3, :3].dot(R1)
    center_pose = np.identity(4)
    center_pose[:3, 3] = init_center_pose[:3, 3]
    center_pose[:3, :3] = inst_center_normal_mat
    return center_pose

def find_max_dir_missing_pc(depth_map):
    ys, xs = np.where(depth_map>0)
    xys = np.stack((xs, ys))
    center_point = np.mean(xys, axis=1).reshape(2, -1)
    center_point = center_point.astype(np.int64)
    xys -= center_point
    S = np.cov(xys)
    eig_val, eig_vec = np.linalg.eig(S)
    max_dir = eig_vec[:, np.argmax(eig_val)]
    w_theta = np.arctan(max_dir[1] / (max_dir[0] + 1e-10)) + np.pi / 2
    return w_theta    

def rectify_missing_center_pose(init_center_pose, inst_pcd_array_cp, depth_map_height, min_SKU_area, max_SKU_area, depth_threshold, proj_dir):
    # trans_matrix = np.eye(4)
    # trans_matrix[:3, :3] = init_center_pose[:3, :3]
    inst_pcd_array_cp -= init_center_pose[:3, 3]
    inst_pcd_array_z_cp = np.abs(np.dot(inst_pcd_array_cp, proj_dir))
    inst_pcd_array_cp = np.dot(inst_pcd_array_cp, init_center_pose[:3, :3])
    # project to depth map along initial center normal direction
    # inst_pcd_array_cp[:, 2] = np.abs(inst_pcd_array_cp[:, 2])
    selected_array = inst_pcd_array_cp[inst_pcd_array_z_cp < depth_threshold]
    if len(selected_array)==0:
        logger.warning("abnormal plane scales: 0!!!!")
        return []
    depth_map, SKU_scale = proj_depth_map(selected_array, depth_map_height)
    if np.sum(depth_map)==0:
        logger.warning("abnormal plane scales: 0!!!!")
        return []

    cx, cy, area = find_max_circle(depth_map, depth_map_height)
    if cx==[]:
        logger.warning("abnormal plane scales: 0!!!!")
        return []
    # check if center (normalized at (d_height//2, d_height//2)) is shifted over 3 mm
    if abs(cx - depth_map_height // 2) + abs(cy - depth_map_height // 2) > 0.003 / SKU_scale * depth_map_height:
        # rectify center location
        delta_x = (cx - depth_map_height // 2) / depth_map_height * SKU_scale
        delta_y = (cy - depth_map_height // 2) / depth_map_height * SKU_scale
        delta = np.dot(np.array([delta_x, delta_y, 0]), np.linalg.inv(init_center_pose[:3, :3]))
        init_center_pose[:3, 3] += delta

    # plt.imshow(depth_map)
    # plt.plot(cx, cy, 'g+')
    # plt.show()
    # inst_center_pcd = o3d.geometry.PointCloud()
    # inst_center_pcd.points = o3d.utility.Vector3dVector(selected_array)
    # inst_pcd = o3d.geometry.PointCloud()
    # inst_pcd.points = o3d.utility.Vector3dVector(inst_pcd_array_cp)
    # inst_center_pcd.paint_uniform_color([1,0,0])
    # inst_pcd.paint_uniform_color([0,0,0])
    # o3d.visualization.draw_geometries([inst_pcd, inst_center_pcd])

    # screen planes with abnormal scales
    mask_area = area * (1 / depth_map_height * SKU_scale) ** 2
    if mask_area < min_SKU_area or mask_area > max_SKU_area:
        logger.warning(f"abnormal plane scales: {mask_area}!!!!")
        return []
    # determine max principle direction
    w_theta = find_max_dir_missing_pc(depth_map)
    R1 = np.array([[np.cos(w_theta), -np.sin(w_theta), 0],
                   [np.sin(w_theta), np.cos(w_theta), 0],
                   [0, 0, 1]])
    inst_center_normal_mat = init_center_pose[:3, :3].dot(R1)
    center_pose = np.identity(4)
    center_pose[:3, 3] = init_center_pose[:3, 3]
    center_pose[:3, :3] = inst_center_normal_mat
    return center_pose

def grasp_pose_augment(detected_pose_list):
    detected_pose_list_aug = np.zeros((len(detected_pose_list) * 2, 7))
    z_90_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    for i in range(len(detected_pose_list)):
        detected_pose_list_aug[i*2: i*2+2] = detected_pose_list[i]
        quat = detected_pose_list[i, 3:]
        R = Rot.from_quat(quat).as_matrix()
        R_aug = np.dot(R, z_90_R)

        # O = o3d.geometry.TriangleMesh.create_coordinate_frame(0.01)
        # O_cp = copy.deepcopy(O)
        # O.rotate(R, center=(0,0,0))
        # O_cp.rotate(R_aug, center=(0,0,0))
        # o3d.visualization.draw_geometries([O, O_cp])

        quat_aug = Rot.from_matrix(R_aug).as_quat()
        quat_aug = quat_aug / np.linalg.norm(quat_aug)
        detected_pose_list_aug[i*2+1, 3:] = quat_aug
    return detected_pose_list_aug

def evaluate_suction_candidates(candidate_centers, candidate_normals, candidate_area, height_direction=np.array([0,0,-1]), height_range=0.5, area_range=0):
    '''
    Compute final scores of all candidates by evaluating their height and smoothness quality
    '''
    # Start evaluating
    message = 'Evaluating suction candidates'
    logger.info(message)

    # timer
    t0 = time.time()

    # get height scores, map heights into [1-height_range, 1]
    heights = np.dot(candidate_centers, height_direction)
    heights = heights - heights.min()
    height_scores = (heights / (heights.max()+1e-9)) * height_range + 1-height_range
    logger.info(f"{heights}, {height_scores}")

    # get smooth scores
    area_scores = candidate_area * area_range + 1-area_range

    # evaluate normals?

    # get final scores
    final_scores = area_scores * height_scores
    sorted_idx = np.argsort(final_scores)

    # output all scores
    for candidate_id in sorted_idx:
        logger.info(f"Candidate #{candidate_id}: area: {area_scores[candidate_id]}, height: {height_scores[candidate_id]}")

    # timer
    logger.info(f"> Done in {time.time() - t0}s")
    return final_scores, sorted_idx

def get_flat_points(inst_pcd, normal_threshold=20, height_threshold=0.2, target_normal=np.array([0, 0, -1])):
    """
    Compute flat points from raw instance point cloud
    """
    # get instance point normals
    inst_normals_array = np.asarray(inst_pcd.normals)

    # select points whose normals are within a threshold
    normal_cos_angle = np.dot(inst_normals_array, target_normal)
    flat_points_idx = normal_cos_angle > np.cos(np.deg2rad(normal_threshold))

    # remove lower points
    inst_pcd_array = np.asarray(inst_pcd.points)
    heights = np.dot(inst_pcd_array, target_normal)
    heights = heights - heights.min()
    flat_points_idx = np.logical_and(flat_points_idx, heights > height_threshold * heights.max())
    return flat_points_idx

def get_center_via_clustering(flat_points, cluster_min_points=10):
    """
    Get new instance center from flat areas, use the center point of the largest cluster
    """
    # clustering
    labels = np.array(flat_points.cluster_dbscan(eps=0.01, min_points=cluster_min_points,
                                                 print_progress=False))  # lower eps to split closer patches
    max_label = labels.max()

    # get the largest flat area
    cluster_point_num = np.array([np.sum(labels == cluster_id) for cluster_id in range(max_label + 1)],
                                 dtype=np.int64)
    flat_area_idx = np.argmax(cluster_point_num).astype(np.int64)
    logger.info(f"point cloud has {max_label + 1} clusters, choose cluster #{flat_area_idx} with {cluster_point_num[flat_area_idx]} points as the flat area")

    # get the area center, use it as the new instance center
    flat_area_pcd = flat_points.select_by_index(np.flatnonzero(labels == flat_area_idx))
    flat_area_center = flat_area_pcd.get_center()  # TODO: need to improve method for computing the center position, mean position is not always good

    # find closest point (to area center) among flat points
    flat_pcd_kdtree = o3d.geometry.KDTreeFlann(flat_area_pcd)
    k, query_idx, query_dis = flat_pcd_kdtree.search_knn_vector_3d(flat_area_center, knn=1)
    new_center_point = flat_area_pcd.select_by_index(query_idx)
    return new_center_point, labels, cluster_point_num


######################################
# Transparent bottles below
######################################

def judge_by_bottle_patch(img, cx, cy, crop_boundary, width_proj, w_theta, patch_tem):
    ## judge bottle direction via image patch matching
    x_min_cr, x_max_cr, y_min_cr, y_max_cr = (max(0, int(cx - crop_boundary)),
                                              min(img.shape[1], int(cx + crop_boundary)),
                                              max(0, int(cy - crop_boundary)),
                                              min(img.shape[0], int(cy + crop_boundary)))
    img_crop = img[y_min_cr:y_max_cr, x_min_cr:x_max_cr]
    cx_cr = int(cx) - x_min_cr
    cy_cr = int(cy) - y_min_cr
    ## rotate img
    rows, cols = img_crop.shape[0], img_crop.shape[1]
    M = cv2.getRotationMatrix2D((cx_cr, cy_cr), np.rad2deg(w_theta), 1)
    img_rot = cv2.warpAffine(img_crop, M, (cols, rows))

    ## crop bottle patch
    img_patch1 = img_rot[cy_cr - width_proj // 2 : cy_cr + width_proj // 2, cx_cr : cx_cr + crop_boundary]
    img_patch2 = img_rot[cy_cr - width_proj // 2 : cy_cr + width_proj // 2, cx_cr - crop_boundary : cx_cr]
    patch_tem = patch_tem[:crop_boundary, :]

    ## sift feature matching
    bf = cv2.BFMatcher()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_patch1, None)
    kp2, des2 = sift.detectAndCompute(img_patch2, None)
    kp3, des3 = sift.detectAndCompute(patch_tem, None)
    matches1 = bf.knnMatch(des1, des3, k=2)
    matches2 = bf.knnMatch(des2, des3, k=2)

    if len(matches1)>len(matches2):
        w_theta_choice = 0
    else:
        w_theta_choice = 1
    return w_theta_choice

def estimate_cap_direction(mask, img, cap_mass, sku_height, sku_width, sku_cap_area, multi_camera_registration, bin_boundary, patch_tem, horizontal_flag):
    cap_center_ho = np.ones((4, 1))
    cap_center_ho[:3, 0] = cap_mass
    cap_center_realsense = np.linalg.inv(multi_camera_registration.relative_mat).dot(cap_center_ho).T[0, :3]
    cap_distance = cap_center_realsense[-1]
    SKU_height_proj = int(sku_height / cap_distance * multi_camera_registration.rgb_intrinsic[0][0])
    SKU_width_proj = int(sku_width / cap_distance * multi_camera_registration.rgb_intrinsic[0][0])
    minAreaRect, mask_area = find_min_rect(mask)
    cap_area = mask_area * (cap_distance / multi_camera_registration.rgb_intrinsic[0][0]) ** 2
    if cap_area < sku_cap_area*0.7 or cap_area > sku_cap_area*1.3:
        logger.warning(f"abnormal cap scales: {cap_area}!!!!")
        return [], []
    cx, cy = minAreaRect[0]
    ## detect principle direction
    rec_bpoints = cv2.boxPoints(minAreaRect)
    dists = np.sum((rec_bpoints[0] - rec_bpoints[1:]) ** 2, axis=1)
    w = np.argmin(dists) + 1
    w_theta = math.atan((rec_bpoints[w, 1] - rec_bpoints[0, 1]) /
                        (rec_bpoints[w, 0] - rec_bpoints[0, 0] + 1e-20))

    if not horizontal_flag:  # no need to template matching
        mask_yx = np.argwhere(mask > 0).transpose((1, 0))
        mask_mass = np.mean(mask_yx, axis=1)
        rough_dir = np.array([mask_mass[1] - cx, mask_mass[0] - cy])
        directions = np.array(
            [[math.cos(w_theta + x * np.pi * 0.5), math.sin(w_theta + x * np.pi * 0.5)] for x in range(4)])
        direc_dots = np.dot(directions, rough_dir)
        w_theta += np.argmax(direc_dots) * np.pi / 2
    else:  # have two possible rotation candidates for rectangle bottle cap
        bott_x0 = cx + SKU_height_proj * math.cos(w_theta)
        bott_y0 = cy + SKU_height_proj * math.sin(w_theta)
        bott_x1 = cx - SKU_height_proj * math.cos(w_theta)
        bott_y1 = cy - SKU_height_proj * math.sin(w_theta)
        x_min, x_max, y_min, y_max = bin_boundary
        # if bott_x0 < x_min or bott_x0 > x_max or bott_y0 < y_min or bott_y0 > y_max:
        #     w_theta_choice = 1
        # elif bott_x1 < x_min or bott_x1 > x_max or bott_y1 < y_min or bott_y1 > y_max:
        #     w_theta_choice = 0
        # else:
        crop_boundary = SKU_height_proj
        w_theta_choice = judge_by_bottle_patch(img, cx, cy, crop_boundary,
                                                SKU_width_proj, w_theta, patch_tem)
        if w_theta_choice == 1:
            w_theta += np.pi

    # for visualization
    # bott_x = cx + SKU_height_proj * math.cos(w_theta)
    # bott_y = cy + SKU_height_proj * math.sin(w_theta)
    # plt.imshow(mask)
    # plt.plot(cx, cy, 'r+')
    # rec_bpoints_copy = np.concatenate((rec_bpoints, rec_bpoints[0:1]), axis=0)
    # plt.plot(rec_bpoints_copy[:, 0], rec_bpoints_copy[:, 1], '-g')
    # plt.plot([cx, bott_x], [cy, bott_y], 'b-')
    # plt.show()

    return [np.math.cos(w_theta), np.math.sin(w_theta)], [cy, cx]

def pca(pts, center_pt, sort=True):
    normalized_pts = pts - center_pt
    H = np.dot(normalized_pts.T, normalized_pts)
    eigen_vectors, eigen_values, _ = np.linalg.svd(H)
    if sort:
        sort = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[sort]
        eigen_vectors = eigen_vectors[:, sort]
    return eigen_values, eigen_vectors

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def clean_scene_pts(scene_pcd, template_pcd, pose, clean_threshold = 5e-3):
    
    scene_pts = np.array(scene_pcd.points)
    template_pcd_cp = copy.deepcopy(template_pcd)
    template_pcd_cp.transform(pose)
    template_pts = np.array(template_pcd_cp.points)

    dis = scene_pts[:, np.newaxis, :] - template_pts[np.newaxis, :, :]
    dis = np.linalg.norm(dis, axis=-1)

    min_dis = np.min(dis, axis=-1)
    clean_flag = np.where(min_dis < clean_threshold)[0]
    clean_pts = scene_pts[clean_flag, :]
    
    clean_scene_pcd = o3d.geometry.PointCloud()
    clean_scene_pcd.points = o3d.utility.Vector3dVector(clean_pts)
    return clean_scene_pcd

def preprocess_cap_pcd(scene_pcd):
    labels = np.array(scene_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))

    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("Set2")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # scene_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([scene_pcd])

    candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
    best_candidate=int(np.unique(labels)[np.where(candidates== np.max(candidates))[0]])
    scene_pcd = scene_pcd.select_by_index(list(np.where(labels==best_candidate)[0]))

    # process normal
    scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    scene_pcd.orient_normals_towards_camera_location()

    normal_pcd = o3d.geometry.PointCloud()
    normal_pcd.points = scene_pcd.normals
    # o3d.visualization.draw_geometries([normal_pcd])

    normal_labels = np.array(normal_pcd.cluster_dbscan(eps=0.2, min_points=5, print_progress=False))

    # max_label = normal_labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("Set2")(normal_labels / (max_label if max_label > 0 else 1))
    # colors[normal_labels < 0] = 0
    # scene_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([scene_pcd])

    candidates=[len(np.where(normal_labels==j)[0]) for j in np.unique(normal_labels)]
    best_candidate=int(np.unique(normal_labels)[np.where(candidates== np.max(candidates))[0]])
    scene_pcd = scene_pcd.select_by_index(list(np.where(normal_labels==best_candidate)[0]))

    # o3d.visualization.draw_geometries([scene_pcd], point_show_normal=True)
    return scene_pcd

# initial pose from prior information of 2D + ICP refactored
def pose_estimation3(scene_pcd, scene_pts, template_pcd, cap_direc):
    time1 = time.time()
    scene_pcd = preprocess_cap_pcd(scene_pcd)
    scene_pts = np.array(scene_pcd.points)

    icp_threshold = 0.008
    # icp_threshold = 0.02
    cap_center = np.mean(scene_pts, axis=0)

    theta_z = np.math.atan2(cap_direc[1], cap_direc[0]) - np.pi / 2
    eigen_values, eigen_vectors = pca(scene_pts, cap_center)
    normal_idx = np.absolute(eigen_vectors[-1, :]).argmax()
    global_normal = eigen_vectors[:, normal_idx]

    if global_normal[-1] < 0:
        global_normal = -global_normal
    init_rotation_x = rotation_matrix_from_vectors(np.array([0, 0, 1]), global_normal)

    init_rotation_z = np.eye(3).astype(np.float32)
    init_rotation_z[0, 0] = np.math.cos(theta_z)
    init_rotation_z[0, 1] = -np.math.sin(theta_z)
    init_rotation_z[1, 0] = np.math.sin(theta_z)
    init_rotation_z[1, 1] = np.math.cos(theta_z)
    init_rotation = np.dot(init_rotation_x, init_rotation_z)

    init_pose = np.eye(4).astype(np.float32)
    init_pose[:3, :3] = init_rotation
    init_pose[:3, 3] = cap_center
    reg_p2p = o3d.pipelines.registration.registration_icp(template_pcd, scene_pcd, icp_threshold, init_pose,
                                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5))
    refine_pose = np.asarray(reg_p2p.transformation)
    ICP_evaluation = o3d.pipelines.registration.evaluate_registration(template_pcd, scene_pcd, 0.002, refine_pose)
    fitness = ICP_evaluation.fitness

    # O = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    # O.transform(init_pose)
    # O1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.015)
    # O1.transform(refine_pose)
    # template_pcd_cp = copy.deepcopy(template_pcd)
    # template_pcd_cp.transform(init_pose)
    # template_pcd_cp.paint_uniform_color([0, 1, 0])
    # template_pcd_cp1 = copy.deepcopy(template_pcd)
    # template_pcd_cp1.transform(refine_pose)
    # template_pcd_cp1.paint_uniform_color([1, 0, 0])

    # o3d.visualization.draw_geometries([template_pcd_cp1, scene_pcd, O1])
    # o3d.visualization.draw_geometries([template_pcd_cp, scene_pcd, O])

    return init_pose, refine_pose, fitness

# pose from prior information of 2D
def pose_estimation4(scene_pcd, template_pcd, cap_pcd, cap_direc, cap_mass, cap_center_rgb, fx, project_relation, center_range, height_dir):
    time1 = time.time()
    center_range_pixels = fx * center_range / cap_mass[2] / 1.414
    _inst_2d_center = np.round(cap_center_rgb).astype(np.int64)
    inst_2d_center_index = \
        np.where(np.logical_and(np.abs(project_relation[1] - _inst_2d_center[0]) < center_range_pixels,
                                np.abs(project_relation[0] - _inst_2d_center[1]) < center_range_pixels))[0]
    cap_center_pcd = scene_pcd.select_by_index(inst_2d_center_index)
    cap_center_pcd_array = np.asarray(cap_center_pcd.points)
    cap_center = np.mean(cap_center_pcd_array, axis=0)

    # cap_center_pcd.paint_uniform_color([1,0,0])
    # o3d.visualization.draw_geometries([cap_pcd, cap_center_pcd])

    # cap_pcd_tree = o3d.geometry.KDTreeFlann(cap_pcd)
    # cap_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.002*2, max_nn=10))
    # cap_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))
    # center_neighbor_idx = get_neighbor_points(cap_pcd_tree, cap_center, center_range)
    # center_neighbor_pcd = cap_pcd.select_by_index(center_neighbor_idx)
    # center_normals = np.asarray(center_neighbor_pcd.normals)
    # center_point_normal = np.mean(center_normals, axis=0)
    # center_point_normal = center_point_normal / np.linalg.norm(center_point_normal)

    theta_z = np.math.atan2(cap_direc[1], cap_direc[0]) - np.pi / 2
    global_normal = height_dir #center_point_normal
    if global_normal[-1] < 0:
        global_normal = -global_normal
    init_rotation_x = rotation_matrix_from_vectors(np.array([0, 0, 1]), global_normal)

    init_rotation_z = np.eye(3).astype(np.float32)
    init_rotation_z[0, 0] = np.math.cos(theta_z)
    init_rotation_z[0, 1] = -np.math.sin(theta_z)
    init_rotation_z[1, 0] = np.math.sin(theta_z)
    init_rotation_z[1, 1] = np.math.cos(theta_z)
    init_rotation = np.dot(init_rotation_x, init_rotation_z)

    init_pose = np.eye(4).astype(np.float32)
    init_pose[:3, :3] = init_rotation
    init_pose[:3, 3] = cap_center

    # O = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)
    # O.transform(init_pose)
    # template_pcd_cp = copy.deepcopy(template_pcd)
    # template_pcd_cp.transform(init_pose)
    # template_pcd_cp.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([template_pcd_cp, cap_pcd, O])

    ICP_evaluation = o3d.pipelines.registration.evaluate_registration(template_pcd, scene_pcd, 0.002, init_pose)
    fitness = ICP_evaluation.fitness
    logger.info(f"Pose estimation time: {time.time() - time1}")
    return init_pose, fitness

def rank_pose(grasp_pose_list, fitness_scores_list, height_dir=None):
    depths = np.asarray(grasp_pose_list)[:, :3, 3]
    fitness_scores = np.asarray(fitness_scores_list)
    if len(grasp_pose_list) > 1:
        if height_dir is not None:
            depths = np.dot(depths, height_dir)
        else:
            depths = depths[:, 2]
        d_min = np.min(depths)
        height_scores = d_min / depths
    else:
        height_scores = np.array([1.0], dtype=np.float32)
    final_scores = height_scores + fitness_scores
    sorted_idxs = np.argsort(final_scores)
    return sorted_idxs

def get_smoothness_score(grasp_pose, grasp_surface_template_array, sampled_point_neighbor_pcd):
    R = grasp_pose[:3, :3]
    t = grasp_pose[:3, 3]
    grasp_surface_template_array = np.asarray(grasp_surface_template_array)[:, [2, 1, 0]]
    grasp_surface_template_array[:, 0] += 0.0005 / 2
    grasp_surface_point_array = np.dot(R, grasp_surface_template_array.T).T + t

    grasp_surface_point_pcd = o3d.geometry.PointCloud()
    grasp_surface_point_pcd.points = o3d.utility.Vector3dVector(grasp_surface_point_array)
    grasp_surface_point_pcd.paint_uniform_color([0, 0.5, 0.5])
    smoothness_score = compute_grasp_surface_intersection_ratio(grasp_surface_point_pcd, sampled_point_neighbor_pcd, 0.003)
    return smoothness_score, grasp_surface_point_pcd

def nms(scores, points, distance_threshold):
    res_ids = list(range(len(scores)))
    after_nms_ids = []
    while len(res_ids) != 0:
        cur_scores = scores[res_ids]
        max_score = np.max(cur_scores)
        max_idx = np.where(scores == max_score)
        if len(max_idx[0]) == 1:
            max_idx = max_idx[0][0]
        else:
            for i in max_idx[0]:
                if i in res_ids:
                    max_idx = i
                    break
        after_nms_ids.append(max_idx)
        try:
            res_ids.remove(max_idx)
        except ValueError:
            logger.warning("0")
        m_point = points[max_idx]
        dists = np.sqrt(np.sum((points - m_point) ** 2, axis=1))
        _res_ids = [i for i in res_ids if dists[i] > distance_threshold]
        res_ids = _res_ids
    return after_nms_ids

def filter_normal(normals, candidate_idxs, threshold=30):
    candidate_idxs = np.array(candidate_idxs)
    candidate_normals = normals[candidate_idxs]
    normal_dists = get_vector_angle(candidate_normals, np.array([0, 0, 1]))
    selected_mask = normal_dists < threshold
    return candidate_idxs[selected_mask]

def filter_smoothness(smoothness_scores, candidate_idxs, threshold):
    candidate_idxs = np.array(candidate_idxs)
    candidate_smoothness_scores = smoothness_scores[candidate_idxs]
    selected_mask = candidate_smoothness_scores > threshold
    return candidate_idxs[selected_mask]

def suction_filter(sampled_points, smoothness_scores, sampled_normals):
    '''
    steps:
    1. get height scores 0-1
    2. get smoothness scores
    3. get center scores
    4. normal threshold
    5. final_score = 0.8*height*smoothness + 0.2*center
    6. nms
    '''

    logger.info("SUCTION ANALYSIS INFO:")
    sampled_points_len = len(sampled_points)
    logger.info(f"Sampled points number:{sampled_points_len}" )
    if sampled_points_len > 1:
        depths = sampled_points[:, 2]
        d_max = np.percentile(depths, 95)
        d_min = np.percentile(depths, 5)

        normalized_depth = (depths - d_min) / (d_max - d_min)
        normalized_depth = np.clip(normalized_depth, 0.0, 1.0)
        height_scores = 1 - normalized_depth
    else:
        height_scores = np.array([1.0], dtype=np.float32)

    smoothness_scores = np.array(smoothness_scores).copy()
    candidate_scores = smoothness_scores

    ## direction thresholding
    complete_idxs = list(range(len(candidate_scores)))
    vertical_candidate_idxs = filter_normal(sampled_normals, complete_idxs, 15)
    if len(vertical_candidate_idxs) == 0:
        logger.info('angle range: 45')
        vertical_candidate_idxs = filter_normal(sampled_normals, complete_idxs, 45)
        if len(vertical_candidate_idxs) == 0:
            logger.warning('Warning!!! Hard Angle!!!!!!!')
            vertical_candidate_idxs = complete_idxs

    # # TODO: debug for nms
    nms_idxs = nms(candidate_scores[vertical_candidate_idxs], sampled_points[vertical_candidate_idxs], 0.005)  # 0.08
    nms_candidate_idxs = vertical_candidate_idxs[nms_idxs]

    smooth_candidate_idxs = filter_smoothness(smoothness_scores, nms_candidate_idxs, 0.5)
    sorted_idxs = np.argsort(candidate_scores[smooth_candidate_idxs])
    candidate_sorted_idxs = smooth_candidate_idxs[sorted_idxs[::-1]]
    return candidate_sorted_idxs
