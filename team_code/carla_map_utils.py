import shapely, math
import numpy as np
from shapely.geometry import LineString, MultiLineString, box, Point, Polygon

from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
from shapely import affinity
from typing import Any, Dict, List, List, Union

class StateSE2:
    def __init__(self, x, y, heading):
        self.x = x
        self.y = y
        self.heading = heading

def get_boundary(map_info, single_lane):
    left_boundary = None
    right_boundary = None
    
    if single_lane['Left'][1] and single_lane['Left'][1] in map_info[single_lane['Left'][0]]:
        left_boundary = np.array([raw_point[0] for raw_point in map_info[single_lane['Left'][0]][single_lane['Left'][1]][0]['Points']])
        
    if single_lane['Right'][1] and single_lane['Right'][1] in map_info[single_lane['Right'][0]]:
        right_boundary = np.array([raw_point[0] for raw_point in map_info[single_lane['Right'][0]][single_lane['Right'][1]][0]['Points']])
    
    return left_boundary, right_boundary

def get_arc_curve(pts) -> float:
    start, end = pts[0], pts[-1]
    l_arc = np.linalg.norm(end - start)
    
    b = np.linalg.norm(pts - start, axis=1)
    c = np.linalg.norm(pts - end, axis=1)
    
    a = l_arc
    tmp = (a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)
    tmp = np.clip(tmp, 0, None)
    
    a = np.clip(a, 1e-10, None)
    
    if np.all(np.abs(tmp) < 1e-6):
        return 10000
    
    dist = np.sqrt(tmp) / (2 * a)
    h = dist.max()
    r = ((a * a) / 4 + h * h) / (2 * h)
    
    return r

def rotate(x, y, angle):
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    res_x = x * cos_angle - y * sin_angle
    res_y = x * sin_angle + y * cos_angle
    return res_x, res_y

# choose the polylines if they are within the ego's radius
def is_within_radius(point, line, radius):
    _, p2 = nearest_points(point, line)
    return point.distance(p2) <= radius

def geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
    """ 将整个地图转化到自车坐标系下, 转化之后, 车头指向y """
    a = np.sin(origin.heading)  # 这里由cos改为sin
    b = -np.cos(origin.heading)  # 这里由sin改为-cos
    d = np.cos(origin.heading)  # 这里由-sin改为cos
    e = np.sin(origin.heading)  # 这里由cos改为sin
    xoff = -origin.x
    yoff = -origin.y
    translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
    rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])
    return rotated_geometry

from shapely.geometry import LineString, MultiLineString, box, Point, Polygon

def interpolate_points(points, intersections, target_distance=1):
    """
    Interpolates points and corresponding intersections such that the distance between each point is approximately target_distance.
    """
    new_points = [np.array(points[0])]
    new_intersections = [intersections[0]]
    accumulated_distance = 0

    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])
        intersection1 = intersections[i]
        intersection2 = intersections[i + 1]
        segment_length = np.linalg.norm(p2 - p1)
        
        while accumulated_distance + segment_length >= target_distance:
            ratio = (target_distance - accumulated_distance) / segment_length
            new_point = p1 + ratio * (p2 - p1)
            new_intersection = intersection1 or intersection2  # 插入的点如果任意一个点是交叉点，则视为交叉点
            new_points.append(new_point)
            new_intersections.append(new_intersection)
            p1 = new_point
            intersection1 = new_intersection
            accumulated_distance = 0
            segment_length = np.linalg.norm(p2 - p1)
        
        accumulated_distance += segment_length

    return new_points, new_intersections

def segment_line_string(line_string, intersections, max_points=5, target_distance=1):
    """
    Segments a LineString such that each segment has approximately target_distance between points,
    each segment has at most max_points points, and processes intersections correspondingly.
    """
    points = list(line_string.coords)
    interpolated_points, interpolated_intersections = interpolate_points(points, intersections, target_distance)
    segmented_lines = []
    segmented_intersections = []

    for i in range(0, len(interpolated_points), max_points):
        segment_points = interpolated_points[i:i + max_points]
        segment_intersections = interpolated_intersections[i:i + max_points]
        
        if len(segment_points) > 1:  # Ensure the segment has more than 1 point
            segmented_lines.append(LineString(segment_points))
            segmented_intersections.append(segment_intersections)
    
    return segmented_lines, segmented_intersections

def clip_line_string(line_string, intersections, x_range=(-50, 50), y_range=(-20, 160)):
    """
    Clips a LineString to the specified rectangular area and processes intersections correspondingly.
    """
    min_x, max_x = x_range
    min_y, max_y = y_range
    clip_box = box(min_x, min_y, max_x, max_y)
    clipped = line_string.intersection(clip_box)
    

    clipped_lines = []
    if isinstance(clipped, LineString):
        clipped_lines.append(clipped)
    elif isinstance(clipped, MultiLineString):
        for part in clipped.geoms:
            clipped_lines.append(part)

    clipped_intersections = []
    for line in clipped_lines:
        line_coords = list(line.coords)
        line_intersections = []
        for coord in line_coords:
            original_index = np.argmin([np.linalg.norm(np.array(coord) - np.array(orig_coord)) for orig_coord in line_string.coords])
            line_intersections.append(intersections[original_index])
        clipped_intersections.append(line_intersections)

    return clipped_lines, clipped_intersections

def interpolate_points_noinc(points, target_distance=1):
    """
    Interpolates points such that the distance between each point is approximately target_distance.
    """
    new_points = [np.array(points[0])]
    accumulated_distance = 0

    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])
        segment_length = np.linalg.norm(p2 - p1)
        while accumulated_distance + segment_length >= target_distance:
            ratio = (target_distance - accumulated_distance) / segment_length
            new_point = p1 + ratio * (p2 - p1)
            new_points.append(new_point)
            p1 = new_point
            accumulated_distance = 0
            segment_length = np.linalg.norm(p2 - p1)
        accumulated_distance += segment_length

    return new_points

def segment_line_string_noinc(line_string, max_points=5, target_distance=1):
    """
    Segments a LineString such that each segment has approximately target_distance between points
    and each segment has at most max_points points.
    """
    points = list(line_string.coords)
    interpolated_points = interpolate_points_noinc(points, target_distance)
    segmented_lines = []

    for i in range(0, len(interpolated_points), max_points):
        segment_points = interpolated_points[i:i + max_points]
        if len(segment_points) > 1:  # Ensure the segment has more than 1 point
            segmented_lines.append(LineString(segment_points))
    
    return segmented_lines

def clip_line_string_noinc(line_string, x_range=(-50, 50), y_range=(-20, 160)):
    """
    Clips a LineString to the specified rectangular area.
    """
    min_x, max_x = x_range
    min_y, max_y = y_range
    clip_box = box(min_x, min_y, max_x, max_y)
    clipped = line_string.intersection(clip_box)
    if isinstance(clipped, (LineString, MultiLineString)):
        return clipped
    return LineString([])

def clip_map_participant(map_participant, ego_origin, radius):
    center_point = Point(ego_origin.x, ego_origin.y)
    clipped_map_participant = {"StopSign": [], "TrafficLight": [], "Center": []}

    for key, path_list in map_participant.items():
        for path_dict in path_list:
            if is_within_radius(center_point, path_dict["baseline"], radius):
                baseline = geometry_local_coords(path_dict["baseline"], ego_origin)
                has_traffic_control = path_dict["has_traffic_control"] # bool
                intersection = path_dict["intersection"] # LIST(bool), None
                left_boundary = path_dict["left_boundary"]  # double, None
                right_boundary = path_dict["right_boundary"] # double, None
                turn_direction = path_dict["turn_direction_type"] # char, None
                cliped_baseline = []
                cliped_intersection = []
                if intersection is not None:
                    segmented_lines, segmented_intersections = segment_line_string(baseline, intersection, max_points=5)
                    for seg, inc in zip(segmented_lines, segmented_intersections):
                        cliped_line, clipped_inc = clip_line_string(seg, inc)
                        all_empty = all(line.is_empty for line in cliped_line)
                        if len(cliped_line) == 0 or all_empty:
                            continue
                        else:
                            cliped_baseline.extend(cliped_line)
                            cliped_intersection.extend(clipped_inc)
                else:
                    for seg in segment_line_string_noinc(baseline, max_points=5):
                        cliped_line = clip_line_string_noinc(seg)
                        if cliped_line.is_empty:
                            continue
                        elif isinstance(cliped_line, MultiLineString):
                            for part in cliped_line.geoms:
                                cliped_baseline.append(part)
                        else:
                            cliped_baseline.append(cliped_line)
                if len(cliped_baseline) == 0:
                    continue
                path_dict = {
                    "baseline": cliped_baseline,
                    "has_traffic_control": has_traffic_control,
                    "intersection": cliped_intersection,
                    "left_boundary": left_boundary,
                    "right_boundary": right_boundary,
                    "turn_direction_type": turn_direction
                }
                clipped_map_participant[key].append(path_dict)

    return clipped_map_participant

def get_map_participant(map_info):
    map_participant = {"StopSign":[], "TrafficLight":[], "Center":[]}
    for road_id, road in map_info.items():
        for lane_id, lane in road.items():
            if lane_id == 'Trigger_Volumes':
                for single_trigger_volume in lane:
                    points = np.array(single_trigger_volume['Points'])
                    points[:,1] *= -1
                    has_traffic_control = True
                    
                    if single_trigger_volume['Type'] == "TrafficLight":
                        assert points.shape[0] % 2 == 0, f"shape is {points.shape}"
                        
                    if single_trigger_volume['Type'] == "StopSign":
                        assert points.shape[0] % 2 == 1, f"shape is {points.shape}"
                        middle_index = points.shape[0] // 2
                        points = np.delete(points, middle_index, axis=0)
                        
                    half = points.shape[0] // 2
                    first_half, second_half = points[:half], points[half:][::-1]
                    points = (first_half + second_half) / 2.0
                    
                    single_trigger_volume_lane_path = LineString(points[:, :2].tolist())
                    map_participant[single_trigger_volume['Type']].append({
                        "baseline": single_trigger_volume_lane_path,
                        "has_traffic_control": has_traffic_control,
                        "intersection": None,
                        "left_boundary": -1e6, # These cases are without boundaries.
                        "right_boundary": -1e6,
                        "turn_direction_type": None,
                    })
            else:
                for idx, single_lane in enumerate(lane):
                    if single_lane['Type'] == "Center":
                        points = np.array([raw_point[0] for raw_point in single_lane['Points']])
                        points[:,1] *= -1
                        if points.shape[0] <= 1:
                            continue
                        has_traffic_control = False
                        
                        intersection = np.array([raw_point[2] for raw_point in single_lane['Points']])
                        assert intersection.shape[0] == points.shape[0], f"shape is {intersection.shape} and {points.shape}"
                        
                        left_boundary, right_boundary = get_boundary(map_info, single_lane)
                        
                        left_distance = np.min(np.linalg.norm(points[:len(left_boundary), :] - left_boundary[:len(points), :], axis=1)) if left_boundary is not None else -1e6
                        right_distance = np.min(np.linalg.norm(points[:len(right_boundary), :] - right_boundary[:len(points), :], axis=1)) if right_boundary is not None else -1e6
                        
                        single_lane_path = LineString(points[:, :2].tolist())
                        traj = points[:, :2]
                        curvature = get_arc_curve(traj)
                        
                        if curvature < 100:
                            start_point, end_point = traj[0], traj[-1]
                            lane_angle = -math.atan2(traj[1][1] - traj[0][1], traj[1][0] - traj[0][0]) + math.pi / 2
                            rotated_end_point = np.array(rotate(end_point[0] - start_point[0], end_point[1] - start_point[1], lane_angle))
                            direction = 'R' if rotated_end_point[0] > 0 else 'L'
                        else:
                            direction = "S"
                        
                        map_participant[single_lane['Type']].append({
                            "baseline": single_lane_path,
                            "has_traffic_control": has_traffic_control,
                            "intersection": intersection,
                            "left_boundary": left_distance,
                            "right_boundary": right_distance,
                            "turn_direction_type": direction,
                        })
    return map_participant

def get_format_output(clipped_map):
    '''
    { 
      当前点坐标:[x,y],
      前一个点坐标: [prev_x, prev_y],
      交通信号灯控制: {True,False},
      是否是交叉路口的点：{True: [1,0], False: [0,1], None: [0,0]},
      转向one-hot:{'R': [1, 0, 0], 'L': [0, 1, 0], 'S': [0, 0, 1], None: [0, 0, 0]}, 
      到左右边界线的平均距离：[d_left, d_right],
      是哪种类型的线one-hot:{"Center": [1,0,0], "StopSign": [0,1,0], "TrafficLight": [0,0,1]}
    }
    '''
    max_lanes = 1000
    max_points_per_lane = 5
    turn_direction_map = {'R': [1, 0, 0], 'L': [0, 1, 0], 'S': [0, 0, 1], None: [0, 0, 0]}
    intersection_map = {True: [1,0], False: [0,1], None: [0,0]}
    line_type_map = {"Center": [1,0,0], "StopSign": [0,1,0], "TrafficLight": [0,0,1]}

    Center = clipped_map["Center"]
    StopSign = clipped_map['StopSign']
    TrafficLight = clipped_map['TrafficLight']

    output_data = np.zeros((max_lanes, max_points_per_lane, 15))
    lane_mask = np.zeros(max_lanes)
    point_mask = np.zeros((max_lanes, max_points_per_lane))
    same_lane_list = []
    input_lane = 0

    for poly_type, poly_line in {"Center": Center, "StopSign": StopSign, "TrafficLight": TrafficLight}.items():
        for lane_idx, line_info in enumerate(poly_line):
            baseline = line_info["baseline"]
            has_traffic_control = line_info["has_traffic_control"]
            intersections = line_info["intersection"]
            left_boundary = line_info["left_boundary"]
            right_boundary = line_info["right_boundary"]
            turn_direction_type = line_info["turn_direction_type"]
            
            if input_lane >= max_lanes:
                break
            in_same_lane = np.zeros((1000,))
            in_same_lane[input_lane: input_lane+len(baseline)] = 1
            same_lane_list.append(in_same_lane)

            for idx, segment in enumerate(baseline):
                if input_lane >= max_lanes:
                    break
                x, y = segment.xy
                num_points = min(len(x), max_points_per_lane)
                lane_mask[input_lane] = 1
                point_mask[input_lane, :num_points] = 1

                intersection = intersections[idx] if len(intersections)!=0 else None

                distance = np.array(list([left_boundary, right_boundary]), dtype=np.float32)

                intersection = np.array([intersection_map.get(i, [0, 0]) for i in intersection], dtype=np.float32) if intersection is not None else np.array([intersection_map.get(None, [0, 0])]*num_points,dtype=np.float32)

                turn_directions = np.array([turn_direction_map.get(turn_direction_type, [0, 0, 0])], dtype=np.float32)

                line_types = np.array([line_type_map.get(poly_type)], dtype=np.float32)

                output_data[input_lane, :num_points, 0] = x[:num_points]  # IDX_CURR_X
                output_data[input_lane, :num_points, 1] = y[:num_points]  # IDX_CURR_Y
                output_data[input_lane, 1:num_points, 2] = x[:num_points-1] # IDX_PREV_X
                output_data[input_lane, 1:num_points, 3] = y[:num_points-1] # IDX_PREV_Y
                output_data[input_lane, :num_points, 4] = [has_traffic_control] * num_points  # IDX_HAS_TRAFFIC_CONTROL
                output_data[input_lane, :num_points, 5:7] = intersection  
                output_data[input_lane, :num_points, 7:10] = turn_directions[None, :]  # TURN_DIRECTION to DISTANCE
                output_data[input_lane, :num_points, 10:12] = distance[None, :]  # DISTANCE to LINE_TYPE
                output_data[input_lane, :num_points, 12:15] = line_types[None, :]  # LINE_TYPE
                input_lane += 1
                
    large_negative_number = -1e6
    output_data = np.where(np.isnan(output_data), large_negative_number, output_data)
    return {
        "road_pts": output_data, 
        "lane_mask": lane_mask, 
        "point_mask": point_mask,
        "same_lane": np.array(same_lane_list),
    }