import copy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from shapely.geometry import Polygon

from track_utils import greedy_assignment

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]


# 99.9 percentile of the l2 velocity error distribution (per clss / 0.5 second)
# This is an earlier statistcs and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
  'car':4,
  'truck':4,
  'bus':5.5,
  'trailer':3,
  'pedestrian':1,
  'motorcycle':13,
  'bicycle':3,  
}


def quaternion_yaw(quat):
  w, x, y, z = [float(v) for v in quat]
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return math.atan2(siny_cosp, cosy_cosp)


def bev_corners(center_xy, size_wlh, yaw):
  width = float(size_wlh[0])
  length = float(size_wlh[1])
  half_l = length * 0.5
  half_w = width * 0.5
  corners = np.array([
    [half_l, half_w],
    [half_l, -half_w],
    [-half_l, -half_w],
    [-half_l, half_w],
  ], dtype=np.float32)

  cos_yaw = math.cos(float(yaw))
  sin_yaw = math.sin(float(yaw))
  rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
  return corners @ rot.T + np.asarray(center_xy, dtype=np.float32).reshape(1, 2)


def rotated_bev_iou(center_a, size_a, yaw_a, center_b, size_b, yaw_b):
  poly_a = Polygon(bev_corners(center_a, size_a, yaw_a))
  poly_b = Polygon(bev_corners(center_b, size_b, yaw_b))
  if (not poly_a.is_valid) or (not poly_b.is_valid):
    return 0.0
  inter = poly_a.intersection(poly_b).area
  union = poly_a.union(poly_b).area
  if union <= 1e-6:
    return 0.0
  return float(inter / union)


def srga_affinity(track, det, det_ct_pred):
  yaw_track = quaternion_yaw(track['rotation'])
  yaw_det = quaternion_yaw(det['rotation'])

  track_size = np.asarray(track['size'][:2], dtype=np.float32)
  det_size = np.asarray(det['size'][:2], dtype=np.float32)

  a_bev = rotated_bev_iou(track['ct'], track['size'], yaw_track, det_ct_pred, det['size'], yaw_det)

  track_diag = math.sqrt(float(track_size[0] ** 2 + track_size[1] ** 2))
  det_diag = math.sqrt(float(det_size[0] ** 2 + det_size[1] ** 2))
  diag_norm = max(0.5 * (track_diag + det_diag), 1e-3)
  center_dist = float(np.linalg.norm(np.asarray(track['ct']) - np.asarray(det_ct_pred)))
  a_center = math.exp(-center_dist / diag_norm)

  size_ratio = (
    abs(math.log(max(float(track_size[1]), 1e-3) / max(float(det_size[1]), 1e-3))) +
    abs(math.log(max(float(track_size[0]), 1e-3) / max(float(det_size[0]), 1e-3)))
  )
  a_size = math.exp(-0.5 * size_ratio)

  lambda_bev = 0.15
  lambda_center = 0.70
  lambda_size = 0.15
  return lambda_bev * a_bev + lambda_center * a_center + lambda_size * a_size



class PubTracker(object):
  def __init__(self,  hungarian=False, max_age=0):
    self.hungarian = hungarian
    self.max_age = max_age

    print("Use hungarian: {}".format(hungarian))

    self.NUSCENE_CLS_VELOCITY_ERROR = NUSCENE_CLS_VELOCITY_ERROR

    self.reset()
  
  def reset(self):
    self.id_count = 0
    self.tracks = []

  def step_centertrack(self, results, time_lag):
    if len(results) == 0:
      self.tracks = []
      return []
    else:
      temp = []
      for det in results:
        # filter out classes not evaluated for tracking 
        if det['detection_name'] not in NUSCENES_TRACKING_NAMES:
          continue 

        det['ct'] = np.array(det['translation'][:2])
        det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
        det['label_preds'] = NUSCENES_TRACKING_NAMES.index(det['detection_name'])
        temp.append(det)

      results = temp

    N = len(results)
    M = len(self.tracks)

    # N X 2 
    if 'tracking' in results[0]:
      dets = np.array(
      [ det['ct'] + det['tracking'].astype(np.float32)
       for det in results], np.float32)
    else:
      dets = np.array(
        [det['ct'] for det in results], np.float32) 

    item_cat = np.array([item['label_preds'] for item in results], np.int32) # N
    track_cat = np.array([track['label_preds'] for track in self.tracks], np.int32) # M

    max_diff = np.array([self.NUSCENE_CLS_VELOCITY_ERROR[box['detection_name']] for box in results], np.float32)

    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    if len(tracks) > 0:  # NOT FIRST FRAME
      dist = (((tracks.reshape(1, -1, 2) - \
                dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M
      dist = np.sqrt(dist) # absolute distance in meter

      invalid = ((dist > max_diff.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0

      affinity = np.zeros((N, M), dtype=np.float32)
      for det_idx in range(N):
        for track_idx in range(M):
          if invalid[det_idx, track_idx]:
            continue
          affinity[det_idx, track_idx] = srga_affinity(
            self.tracks[track_idx],
            results[det_idx],
            dets[det_idx],
          )

      dist = (1.0 - affinity) + invalid * 1e18
      if self.hungarian:
        dist[dist > 1e18] = 1e18
        matched_indices = np.array(linear_assignment(copy.deepcopy(dist)))
        matched_indices = matched_indices.transpose()
      else:
        matched_indices = greedy_assignment(copy.deepcopy(dist))
    else:  # first few frame
      assert M == 0
      matched_indices = np.array([], np.int32).reshape(-1, 2)

    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]

    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    if self.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    ret = []
    for m in matches:
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']      
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      ret.append(track)

    for i in unmatched_dets:
      track = results[i]
      self.id_count += 1
      track['tracking_id'] = self.id_count
      track['age'] = 1
      track['active'] =  1
      ret.append(track)

    # still store unmatched tracks if its age doesn't exceed max_age, however, we shouldn't output 
    # the object in current frame 
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.max_age:
        track['age'] += 1
        track['active'] = 0
        ct = track['ct']

        # movement in the last second
        if 'tracking' in track:
            offset = track['tracking'] * -1 # move forward 
            track['ct'] = ct + offset 
        ret.append(track)

    self.tracks = ret
    return ret
