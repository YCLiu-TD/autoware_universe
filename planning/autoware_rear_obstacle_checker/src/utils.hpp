// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include "autoware/behavior_path_planner_common/marker_utils/colors.hpp"

#include <autoware_lanelet2_extension/utility/message_conversion.hpp>
#include <autoware_lanelet2_extension/utility/utilities.hpp>
#include <autoware_utils/geometry/boost_geometry.hpp>
#include <autoware_utils/geometry/boost_polygon_utils.hpp>
#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/unit_conversion.hpp>
#include <autoware_utils/ros/marker_helper.hpp>
#include <autoware_utils/transform/transforms.hpp>

#include <boost/assert.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/format.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/algorithms/area.hpp>
#include <boost/geometry/algorithms/distance.hpp>
#include <boost/geometry/algorithms/length.hpp>
#include <boost/geometry/algorithms/within.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/geometry/Point.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::rear_obstacle_checker::utils
{

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

namespace
{
std::uint8_t get_highest_prob_label(
  const std::vector<autoware_perception_msgs::msg::ObjectClassification> & classifications)
{
  std::uint8_t label = autoware_perception_msgs::msg::ObjectClassification::UNKNOWN;
  float highest_prob = 0.0;
  for (const auto & classification : classifications) {
    if (highest_prob < classification.probability) {
      highest_prob = classification.probability;
      label = classification.label;
    }
  }
  return label;
}
}  // namespace

bool should_activate(
  const autoware_internal_planning_msgs::msg::PlanningFactor & factor,
  const rear_obstacle_checker_node::Params & parameters)
{
  const auto config = parameters.scene_map.at(factor.module);

  if (config.safety_condition == "safe") {
    if (!factor.safety_factors.is_safe) {
      return false;
    }
  }

  if (config.safety_condition == "unsafe") {
    if (factor.safety_factors.is_safe) {
      return false;
    }
  }

  switch (factor.behavior) {
    case PlanningFactor::SLOW_DOWN:
      return std::any_of(
        config.behavior.begin(), config.behavior.end(),
        [](const auto & condition) { return condition == "slow_down"; });
    case PlanningFactor::STOP:
      return std::any_of(
        config.behavior.begin(), config.behavior.end(),
        [](const auto & condition) { return condition == "stop"; });
    case PlanningFactor::SHIFT_LEFT:
      return std::any_of(
        config.behavior.begin(), config.behavior.end(),
        [](const auto & condition) { return condition == "shift_left"; });
    case PlanningFactor::SHIFT_RIGHT:
      return std::any_of(
        config.behavior.begin(), config.behavior.end(),
        [](const auto & condition) { return condition == "shift_right"; });
    case PlanningFactor::TURN_LEFT:
      return std::any_of(
        config.behavior.begin(), config.behavior.end(),
        [](const auto & condition) { return condition == "turn_left"; });
    case PlanningFactor::TURN_RIGHT:
      return std::any_of(
        config.behavior.begin(), config.behavior.end(),
        [](const auto & condition) { return condition == "turn_right"; });
  }

  return false;
}

bool is_target(
  const autoware_perception_msgs::msg::PredictedObject & object,
  const autoware_internal_planning_msgs::msg::PlanningFactor & factor,
  const rear_obstacle_checker_node::Params & parameters)
{
  using autoware_perception_msgs::msg::ObjectClassification;

  const auto config = parameters.scene_map.at(factor.module);

  const auto label = get_highest_prob_label(object.classification);
  switch (label) {
    case ObjectClassification::UNKNOWN:
      return std::any_of(
        config.target_type.begin(), config.target_type.end(),
        [](const auto & target_type) { return target_type == "unknown"; });
    case ObjectClassification::CAR:
      return std::any_of(
        config.target_type.begin(), config.target_type.end(),
        [](const auto & target_type) { return target_type == "car"; });
    case ObjectClassification::TRUCK:
      return std::any_of(
        config.target_type.begin(), config.target_type.end(),
        [](const auto & target_type) { return target_type == "truck"; });
    case ObjectClassification::TRAILER:
      return std::any_of(
        config.target_type.begin(), config.target_type.end(),
        [](const auto & target_type) { return target_type == "trailer"; });
    case ObjectClassification::BUS:
      return std::any_of(
        config.target_type.begin(), config.target_type.end(),
        [](const auto & target_type) { return target_type == "bus"; });
    case ObjectClassification::MOTORCYCLE:
      return std::any_of(
        config.target_type.begin(), config.target_type.end(),
        [](const auto & target_type) { return target_type == "motorcycle"; });
    case ObjectClassification::PEDESTRIAN:
      return std::any_of(
        config.target_type.begin(), config.target_type.end(),
        [](const auto & target_type) { return target_type == "pedestrian"; });
    case ObjectClassification::BICYCLE:
      return std::any_of(
        config.target_type.begin(), config.target_type.end(),
        [](const auto & target_type) { return target_type == "bicycle"; });
  }

  return false;
}

lanelet::ConstLanelets get_previous_lanes_recursively(
  const lanelet::ConstLanelet & lane, const double length, const double threshold,
  const std::shared_ptr<autoware::route_handler::RouteHandler> & route_handler)
{
  lanelet::ConstLanelets ret{lane};
  for (const auto & prev_lane : route_handler->getPreviousLanelets(lane)) {
    const double total_length = length + lanelet::utils::getLaneletLength2d(prev_lane);
    if (total_length < threshold) {
      const auto prev_lanes =
        get_previous_lanes_recursively(prev_lane, total_length, threshold, route_handler);
      ret.insert(ret.end(), prev_lanes.begin(), prev_lanes.end());
    }
  }

  return ret;
}

lanelet::ConstLanelets get_adjacent_lanes(
  const lanelet::ConstLanelets & current_lanes, const geometry_msgs::msg::Pose & vehicle_pose,
  const std::shared_ptr<autoware::route_handler::RouteHandler> & route_handler, const bool is_right,
  const double backward_distance)
{
  const auto ego_coordinate_on_arc = lanelet::utils::getArcCoordinates(current_lanes, vehicle_pose);

  lanelet::ConstLanelets lanes{};

  const auto exist_in_current_lane = [&current_lanes](const auto id) {
    const auto itr = std::find_if(
      current_lanes.begin(), current_lanes.end(),
      [&id](const auto & lane) { return lane.id() == id; });
    return itr != current_lanes.end();
  };

  const auto exist = [&lanes](const auto id) {
    const auto itr = std::find_if(
      lanes.begin(), lanes.end(), [&id](const auto & lane) { return lane.id() == id; });
    return itr != lanes.end();
  };

  double length = 0.0;
  for (const auto & lane : current_lanes) {
    const auto residual_length = backward_distance - ego_coordinate_on_arc.length + length;
    const auto opt_left_lane = route_handler->getLeftLanelet(lane, true, false);
    if (!is_right && opt_left_lane) {
      lanes.push_back(opt_left_lane.value());

      for (const auto & prev_lane : get_previous_lanes_recursively(
             opt_left_lane.value(), 0.0, residual_length, route_handler)) {
        if (!exist(prev_lane.id()) && !exist_in_current_lane(prev_lane.id())) {
          lanes.push_back(prev_lane);
        }
      }
    }

    const auto opt_right_lane = route_handler->getRightLanelet(lane, true, false);
    if (is_right && opt_right_lane) {
      lanes.push_back(opt_right_lane.value());

      for (const auto & prev_lane : get_previous_lanes_recursively(
             opt_right_lane.value(), 0.0, residual_length, route_handler)) {
        if (!exist(prev_lane.id()) && !exist_in_current_lane(prev_lane.id())) {
          lanes.push_back(prev_lane);
        }
      }
    }

    length += lanelet::utils::getLaneletLength2d(lane);
  }

  return lanes;
}

auto calculate_overhang_distance(
  const lanelet::ConstLanelets & lanelets, const geometry_msgs::msg::Pose & ego_pose,
  const autoware::vehicle_info_utils::VehicleInfo & vehicle_info)
  -> std::pair<double /*min*/, double /*max*/>
{
  const auto transform = autoware_utils::pose2transform(ego_pose);
  const auto footprint =
    autoware_utils::transform_vector(vehicle_info.createFootprint(), transform);

  std::vector<double> lateral_distances;
  for (const auto & p : footprint) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = p.x();
    pose.position.y = p.y();
    const auto ego_coordinate_on_arc = lanelet::utils::getArcCoordinates(lanelets, pose);
    // // const auto lanelet_point = lanelet::utils::conversion::toLaneletPoint(pose.position);
    // const auto arc_coordinates = lanelet::geometry::toArcCoordinates(
    //     lanelet::utils::to2D(lanelet.centerline()), p);
    //     // lanelet::utils::to2D(lanelt.centerline()),
    //     lanelet::utils::to2D(lanelet_point).basicPoint());
    lateral_distances.push_back(ego_coordinate_on_arc.distance);
  }

  std::sort(lateral_distances.begin(), lateral_distances.end());

  return std::make_pair(lateral_distances.front(), lateral_distances.back());
}

std::optional<lanelet::CompoundPolygon3d> generate_polygon(
  const lanelet::ConstLanelets & lanelets, const geometry_msgs::msg::Pose & ego_pose,
  const double forward_detection_length, const double backward_detection_length)
{
  const auto ego_coordinate_on_arc = lanelet::utils::getArcCoordinates(lanelets, ego_pose).length;

  return lanelet::utils::getPolygonFromArcLength(
    lanelets, ego_coordinate_on_arc - backward_detection_length,
    ego_coordinate_on_arc + forward_detection_length);
}

visualization_msgs::msg::MarkerArray create_polygon_marker_array(
  const lanelet::CompoundPolygon3d & polygon, const std::string & ns, const bool should_check,
  const bool is_safe)
{
  visualization_msgs::msg::MarkerArray msg;

  visualization_msgs::msg::Marker marker{};
  marker.header.frame_id = "map";

  marker.ns = ns;
  marker.id = 0L;
  marker.lifetime = rclcpp::Duration::from_seconds(0.3);
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.action = visualization_msgs::msg::Marker::ADD;
  marker.pose.orientation = autoware_utils::create_marker_orientation(0, 0, 0, 1.0);
  marker.scale = autoware_utils::create_marker_scale(0.1, 0.0, 0.0);
  if (should_check) {
    if (is_safe) {
      marker.color = autoware_utils::create_marker_color(0.0, 1.0, 0.0, 0.999);
    } else {
      marker.color = autoware_utils::create_marker_color(1.0, 0.0, 0.0, 0.999);
    }
  } else {
    marker.color = autoware_utils::create_marker_color(1.0, 1.0, 1.0, 0.999);
  }
  for (const auto & p : polygon) {
    geometry_msgs::msg::Point point;
    point.x = p.x();
    point.y = p.y();
    point.z = p.z();
    marker.points.push_back(point);
  }
  if (!marker.points.empty()) {
    marker.points.push_back(marker.points.front());
  }
  msg.markers.push_back(marker);

  return msg;
}

lanelet::ConstLanelet generateHalfLanelet(
  const lanelet::ConstLanelet lanelet, const bool is_right,
  const double ignore_width_from_centerline)
{
  lanelet::Points3d lefts, rights;

  const double offset = !is_right ? ignore_width_from_centerline : -ignore_width_from_centerline;
  const auto offset_centerline = lanelet::utils::getCenterlineWithOffset(lanelet, offset);

  const auto original_left_bound = !is_right ? lanelet.leftBound() : offset_centerline;
  const auto original_right_bound = !is_right ? offset_centerline : lanelet.rightBound();

  for (const auto & pt : original_left_bound) {
    lefts.emplace_back(pt);
  }
  for (const auto & pt : original_right_bound) {
    rights.emplace_back(pt);
  }
  const auto left_bound = lanelet::LineString3d(lanelet::InvalId, std::move(lefts));
  const auto right_bound = lanelet::LineString3d(lanelet::InvalId, std::move(rights));
  auto half_lanelet = lanelet::Lanelet(lanelet::InvalId, left_bound, right_bound);
  return half_lanelet;
}

lanelet::ConstLanelet generate_offset_lanelet(
  const lanelet::ConstLanelet lanelet, const double offset_near, const double offset_far,
  const bool is_right)
{
  lanelet::Points3d lefts, rights;

  const auto bound_near = lanelet::utils::getCenterlineWithOffset(lanelet, offset_near);
  const auto bound_far = lanelet::utils::getCenterlineWithOffset(lanelet, offset_far);

  const auto original_left_bound = is_right ? bound_near : bound_far;
  const auto original_right_bound = is_right ? bound_far : bound_near;

  for (const auto & pt : original_left_bound) {
    lefts.emplace_back(pt);
  }
  for (const auto & pt : original_right_bound) {
    rights.emplace_back(pt);
  }
  const auto left_bound = lanelet::LineString3d(lanelet::InvalId, std::move(lefts));
  const auto right_bound = lanelet::LineString3d(lanelet::InvalId, std::move(rights));
  auto half_lanelet = lanelet::Lanelet(lanelet::InvalId, left_bound, right_bound);
  return half_lanelet;
}

bool is_within_lane(
  const lanelet::ConstLanelet & closest_lanelet, const geometry_msgs::msg::Pose & ego_pose,
  const std::shared_ptr<autoware::route_handler::RouteHandler> & route_handler,
  const autoware::vehicle_info_utils::VehicleInfo & vehicle_info)
{
  const auto transform = autoware_utils::pose2transform(ego_pose);
  const auto footprint =
    autoware_utils::transform_vector(vehicle_info.createFootprint(), transform);

  lanelet::ConstLanelets concat_lanelets{};

  // push previous lanelet
  lanelet::ConstLanelets prev_lanelet;
  if (route_handler->getPreviousLaneletsWithinRoute(closest_lanelet, &prev_lanelet)) {
    concat_lanelets.push_back(prev_lanelet.front());
  }

  // push nearest lanelet
  {
    concat_lanelets.push_back(closest_lanelet);
  }

  // push next lanelet
  lanelet::ConstLanelet next_lanelet;
  if (route_handler->getNextLaneletWithinRoute(closest_lanelet, &next_lanelet)) {
    concat_lanelets.push_back(next_lanelet);
  }

  const auto combine_lanelet = lanelet::utils::combineLaneletsShape(concat_lanelets);

  return boost::geometry::within(footprint, combine_lanelet.polygon2d().basicPolygon());
}

MarkerArray showPolygon(
  const behavior_path_planner::utils::path_safety_checker::CollisionCheckDebugMap & obj_debug_vec,
  std::string && ns)
{
  if (obj_debug_vec.empty()) {
    return MarkerArray{};
  }

  int32_t id{0};
  const auto now = rclcpp::Clock{RCL_ROS_TIME}.now();

  constexpr float line_scale_val{0.2};
  const auto line_marker_scale =
    autoware_utils::create_marker_scale(line_scale_val, line_scale_val, line_scale_val);

  auto default_line_marker = [&](const auto & color = marker_utils::colors::green()) {
    return autoware_utils::create_default_marker(
      "map", now, ns, ++id, Marker::LINE_STRIP, line_marker_scale, color);
  };

  constexpr float text_scale_val{1.5};
  const auto text_marker_scale =
    autoware_utils::create_marker_scale(text_scale_val, text_scale_val, text_scale_val);

  auto default_text_marker = [&]() {
    return autoware_utils::create_default_marker(
      "map", now, ns + "_text", ++id, visualization_msgs::msg::Marker::TEXT_VIEW_FACING,
      text_marker_scale, marker_utils::colors::white());
  };

  auto default_cube_marker = [&](
                               const auto & width, const auto & depth,
                               const auto & color = marker_utils::colors::green()) {
    return autoware_utils::create_default_marker(
      "map", now, ns + "_cube", ++id, visualization_msgs::msg::Marker::CUBE,
      autoware_utils::create_marker_scale(width, depth, 1.0), color);
  };

  MarkerArray marker_array;
  marker_array.markers.reserve(
    obj_debug_vec.size() * 5);  // poly ego, text ego, poly obj, text obj, cube obj

  int32_t idx = {0};
  for (const auto & [uuid, info] : obj_debug_vec) {
    const auto color = info.is_safe ? marker_utils::colors::green() : marker_utils::colors::red();
    const auto poly_z = info.current_obj_pose.position.z;  // temporally

    const auto insert_polygon_marker = [&](const auto & polygon) {
      marker_array.markers.emplace_back();
      auto & polygon_marker = marker_array.markers.back();
      polygon_marker = default_line_marker(color);
      polygon_marker.points.reserve(polygon.outer().size());
      for (const auto & p : polygon.outer()) {
        polygon_marker.points.push_back(autoware_utils::create_point(p.x(), p.y(), poly_z));
      }
    };

    insert_polygon_marker(info.extended_ego_polygon);
    insert_polygon_marker(info.extended_obj_polygon);

    const auto str_idx = std::to_string(++idx);
    const auto insert_text_marker = [&](const auto & pose) {
      marker_array.markers.emplace_back();
      auto & text_marker = marker_array.markers.back();
      text_marker = default_text_marker();
      text_marker.text = str_idx;
      text_marker.pose = pose;
    };

    insert_text_marker(info.expected_ego_pose);
    insert_text_marker(info.expected_obj_pose);

    const auto insert_cube_marker = [&](const auto & pose) {
      marker_array.markers.emplace_back();
      auto & cube_marker = marker_array.markers.back();
      cube_marker = default_cube_marker(1.0, 1.0, color);
      cube_marker.pose = pose;
    };
    insert_cube_marker(info.current_obj_pose);
  }
  return marker_array;
}

MarkerArray showPredictedPath(
  const behavior_path_planner::utils::path_safety_checker::CollisionCheckDebugMap & obj_debug_vec,
  std::string && ns)
{
  int32_t id{0};
  const auto current_time{rclcpp::Clock{RCL_ROS_TIME}.now()};
  const auto arrow_marker_scale = autoware_utils::create_marker_scale(1.0, 0.3, 0.3);
  const auto default_arrow_marker = [&](const auto & color) {
    return autoware_utils::create_default_marker(
      "map", current_time, ns, ++id, Marker::ARROW, arrow_marker_scale, color);
  };

  MarkerArray marker_array;
  marker_array.markers.reserve(std::accumulate(
    obj_debug_vec.cbegin(), obj_debug_vec.cend(), 0UL,
    [&](const auto current_sum, const auto & obj_debug) {
      const auto & [uuid, info] = obj_debug;
      return current_sum + info.ego_predicted_path.size() + info.obj_predicted_path.size() + 2;
    }));

  for (const auto & [uuid, info] : obj_debug_vec) {
    const auto insert_marker = [&](const auto & path, const auto & color) {
      for (const auto & pose : path) {
        marker_array.markers.emplace_back();
        auto & marker = marker_array.markers.back();
        marker = default_arrow_marker(color);
        marker.pose = pose.pose;
      }
    };

    insert_marker(info.ego_predicted_path, marker_utils::colors::aqua());
    insert_marker(info.obj_predicted_path, marker_utils::colors::yellow());
    const auto insert_expected_pose_marker = [&](const auto & pose, const auto & color) {
      // instead of checking for distance, inserting a new marker might be more efficient
      marker_array.markers.emplace_back();
      auto & marker = marker_array.markers.back();
      marker = default_arrow_marker(color);
      marker.pose = pose;
      marker.pose.position.z += 0.05;
    };

    insert_expected_pose_marker(info.expected_ego_pose, marker_utils::colors::red());
    insert_expected_pose_marker(info.expected_obj_pose, marker_utils::colors::red());
  }
  return marker_array;
}

MarkerArray showSafetyCheckInfo(
  const behavior_path_planner::utils::path_safety_checker::CollisionCheckDebugMap & obj_debug_vec,
  std::string && ns)
{
  int32_t id{0};
  auto default_text_marker = [&]() {
    return autoware_utils::create_default_marker(
      "map", rclcpp::Clock{RCL_ROS_TIME}.now(), ns, ++id, Marker::TEXT_VIEW_FACING,
      autoware_utils::create_marker_scale(0.5, 0.5, 0.5), marker_utils::colors::aqua());
  };

  MarkerArray marker_array;

  marker_array.markers.reserve(obj_debug_vec.size());

  int idx{0};

  for (const auto & [uuid, info] : obj_debug_vec) {
    auto safety_check_info_text = default_text_marker();
    safety_check_info_text.pose = info.current_obj_pose;

    std::ostringstream ss;

    ss << "Idx: " << ++idx << "\nUnsafe reason: " << info.unsafe_reason
       << "\nRSS dist: " << std::setprecision(4) << info.rss_longitudinal
       << "\nEgo to obj: " << info.inter_vehicle_distance
       << "\nExtended polygon: " << (info.is_front ? "ego" : "object")
       << "\nExtended polygon lateral offset: " << info.lat_offset
       << "\nExtended polygon forward longitudinal offset: " << info.forward_lon_offset
       << "\nExtended polygon backward longitudinal offset: " << info.backward_lon_offset
       << "\nLast checked position: " << (info.is_front ? "obj in front ego" : "obj at back ego")
       << "\nSafe: " << (info.is_safe ? "Yes" : "No");

    safety_check_info_text.text = ss.str();
    marker_array.markers.push_back(safety_check_info_text);
  }
  return marker_array;
}
}  // namespace autoware::rear_obstacle_checker::utils

#endif  // UTILS_HPP_
