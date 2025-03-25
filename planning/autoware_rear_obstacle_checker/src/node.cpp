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

#include "node.hpp"

#include "utils.hpp"

#include <autoware_lanelet2_extension/visualization/visualization.hpp>

#include <lanelet2_core/Forward.h>
#include <lanelet2_core/geometry/Polygon.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>
#endif

#include <memory>
#include <string>
#include <utility>
#include <vector>

#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace autoware::rear_obstacle_checker
{

using std::chrono_literals::operator""ms;

RearObstacleCheckerNode::RearObstacleCheckerNode(const rclcpp::NodeOptions & node_options)
: Node("rear_obstacle_checker_node", node_options),
  timer_{rclcpp::create_timer(
    this, get_clock(), 100ms, std::bind(&RearObstacleCheckerNode::on_timer, this))},
  tf_buffer_{this->get_clock()},
  tf_listener_{tf_buffer_},
  pub_debug_marker_{this->create_publisher<MarkerArray>("~/debug_marker", 20)},
  route_handler_{std::make_shared<autoware::route_handler::RouteHandler>()},
  param_listener_{std::make_shared<rear_obstacle_checker_node::ParamListener>(
    this->get_node_parameters_interface())},
  diag_updater_{std::make_unique<diagnostic_updater::Updater>(this)},
  vehicle_info_{autoware::vehicle_info_utils::VehicleInfoUtils(*this).getVehicleInfo()}
{
  for (const auto & [key, param] : param_listener_->get_params().scene_map) {
    sub_planning_factor_map_.emplace(
      key, autoware_utils::InterProcessPollingSubscriber<PlanningFactorArray>{this, param.topic});
  }

  diag_updater_->setHardwareID("rear_obstacle_checker");
  diag_updater_->add("collision_risk", this, &RearObstacleCheckerNode::update);
}

void RearObstacleCheckerNode::take_data()
{
  // route
  {
    const auto msg = sub_route_.take_data();
    if (msg) {
      if (msg->segments.empty()) {
        RCLCPP_ERROR(get_logger(), "input route is empty. ignored");
      } else {
        route_handler_->setRoute(*msg);
      }
    }
  }

  // map
  {
    const auto msg = sub_lanelet_map_bin_.take_data();
    if (msg) {
      route_handler_->setMap(*msg);
    }
  }

  // odometry
  {
    odometry_ptr_ = sub_odometry_.take_data();
  }

  // objects
  {
    object_ptr_ = sub_dynamic_objects_.take_data();
  }

  // pointcloud
  {
    pointcloud_ptr_ = sub_pointcloud_.take_data();
  }

  // trajectory
  {
    path_ptr_ = sub_path_.take_data();
  }

  {
    PlanningFactorArray planning_factors;
    for (auto & [name, subscriber] : sub_planning_factor_map_) {
      const auto msg = subscriber.take_data();
      if (!msg) {
        continue;
      }

      planning_factors.factors.insert(
        planning_factors.factors.end(), msg->factors.begin(), msg->factors.end());
    }

    factors_ptr_ = std::make_shared<PlanningFactorArray>(planning_factors);
  }
}

bool RearObstacleCheckerNode::is_ready() const
{
  if (!route_handler_->isHandlerReady()) {
    return false;
  }

  if (!odometry_ptr_) {
    return false;
  }

  if (!object_ptr_) {
    return false;
  }

  if (!path_ptr_) {
    return false;
  }

  return true;
}

void RearObstacleCheckerNode::update(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  take_data();

  if (!is_ready()) {
    return;
  }

  DebugData debug_data;

  if (!is_safe(debug_data)) {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "obstacles exist beside ego");
  } else {
    stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "validated.");
  }

  publish_marker(debug_data);
}

void RearObstacleCheckerNode::on_timer()
{
  diag_updater_->force_update();
}

auto RearObstacleCheckerNode::generate_detection_area(
  const PlanningFactor & factor, const lanelet::ConstLanelets & current_lanes) const
  -> lanelet::ConstLanelets
{
  const auto p = param_listener_->get_params();

  const auto config = p.scene_map.at(factor.module);

  lanelet::ConstLanelets detection_lanes{};

  if (factor.behavior == PlanningFactor::SHIFT_LEFT) {
    if (config.adjacent_lane) {
      const auto adjacent_lanes = utils::get_adjacent_lanes(
        current_lanes, odometry_ptr_->pose.pose, route_handler_, false, p.common.range.backward);
      detection_lanes.insert(detection_lanes.end(), adjacent_lanes.begin(), adjacent_lanes.end());
    }
    if (config.current_lane) {
      const auto half_lanes = [&current_lanes, this]() {
        lanelet::ConstLanelets ret{};
        for (const auto & lane : current_lanes) {
          ret.push_back(
            utils::generateHalfLanelet(lane, false, 0.5 * vehicle_info_.vehicle_width_m));
        }
        return ret;
      }();
      detection_lanes.insert(detection_lanes.end(), half_lanes.begin(), half_lanes.end());
    }
  }

  if (factor.behavior == PlanningFactor::TURN_LEFT) {
    if (config.adjacent_lane) {
      const auto adjacent_lanes = utils::get_adjacent_lanes(
        current_lanes, odometry_ptr_->pose.pose, route_handler_, false, p.common.range.backward);
      detection_lanes.insert(detection_lanes.end(), adjacent_lanes.begin(), adjacent_lanes.end());
    }
    if (config.current_lane) {
      const auto half_lanes = [&current_lanes, this]() {
        lanelet::ConstLanelets ret{};
        for (const auto & lane : current_lanes) {
          ret.push_back(
            utils::generateHalfLanelet(lane, false, 0.5 * vehicle_info_.vehicle_width_m));
        }
        return ret;
      }();
      detection_lanes.insert(detection_lanes.end(), half_lanes.begin(), half_lanes.end());
    }
  }

  if (factor.behavior == PlanningFactor::SHIFT_RIGHT) {
    if (config.adjacent_lane) {
      const auto adjacent_lanes = utils::get_adjacent_lanes(
        current_lanes, odometry_ptr_->pose.pose, route_handler_, true, p.common.range.backward);
      detection_lanes.insert(detection_lanes.end(), adjacent_lanes.begin(), adjacent_lanes.end());
    }
    if (config.current_lane) {
      const auto half_lanes = [&current_lanes, this]() {
        lanelet::ConstLanelets ret{};
        for (const auto & lane : current_lanes) {
          ret.push_back(
            utils::generateHalfLanelet(lane, true, 0.5 * vehicle_info_.vehicle_width_m));
        }
        return ret;
      }();
      detection_lanes.insert(detection_lanes.end(), half_lanes.begin(), half_lanes.end());
    }
  }

  if (factor.behavior == PlanningFactor::TURN_RIGHT) {
    if (config.adjacent_lane) {
      const auto adjacent_lanes = utils::get_adjacent_lanes(
        current_lanes, odometry_ptr_->pose.pose, route_handler_, true, p.common.range.backward);
      detection_lanes.insert(detection_lanes.end(), adjacent_lanes.begin(), adjacent_lanes.end());
    }
    if (config.current_lane) {
      const auto half_lanes = [&current_lanes, this]() {
        lanelet::ConstLanelets ret{};
        for (const auto & lane : current_lanes) {
          ret.push_back(
            utils::generateHalfLanelet(lane, true, 0.5 * vehicle_info_.vehicle_width_m));
        }
        return ret;
      }();
      detection_lanes.insert(detection_lanes.end(), half_lanes.begin(), half_lanes.end());
    }
  }

  return detection_lanes;
}

bool RearObstacleCheckerNode::is_safe(DebugData & debug)
{
  lanelet::ConstLanelet closest_lanelet;
  if (!route_handler_->getClosestLaneletWithinRoute(odometry_ptr_->pose.pose, &closest_lanelet)) {
    return true;
  }

  const auto p = param_listener_->get_params();

  const auto current_lanes = route_handler_->getLaneletSequence(
    closest_lanelet, odometry_ptr_->pose.pose, p.common.range.forward, p.common.range.backward);

  const auto is_with_current_lane =
    utils::is_within_lane(closest_lanelet, odometry_ptr_->pose.pose, route_handler_, vehicle_info_);
  if (!is_with_current_lane) {
    return true;
  }

  pcl::PointCloud<pcl::PointXYZ> transformed_pointcloud;
  if (!pointcloud_ptr_->data.empty()) {
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
      transform_stamped = tf_buffer_.lookupTransform(
        "map", pointcloud_ptr_->header.frame_id, pointcloud_ptr_->header.stamp,
        rclcpp::Duration::from_seconds(0.1));
    } catch (tf2::TransformException & e) {
      RCLCPP_WARN(get_logger(), "no transform found for no_ground_pointcloud: %s", e.what());
    }

    Eigen::Affine3f isometry = tf2::transformToEigen(transform_stamped.transform).cast<float>();
    pcl::fromROSMsg(*pointcloud_ptr_, transformed_pointcloud);
    autoware_utils::transform_pointcloud(transformed_pointcloud, transformed_pointcloud, isometry);
  }

  PredictedObjects objects;
  for (const auto & factor : factors_ptr_->factors) {
    if (p.scene_map.count(factor.module) == 0) {
      continue;
    }

    if (!utils::should_activate(factor, p)) {
      continue;
    }

    if (factor.control_points.empty()) {
      continue;
    }

    const auto detection_lanes = generate_detection_area(factor, current_lanes);
    debug.detection_lanes.insert(
      debug.detection_lanes.end(), detection_lanes.begin(), detection_lanes.end());

    const auto [targets, others] =
      behavior_path_planner::utils::path_safety_checker::separateObjectsByLanelets(
        *object_ptr_, detection_lanes,
        [&factor, &p](const auto & obj, const auto & lane, const auto yaw_threshold = M_PI_2) {
          if (!utils::is_target(obj, factor, p)) {
            return false;
          }
          return behavior_path_planner::utils::path_safety_checker::isPolygonOverlapLanelet(
            obj, lane, yaw_threshold);
        });
    objects.objects.insert(objects.objects.end(), targets.objects.begin(), targets.objects.end());

    const auto danger_points = utils::get_obstacle_points(detection_lanes, transformed_pointcloud);
    debug.obstacle_points = danger_points;
  }

  const auto now = this->now();
  if (is_safe(objects, debug)) {
    last_safe_time_ = now;
    if ((now - last_unsafe_time_).seconds() > p.common.off_time_buffer) {
      return true;
    }
  } else {
    if ((now - last_safe_time_).seconds() < p.common.on_time_buffer) {
      return true;
    }
  }

  return false;
}

bool RearObstacleCheckerNode::is_safe(const PredictedObjects & objects, DebugData & debug) const
{
  const auto p = param_listener_->get_params();
  // const auto ego_coordinate_on_arc =
  //   lanelet::utils::getArcCoordinates(lanelets, odometry_ptr_->pose.pose);

  const auto ego_predicted_path_params =
    std::make_shared<behavior_path_planner::utils::path_safety_checker::EgoPredictedPathParams>(
      get_predicted_path_params());

  std::vector<behavior_path_planner::utils::path_safety_checker::ExtendedPredictedObject>
    target_objects;

  std::for_each(objects.objects.begin(), objects.objects.end(), [&](const auto & object) {
    target_objects.push_back(behavior_path_planner::utils::path_safety_checker::transform(
      object, p.common.predicted_path.time_horizon, p.common.predicted_path.time_resolution));
  });

  const bool limit_to_max_velocity = false;
  const size_t ego_seg_idx =
    autoware::motion_utils::findFirstNearestSegmentIndexWithSoftConstraints(
      path_ptr_->points, odometry_ptr_->pose.pose, 1.0, M_PI_2);
  const auto ego_predicted_path =
    behavior_path_planner::utils::path_safety_checker::createPredictedPath(
      ego_predicted_path_params, path_ptr_->points, odometry_ptr_->pose.pose,
      odometry_ptr_->twist.twist.linear.x, ego_seg_idx, true, limit_to_max_velocity);

  // if
  // (!behavior_path_planner::utils::path_safety_checker::checkSafetyWithIntegralPredictedPolygon(
  //       ego_predicted_path, vehicle_info_, target_objects, true, get_integral_params(),
  //       debug.collision_check)) {
  //   return false;
  // }

  for (const auto & object : target_objects) {
    auto current_debug_data =
      behavior_path_planner::utils::path_safety_checker::createObjectDebug(object);

    const auto obj_polygon = autoware_utils::to_polygon2d(object.initial_pose, object.shape);

    const auto is_object_front =
      behavior_path_planner::utils::path_safety_checker::isTargetObjectFront(
        odometry_ptr_->pose.pose, obj_polygon, vehicle_info_.max_longitudinal_offset_m);
    if (is_object_front) {
      continue;
    }

    const auto obj_predicted_paths =
      behavior_path_planner::utils::path_safety_checker::getPredictedPathFromObj(object, false);

    for (const auto & obj_path : obj_predicted_paths) {
      if (!behavior_path_planner::utils::path_safety_checker::checkCollision(
            *path_ptr_, ego_predicted_path, object, obj_path, get_vehicle_params(),
            get_rss_params(), 1.0, M_PI_2, current_debug_data.second)) {
        behavior_path_planner::utils::path_safety_checker::updateCollisionCheckDebugMap(
          debug.collision_check, current_debug_data, false);

        return false;
      }
    }
    behavior_path_planner::utils::path_safety_checker::updateCollisionCheckDebugMap(
      debug.collision_check, current_debug_data, true);
  }

  return true;
}

void RearObstacleCheckerNode::publish_marker(const DebugData & debug) const
{
  MarkerArray msg;

  const auto add = [&msg](const MarkerArray & added) {
    autoware_utils::append_marker_array(added, &msg);
  };

  {
    add(lanelet::visualization::laneletsAsTriangleMarkerArray(
      "detection_lanes", debug.detection_lanes,
      autoware_utils::create_marker_color(1.0, 0.0, 0.0, 0.2)));
  }

  {
    add(utils::createPointsMarkerArray(debug.obstacle_points, "obstacle_points"));
  }

  {
    add(utils::showSafetyCheckInfo(debug.collision_check, "object_debug_info"));
    add(utils::showPredictedPath(debug.collision_check, "ego_predicted_path"));
    add(utils::showPolygon(debug.collision_check, "ego_and_target_polygon_relation"));
  }

  std::for_each(msg.markers.begin(), msg.markers.end(), [](auto & marker) {
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
  });

  pub_debug_marker_->publish(msg);
}
}  // namespace autoware::rear_obstacle_checker

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::rear_obstacle_checker::RearObstacleCheckerNode)
