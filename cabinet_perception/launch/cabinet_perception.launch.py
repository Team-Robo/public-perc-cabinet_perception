"""Cabinet perception + static map -> cabinet TF from the prior.

  ros2 launch cabinet_perception cabinet_perception.launch.py                    # real camera
  ros2 launch cabinet_perception cabinet_perception.launch.py synthetic:=true    # no robot, 4 x left/right
  ros2 launch cabinet_perception cabinet_perception.launch.py synthetic:=true synthetic_scene:=grid   # 3 x 3
  ros2 launch cabinet_perception cabinet_perception.launch.py localize:=true relocalize:=true live_period:=1.0   # handheld
  ros2 service call /cabinet_perception/scan std_srvs/srv/Trigger

localize:=true estimates the cabinet pose from the cloud instead of the prior's
pose, so the static map -> cabinet TF is not published (the node broadcasts it).
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

SHARE = get_package_share_directory("cabinet_perception")
SYNTH_PRIORS = {"demo": "cabinet_prior.yaml", "grid": "synthetic_grid_prior.yaml"}


def _setup(context):
    lc = lambda n: context.launch_configurations.get(n, "")  # noqa: E731
    synthetic = lc("synthetic").lower() == "true"
    localize = lc("localize").lower() == "true"
    prior_file = lc("prior_file")
    if not prior_file:
        name = SYNTH_PRIORS.get(lc("synthetic_scene"), "cabinet_prior.yaml") if synthetic else "cabinet_prior.yaml"
        prior_file = os.path.join(SHARE, "config", name)
    actions = []
    if not localize:
        with open(prior_file) as f:
            cab = (yaml.safe_load(f) or {}).get("cabinet", {})
        pose = cab.get("pose", {}) or {}
        xyz = [str(v) for v in pose.get("xyz", [0, 0, 0])]
        r, p, y = (str(v) for v in pose.get("rpy", [0, 0, 0]))
        actions.append(Node(package="tf2_ros", executable="static_transform_publisher", name="cabinet_prior_tf",
                            arguments=["--x", xyz[0], "--y", xyz[1], "--z", xyz[2], "--roll", r, "--pitch", p,
                                       "--yaw", y, "--frame-id", cab.get("parent_frame", "map"),
                                       "--child-frame-id", cab.get("frame_id", "cabinet")]))
    if synthetic:
        actions.append(Node(package="cabinet_perception", executable="synthetic_camera", name="synthetic_camera",
                            parameters=[{"scene": lc("synthetic_scene")}]))
    params = {"object_list": lc("object_list"), "localize": localize, "fixed_frame": lc("fixed_frame"),
              "relocalize": lc("relocalize").lower() == "true",
              "live_period": float(lc("live_period"))}
    # localize with no prior given: the default prior is a dummy, take the size from the cloud too
    if not (localize and not lc("prior_file")):
        params["prior_file"] = prior_file
    actions.append(Node(package="cabinet_perception", executable="cabinet_node", name="cabinet_perception",
                        output="screen", parameters=[lc("params_file"), params]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("prior_file", default_value="",
                              description="setup-day prior; empty = config/cabinet_prior.yaml (or the "
                                          "synthetic scene's prior; with localize:=true, size from the cloud)"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(SHARE, "config", "params.yaml")),
        DeclareLaunchArgument("object_list", default_value="",
                              description="object_list.yaml (name -> category) shared with object_grouping"),
        DeclareLaunchArgument("synthetic", default_value="false",
                              description="publish a synthetic cabinet instead of a real camera"),
        DeclareLaunchArgument("synthetic_scene", default_value="demo", description="demo (4 x 2) | grid (3 x 3)"),
        DeclareLaunchArgument("localize", default_value="false",
                              description="estimate fixed_frame -> cabinet from the cloud (no prior pose)"),
        DeclareLaunchArgument("relocalize", default_value="false",
                              description="with localize: re-estimate the cabinet on every scan (handheld)"),
        DeclareLaunchArgument("fixed_frame", default_value="map"),
        DeclareLaunchArgument("live_period", default_value="0.0",
                              description="> 0: rebuild from the newest frame every N s (visualisation)"),
        OpaqueFunction(function=_setup),
    ])
