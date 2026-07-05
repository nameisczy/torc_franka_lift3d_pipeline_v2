import json
from rospkg import RosPack
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.types.file_path import ContentPath
from curobo.cuda_robot_model.util import load_robot_yaml

rp = RosPack()
root = rp.get_path('lab_vbnpm')
root += '/robots/motoman/curobo/'

content_path = ContentPath(
    robot_config_absolute_path=root + 'motoman.yml',
    robot_urdf_absolute_path=root + 'motoman.urdf',
    robot_usd_absolute_path=root + 'motoman.usd',
    robot_asset_absolute_path=root,
)

robot_config = load_robot_yaml(content_path)
print(json.dumps(robot_config['robot_cfg'],indent=4))
print(robot_config['robot_cfg']['kinematics']['ee_link'])
# robot_config = RobotConfig.from_dict(robot_config, TensorDeviceType())
# print(robot_config.kinematics)
