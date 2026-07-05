import os
import sys
import mujoco
import xml.etree.ElementTree as ET

robot_file = './tests/scenes/shelf_structured/scene100.xml'
# robot_file = "tests/scenes/adjusted/tabletop_unstructured_14_0.xml"
in_file = sys.argv[1]
out_file = sys.argv[2]

## include robot.xml in temp xml file ##
robot_root = ET.parse(robot_file).getroot()
root = ET.parse(in_file).getroot()

robot_body = robot_root.find('.//body[@name="base"]')
world_body = root.find('.//worldbody')
for child in world_body:
    if child.tag == 'body' and child.attrib.get('name') == 'base':
        world_body.remove(child)
        break
world_body.insert(2, robot_body)

out_string = ET.tostring(root).decode('utf-8').replace(' />', '/>')
with open(out_file, 'w') as f:
    f.write(out_string)
