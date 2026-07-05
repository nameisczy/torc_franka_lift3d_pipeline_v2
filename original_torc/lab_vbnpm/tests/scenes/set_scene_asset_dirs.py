import re
import os
import rospkg
import xml.etree.ElementTree as ET 

# INPUT: GC6D directory, relative to lab repo
# OUTPUT: Modified xmls

def set_asset_dirs(xml_path, mesh_dir, texture_dir):
    """
    Set the mesh and texture directories for the MJCF
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    com = root.findall('.//compiler')

    if len(com) == 0:
        com = ET.SubElement(root, 'compiler')
    else:
        # There should only be one compiler
        com = com[0]

    com.set('meshdir', mesh_dir)
    com.set('texturedir', texture_dir)

    ET.indent(tree, space="    ", level=0)

    xml_string = ET.tostring(root, encoding="unicode")



    xml_string = xml_string.replace(substr, new_substr)
    # xml_string now contains literal characters
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_string)

# Set correct asset dir
substr = "models_obj_m/"
new_substr = "models/objects/GraspClutter6D/models_obj_m/"


GC6D_relative_to_lab = "models/objects/GraspClutter6D"
GC6D_relative_to_lab = f"/data/local/kc1317/workspace/src/lab_vbnpm/"#{GC6D_relative_to_lab}"
scene_base_path = "/data/local/kc1317/graspclutter6d_mujoco_sim/scenes"

subdirs = ["shelf_structured/", 
           "shelf_unstructured/", 
           "tabletop_structured/", 
           "tabletop_unstructured/"]

for subdir in subdirs:
    abs_subdir_path = f"{scene_base_path}/{subdir}"
    
    # Get all files in the subdir
    for file in os.listdir(f"{abs_subdir_path}"):
        # Check if the file is an XML file
        if file.endswith(".xml"):
            # set_asset_dirs(f"")
            abs_file_path = f"{abs_subdir_path}/{file}"
            set_asset_dirs(xml_path=abs_file_path,
                           mesh_dir=GC6D_relative_to_lab,
                           texture_dir=GC6D_relative_to_lab)