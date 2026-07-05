import os
import rospkg
import xml.etree.ElementTree as ET

# INPUT: provide path to lab repo
# This can be a relative or absolute path; for a relative path, it should be relative to the GraspClutter6D path

rp = rospkg.RosPack()
lab_path = rp.get_path('lab_vbnpm')

subdirs = ["structured_shelf/", 
           "unstructured_shelf/", 
           "structured_tabletop/", 
           "unstructured_tabletop/"]

for subdir in subdirs:
    abs_subdir_path = f"{lab_path}/tests/scenes1/{subdir}"
    
    # Get all files in the subdir
    for file in os.listdir(f"{abs_subdir_path}"):
        # Check if the file is an XML file
        if file.endswith(".xml"):
            # Parse the XML file
            tree = ET.parse(os.path.join(abs_subdir_path, file))
            root = tree.getroot()

            # Find all elements (recursively) with the attribute 'file' with value containing "lab"
            for elem in root.iter():
                if 'file' in elem.attrib and 'lab_vbnpm' in elem.attrib['file']:
                    print(elem)

                    # Get the substring starting after "lab_vbnpm"
                    start_index = elem.attrib['file'].index("lab_vbnpm") + len("lab_vbnpm") 
                    rel_path = elem.attrib['file'][start_index:]   

                    print(rel_path)

                    # Update the 'file' attribute with the new path
                    new_path = os.path.join(lab_path, rel_path)
                    elem.set('file', new_path)