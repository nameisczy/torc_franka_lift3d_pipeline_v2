#!/usr/bin/env python
"""
Script to add inactive weld equality constraints to a MuJoCo XML scene file.

For each body with a name prefix "obj_", adds a weld constraint between
that body and the "ee_right" body with active="false".

Usage:
    python add_weld_constraints.py input.xml [output.xml]
    
If output.xml is not specified, the input file will be modified in-place.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def find_obj_bodies(root: ET.Element) -> list[str]:
    """Find all body names that start with 'obj_'."""
    obj_bodies = []
    for body in root.iter("body"):
        name = body.get("name", "")
        if name.startswith("obj_"):
            obj_bodies.append(name)
    return obj_bodies


def find_or_create_equality(root: ET.Element) -> ET.Element:
    """Find or create the <equality> element in the MuJoCo XML."""
    # Look for existing equality element
    equality = root.find("equality")
    if equality is not None:
        return equality
    
    # Create new equality element
    # Insert it after worldbody if possible, otherwise at the end
    equality = ET.Element("equality")
    
    # Find worldbody index to insert after it
    children = list(root)
    worldbody_idx = None
    for i, child in enumerate(children):
        if child.tag == "worldbody":
            worldbody_idx = i
            break
    
    if worldbody_idx is not None:
        root.insert(worldbody_idx + 1, equality)
    else:
        root.append(equality)
    
    return equality


def get_existing_welds(equality: ET.Element) -> set[str]:
    """Get names of existing weld constraints."""
    existing = set()
    for weld in equality.findall("weld"):
        name = weld.get("name", "")
        if name:
            existing.add(name)
    return existing


def add_weld_constraints(xml_path: str, output_path: str = None) -> int:
    """
    Add inactive weld constraints between obj_* bodies and ee_right.
    
    Args:
        xml_path: Path to input XML file
        output_path: Path to output XML file (if None, overwrites input)
        
    Returns:
        Number of weld constraints added
    """
    if output_path is None:
        output_path = xml_path
    
    # Parse the XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find all obj_ bodies
    obj_bodies = find_obj_bodies(root)
    if not obj_bodies:
        print(f"No bodies with 'obj_' prefix found in {xml_path}")
        return 0
    
    print(f"Found {len(obj_bodies)} obj_ bodies: {obj_bodies}")
    
    # Find or create equality element
    equality = find_or_create_equality(root)
    
    # Get existing weld names to avoid duplicates
    existing_welds = get_existing_welds(equality)
    
    # Add weld constraints
    added = 0
    for body_name in obj_bodies:
        weld_name = f"{body_name}_weld"
        
        if weld_name in existing_welds:
            print(f"  Skipping {weld_name} (already exists)")
            continue
        
        weld = ET.SubElement(equality, "weld")
        weld.set("name", weld_name)
        weld.set("body1", body_name)
        weld.set("body2", "ee_right")
        weld.set("active", "false")
        
        print(f"  Added {weld_name}")
        added += 1
    
    # Write output with nice formatting
    indent_xml(root)
    tree.write(output_path, encoding="unicode", xml_declaration=False)
    
    # Add XML declaration manually for proper formatting
    with open(output_path, "r") as f:
        content = f.read()
    
    print(f"\nAdded {added} weld constraints to {output_path}")
    return added


def indent_xml(elem: ET.Element, level: int = 0):
    """Add indentation to XML elements for pretty printing."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_path).exists():
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)
    
    add_weld_constraints(input_path, output_path)


if __name__ == "__main__":
    main()