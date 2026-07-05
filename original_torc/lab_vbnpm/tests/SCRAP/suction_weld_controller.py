#!/usr/bin/env python3
"""
Weld-based Suction Controller for MuJoCo
==========================================

This module provides a controller for simulating suction using weld equality constraints.
Instead of using adhesion actuators, this approach dynamically enables/disables weld
constraints between the suction cup and nearby objects.

Usage Example:
    import mujoco
    import mujoco.viewer
    
    model = mujoco.MjModel.from_xml_path("TEST.xml")
    data = mujoco.MjData(model)
    
    controller = SuctionWeldController(model, data)
    
    # In your control loop:
    controller.update(data, suction_active=True)
"""

import numpy as np
import mujoco


class SuctionWeldController:
    """Controller for weld-based suction simulation."""
    
    def __init__(self, model, data, distance_threshold=1.0):
        """
        Initialize the suction controller.
        
        Args:
            model: MuJoCo model (mjModel)
            data: MuJoCo data (mjData)
            distance_threshold: Maximum distance (m) for suction activation
        """
        self.model = model
        self.distance_threshold = distance_threshold
        
        # Get suction site ID
        self.suction_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "suction_site"
        )
        
        # Build mapping of object sites to weld constraint IDs
        self.object_mapping = {}
        object_names = ["obj_0", "obj_1", "obj_2", "obj_3", "obj_4", "obj_5", "obj_6"]
        
        for obj_name in object_names:
            site_name = f"{obj_name}_site"
            weld_name = f"weld_{obj_name}"
            
            try:
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
                self.object_mapping[obj_name] = {
                    'site_id': site_id,
                    'weld_id': weld_id
                }
            except Exception as e:
                print(f"Warning: Could not find {obj_name}: {e}")
        
        self.current_grasped_object = None
        print(f"Initialized SuctionWeldController with {len(self.object_mapping)} objects")
    
    def get_closest_object(self, data):
        """
        Find the closest object within suction range.
        
        Args:
            data: MuJoCo data (mjData)
            
        Returns:
            Tuple of (object_name, distance) or (None, inf) if no object in range
        """
        suction_pos = data.site_xpos[self.suction_site_id]
        
        min_dist = float('inf')
        closest_obj = None
        
        for obj_name, obj_info in self.object_mapping.items():
            obj_pos = data.site_xpos[obj_info['site_id']]
            dist = np.linalg.norm(suction_pos - obj_pos)
            
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj_name
        
        if min_dist <= self.distance_threshold:
            return closest_obj, min_dist
        else:
            return None, min_dist
    
    def activate_suction(self, data, object_name):
        """
        Activate suction by enabling the weld constraint for the specified object.
        
        Args:
            data: MuJoCo data (mjData)
            object_name: Name of the object to grasp
            
        Returns:
            bool: True if suction was activated successfully
        """
        if object_name not in self.object_mapping:
            print(f"Warning: Unknown object '{object_name}'")
            return False
        
        weld_id = self.object_mapping[object_name]['weld_id']
        
        # Debug: print site positions before activation
        suction_pos = data.site_xpos[self.suction_site_id]
        obj_site_id = self.object_mapping[object_name]['site_id']
        obj_pos = data.site_xpos[obj_site_id]
        distance = np.linalg.norm(suction_pos - obj_pos)
        
        print(f"Activating weld {weld_id} for {object_name}")
        print(f"  Suction pos: {suction_pos}")
        print(f"  Object pos:  {obj_pos}")
        print(f"  Distance:    {distance:.4f}m")
        
        self.model.eq_active0[weld_id] = 1
        data.eq_active[weld_id] = 1  # Set in data for immediate effect
        print(f"  eq_active0[{weld_id}] = {self.model.eq_active0[weld_id]}")
        print(f"  eq_active[{weld_id}] = {data.eq_active[weld_id]}")
        
        self.current_grasped_object = object_name
        
        print(f"✓ Suction activated on {object_name}")
        return True
    
    def deactivate_suction(self, data):
        """
        Deactivate suction by disabling all weld constraints.
        
        Args:
            data: MuJoCo data (mjData)
        
        Returns:
            bool: True if suction was deactivated
        """
        if self.current_grasped_object is None:
            return False
        
        weld_id = self.object_mapping[self.current_grasped_object]['weld_id']
        self.model.eq_active0[weld_id] = 0
        data.eq_active[weld_id] = 0  # Set in data for immediate effect
        
        print(f"✗ Suction deactivated from {self.current_grasped_object}")
        self.current_grasped_object = None
        return True
    
    def update(self, data, suction_active):
        """
        Update suction state based on desired activation and proximity to objects.
        
        This is the main method to call in your control loop.
        
        Args:
            data: MuJoCo data (mjData)
            suction_active: bool, True to activate/maintain suction, False to deactivate
        """
        if suction_active:
            # If already grasping, maintain the grasp
            if self.current_grasped_object is not None:
                return
            
            # Try to grasp the closest object
            closest_obj, distance = self.get_closest_object(data)
            
            if closest_obj is not None:
                self.activate_suction(data, closest_obj)
            else:
                if distance < self.distance_threshold * 2:
                    print(f"Object nearby (dist={distance:.4f}m) but out of range")
        else:
            # Deactivate suction if currently active
            if self.current_grasped_object is not None:
                self.deactivate_suction(data)
    
    def get_status(self):
        """
        Get current suction status.
        
        Returns:
            dict: Status information including grasped object
        """
        return {
            'active': self.current_grasped_object is not None,
            'grasped_object': self.current_grasped_object,
            'num_objects': len(self.object_mapping)
        }


# Example usage and test function
def test_suction_controller():
    """Test the suction controller with a simple simulation."""
    import mujoco.viewer
    
    # Load model
    model = mujoco.MjModel.from_xml_path("TEST.xml")
    data = mujoco.MjData(model)
    
    # Initialize controller
    controller = SuctionWeldController(model, data)
    
    # Simulation parameters
    suction_active = False
    
    def key_callback(keycode):
        nonlocal suction_active
        if keycode == 32:  # Spacebar
            suction_active = not suction_active
            print(f"\n{'ACTIVATING' if suction_active else 'DEACTIVATING'} suction")
    
    print("\n" + "="*60)
    print("Weld-Based Suction Test")
    print("="*60)
    print("Controls:")
    print("  SPACEBAR - Toggle suction on/off")
    print("  Move robot close to objects and press SPACEBAR to grasp")
    print("="*60 + "\n")
    
    # Run simulation with viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            # Update controller
            controller.update(data, suction_active)
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Update viewer
            viewer.sync()


if __name__ == "__main__":
    test_suction_controller()
