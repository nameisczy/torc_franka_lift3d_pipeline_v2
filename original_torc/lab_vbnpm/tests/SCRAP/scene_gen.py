import glob
import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa
from scene_synthesizer import assets as aa

s = pa.TableAsset(1., 2, 0.7, 0.04, 0.02).scene('table')

s.label_support("table_surface", min_area=0.3)

files = glob.glob('../../models/objects/GraspClutter6D/models_obj_m/obj_*/obj_*.xml')
ag = aa.asset_generator(files)

num_assets = 10
i = 0
for asset in ag:
    if i >= num_assets:
        break
    i+=1
    s.place_object(
        obj_id=f"obj_{i}",
        obj_asset=asset,
        support_id="table_surface",
        obj_orientation_iterator=synth.utils.orientation_generator_stable_poses(asset),
    )

# s.colorize()
# s.show()
s.export("test.urdf")
