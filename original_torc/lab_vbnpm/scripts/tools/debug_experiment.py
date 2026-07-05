import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Paths (yours) ---
parent_dir = "./experiments/runs"
# trial_dir = "trial_2025-09-14_13-14-01__unstructured"
# trial_dir = "trial_2025-09-14_15-42-54__unstructured__vlm_dg"
trial_dir = "trial_2025-09-22_18-19-00__unstructured__vlm_dg"
# experiment_dir = "30_obj_000066_0"
# experiment_dir = "30__obj_000047_0__vlm_dg"
# experiment_dir = "30__obj_000066_0__vlm_dg"
# experiment_dir = "10__obj_000043_0__vlm_dg"
# experiment_dir = "14__obj_000065_0__vlm_dg"
# experiment_dir = "110__obj_000044_0__vlm_dg"

# experiment_dir = "122__obj_000047_1__vlm_dg"
# experiment_dir = "128__obj_000040_0__vlm_dg"
# experiment_dir = "14__obj_000065_0__vlm_dg"
experiment_dir = "30__obj_000066_0__vlm_dg"

trial_dir_global = os.path.join(parent_dir, trial_dir)

trial_dir_global_list = os.listdir(trial_dir_global)

result_dirs = [result_dir for result_dir in trial_dir_global_list if os.path.isdir(os.path.join(trial_dir_global, result_dir))]

result_dirs_global = [os.path.join(trial_dir_global, result_dir) for result_dir in result_dirs]

result_dir = os.path.join(parent_dir, trial_dir, experiment_dir)

# --- Helpers ---
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def is_depgraph(obj) -> bool:
    # Duck-typing for your DepGraph
    return hasattr(obj, "draw") and hasattr(obj, "nx_graph")

def render_depgraph_to_rgba(depgraph_obj, dpi=120):
    """
    Render a DepGraph into an offscreen Matplotlib figure and return an RGBA uint8 array.
    This avoids recreating InteractiveGraph on every navigation.
    """
    # Small figure that still looks crisp; adjust as needed
    fig = plt.Figure(figsize=(6, 4), dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    try:
        # Use your renderer; draw into this axis without blocking
        depgraph_obj.draw(to_show=False, block=False, axis=ax)
    except TypeError:
        depgraph_obj.draw(to_show=False, axis=ax)

    fig.tight_layout()
    canvas.draw()  # render to the Agg buffer
    buf = np.asarray(canvas.buffer_rgba())
    # Copy to detach from canvas memory
    return buf.copy()

# --- Collect files ---
pngs = sorted([f for f in os.listdir(result_dir) if f.endswith(".png") and f.startswith("img_labeled")], key=lambda f: int(f[:-4].rsplit("_", 1)[-1]))
depgraphs = sorted([f for f in os.listdir(result_dir) if f.endswith(".depgraph")], key=lambda f: int(f[:-9].rsplit("_", 1)[-1]))

if not pngs:
    raise RuntimeError(f"No PNGs found in {result_dir}")

# Group expected: 1 VLM + 1 non-VLM depgraph per png (by index substring j = i+1)
groups = []
for i, png in enumerate(pngs):
    j = i + 1
    related = [d for d in depgraphs if str(j) in d]
    vlm = [d for d in related if "vlm" in d.lower()]
    nonvlm = [d for d in related if "vlm" not in d.lower()]

    if len(vlm) != 1 or len(nonvlm) != 1:
        print(f"Warning: expected 1 VLM + 1 Non-VLM for {png}; "
              f"found: vlm={len(vlm)}, nonvlm={len(nonvlm)} (related={len(related)})")

    groups.append({
        "png": png,
        "vlm": vlm[0] if vlm else None,
        "nonvlm": nonvlm[0] if nonvlm else None,
    })

# --- Caches to speed up navigation ---
img_cache = {}          # key: png filename -> np.ndarray (image)
pickle_cache = {}       # key: depgraph filename -> loaded object
render_cache = {}       # key: depgraph filename -> np.ndarray (RGBA render)

def get_image(png_name):
    if png_name not in img_cache:
        img_cache[png_name] = plt.imread(os.path.join(result_dir, png_name))
    return img_cache[png_name]

def get_depgraph_obj(dep_name):
    if dep_name is None:
        return None
    if dep_name not in pickle_cache:
        obj = load_pickle(os.path.join(result_dir, dep_name))
        pickle_cache[dep_name] = obj
    return pickle_cache[dep_name]

def get_depgraph_rgba(dep_name):
    if dep_name is None:
        return None
    if dep_name not in render_cache:
        obj = get_depgraph_obj(dep_name)
        if (obj is None) or (not is_depgraph(obj)):
            render_cache[dep_name] = None
        else:
            # Optional pre-processing before rendering:
            # try:
            #     obj.normalize_edges()
            #     obj.add_hidden_edges()
            # except Exception as e:
            #     print(f"[warn] normalize/add_hidden failed for {dep_name}: {e}")
            render_cache[dep_name] = render_depgraph_to_rgba(obj)
    return render_cache[dep_name]

# --- Viz ---
fig, axs = plt.subplots(1, 3, figsize=(16, 6))
plt.tight_layout()
index = 0

def show_group(i: int):
    g = groups[i]
    fig.suptitle(g["png"], fontsize=14)

    # 1) Image
    axs[0].clear()
    img = get_image(g["png"])
    axs[0].imshow(img)
    axs[0].set_title("Image")
    axs[0].axis("off")

    # 2) Non-VLM DepGraph (rendered RGBA)
    axs[1].clear()
    nonvlm_rgba = get_depgraph_rgba(g["nonvlm"])
    axs[1].set_title(f"Non-VLM DepGraph\n({g['nonvlm'] or 'missing'})")
    if nonvlm_rgba is None:
        # Try to hint at what went wrong
        obj = get_depgraph_obj(g["nonvlm"]) if g["nonvlm"] else None
        if obj is None:
            axs[1].text(0.5, 0.5, "Missing depgraph", ha="center", va="center")
        elif not is_depgraph(obj):
            axs[1].text(0.5, 0.5, f"Not a DepGraph:\n{type(obj)}", ha="center", va="center", wrap=True)
        else:
            axs[1].text(0.5, 0.5, f"Render failed", ha="center", va="center")
        axs[1].set_axis_off()
    else:
        axs[1].imshow(nonvlm_rgba)
        axs[1].axis("off")

    # 3) VLM DepGraph (rendered RGBA)
    axs[2].clear()
    vlm_rgba = get_depgraph_rgba(g["vlm"])
    axs[2].set_title(f"VLM DepGraph\n({g['vlm'] or 'missing'})")
    if vlm_rgba is None:
        obj = get_depgraph_obj(g["vlm"]) if g["vlm"] else None
        if obj is None:
            axs[2].text(0.5, 0.5, "Missing depgraph", ha="center", va="center")
        elif not is_depgraph(obj):
            axs[2].text(0.5, 0.5, f"Not a DepGraph:\n{type(obj)}", ha="center", va="center", wrap=True)
        else:
            axs[2].text(0.5, 0.5, f"Render failed", ha="center", va="center")
        axs[2].set_axis_off()
    else:
        axs[2].imshow(vlm_rgba)
        axs[2].axis("off")

    plt.tight_layout()
    plt.draw()

show_group(index)
plt.show(block=False)

def on_key(event):
    global index
    if event.key == "right":
        index = (index + 1) % len(groups)
    elif event.key == "left":
        index = (index - 1) % len(groups)
    elif event.key == "escape":
        plt.close(event.canvas.figure)
        return
    show_group(index)

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
