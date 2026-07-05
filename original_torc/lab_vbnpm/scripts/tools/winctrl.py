import argparse
import yaml
import subprocess
import time
import re
from typing import Dict, List, TypedDict, Optional
from dataclasses import dataclass
import prettyprinter as pp

pp.install_extras(exclude=["ipython", "django"])


@dataclass
class Window:
    """Represents a Linux window from wmctrl."""

    id: str
    desktop: int
    x: int
    y: int
    width: int
    height: int
    machine: str
    title: str


def get_windows() -> List[Window]:
    """Gets a list of all window IDs and titles using wmctrl."""
    try:
        output = subprocess.check_output(["wmctrl", "-l", "-G"]).decode("utf-8")
        windows = []
        for line in output.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 8:
                title = " ".join(parts[7:])
                windows.append(
                    Window(
                        id=parts[0],
                        desktop=int(parts[1]),
                        x=int(parts[2]),
                        y=int(parts[3]),
                        width=int(parts[4]),
                        height=int(parts[5]),
                        machine=parts[6],
                        title=title,
                    )
                )
        return windows
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "Error: wmctrl not found or failed to execute. Please ensure it's installed."
        )
        return []


def get_window(id: str) -> Optional[Window]:
    """Gets a window by a specific window id."""
    windows = get_windows()
    for window in windows:
        if window.id == id:
            return window
    return None


@dataclass
class Monitor:
    """Represents a monitor's information from xrandr."""

    id: int
    width: float
    height: float
    x: float
    y: float


def get_monitors() -> List[Monitor]:
    """Gets list of monitors using xrandr."""
    try:
        output = subprocess.check_output(["xrandr"]).decode("utf-8")
        monitors = []
        monitor_id = 0

        # Regex to find monitor lines (e.g., '1920x1080+0+0')
        pattern = re.compile(r"(\d+)x(\d+)\+(\d+)\+(\d+)")

        for line in output.split("\n"):
            match = pattern.search(line)
            if " connected" in line and match:
                width, height, x, y = map(int, match.groups())
                monitors.append(
                    Monitor(id=monitor_id, width=width, height=height, x=x, y=y)
                )
                monitor_id += 1
        return monitors
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "Error: xrandr not found or failed to execute. Cannot get monitor information."
        )
        return []


def title_to_window_dict() -> Dict[str, Monitor]:
    """Returns a dictionary of window names to Windows."""
    windows = get_windows()
    return {w.title: w for w in windows}


def id_to_monitor_dict() -> Dict[int, Monitor]:
    """Returns a dictionary of monitor id to Monitors."""
    monitors = get_monitors()
    return {m.id: m for m in monitors}


def parse_value(value: str, dimension: int) -> int:
    """Parses a string value, converting percentages to pixels."""
    if value.endswith("%"):
        percent = float(value.strip("%")) / 100
        return int(percent * dimension)
    return int(value)


MONITOR_DICT: Dict[int, Monitor] = {}


@dataclass
class ConfigWindow:
    """Window with a desired position and size."""

    title: str
    x: int
    y: int
    width: int
    height: int
    active: bool = False

    @classmethod
    def from_config(cls, config: dict):
        """Creates a window from a config dict.

        The config dict has the following type:
        {
            "title": str
            "position": Optional[str]
            "size": Optional[str]
            "monitor": Optional[int]
        }
        """
        pos = config.get("position", "0,0")
        size = config.get("size", "100,100")
        monitor_id = config.get("monitor", 0)

        monitor = MONITOR_DICT.get(monitor_id)
        if not monitor:
            print(
                f"Warning: Monitor {monitor_id} not found. Skipping '{config.get('name', 'window')}'."
            )
            return

        pos_x_str, pos_y_str = pos.split(",")
        size_width_str, size_height_str = size.split(",")

        res = cls(
            title=config.get("title"),
            x=parse_value(pos_x_str, monitor.width) + monitor.x,
            y=parse_value(pos_y_str, monitor.height) + monitor.y,
            width=parse_value(size_width_str, monitor.width),
            height=parse_value(size_height_str, monitor.height),
        )
        return res


def move_and_resize_window(window_id: str, config: ConfigWindow) -> None:
    """Moves and resizes a window based on the configuration, handling percentages."""

    def wmctrl_set_window(
        window_id: int, gravity: int, x: int, y: int, width: int, height: int
    ) -> str:
        """
        Sets a window's position and size using wmctrl.
        Returns the command used to edit the window.
        """
        geometry_string = f"{gravity},{x},{y},{width},{height}"
        subprocess.run(
            [
                "wmctrl",
                "-i",
                "-r",
                window_id,
                "-b",
                "remove,maximized_vert,maximized_horz",
            ]
        )
        wmctrl_command = ["wmctrl", "-i", "-r", window_id, "-e", geometry_string]
        subprocess.run(wmctrl_command, check=True)
        wmctrl_active_command = ["wmctrl", "-i", "-a", window_id]
        subprocess.run(wmctrl_active_command, check=True)
        return " ".join(wmctrl_command)

    print(f"Move resize window: '{config.title}'")
    try:
        x = config.x
        y = config.y
        width = config.width
        height = config.height
        gravity = 0
        cmd = wmctrl_set_window(window_id, gravity, x, y, width, height)

        print(f"    Moved to ({x},{y}) with size {width}x{height}")
        print(f"    '- {cmd}")

    except (ValueError, subprocess.CalledProcessError) as e:
        print(f"    Warning: Failed to move/resize '{config.title}'. Error: {e}")


def main() -> None:
    """Main CLI function."""
    global MONITOR_DICT

    parser = argparse.ArgumentParser(
        description="A window manager that positions windows based on a YAML config file."
    )
    parser.add_argument("config_file", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
            if "windows" not in config:
                print("Error: YAML file must contain a 'windows' key.")
                return
    except FileNotFoundError:
        print(f"Error: The file '{args.config_file}' was not found.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    MONITOR_DICT = id_to_monitor_dict()
    if not MONITOR_DICT:
        print("Could not retrieve monitor information. Exiting.")
        return

    config_windows: List[ConfigWindow] = [
        ConfigWindow.from_config({"title": title, **win})
        for title, win in config["windows"].items()
    ]
    print("Config Windows:")
    pp.pprint(config_windows)

    print("Window manager started. Monitoring for windows...")
    while True:
        curr_windows_dict = title_to_window_dict()

        for config_window in config_windows:
            window = curr_windows_dict.get(config_window.title)

            if window and not config_window.active:
                move_and_resize_window(window.id, config_window)
                config_window.active = True
            elif not window and config_window.active:
                print(
                    f"Window '{config_window.title}' closed. Will watch for it again."
                )
                config_window.active = False

        time.sleep(0.5)


if __name__ == "__main__":
    main()
