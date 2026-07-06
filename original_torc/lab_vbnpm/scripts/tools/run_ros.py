import os
import signal
import subprocess
import time
import yaml
from typing import List, Dict, Any, Optional
import argparse
from dataclasses import dataclass
import re
import sys
import shlex

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
)

USER_PREFIX = ""
if "USER" in os.environ:
    USER_PREFIX = os.environ["USER"] + "_"


class TMuxWindow:
    """Represents tmux window. Windows have a name, a command, and some delay"""

    def __init__(
        self,
        name: str,
        commands: List[str],
        pre_delay: float = 0.0,
        post_delay: float = 0.0,
    ):
        """Constructs a TMuxWindow.

        Args:
            name (str): Name of the window.
            commands (List[str]): List of commands to run in the window.
            pre_delay (float, optional): Delay before running the command. Defaults to 0.0.
            post_delay (float, optional): Delay after running the command. Defaults to 0.0.
        """
        self.name = name
        self.commands = commands
        self.pre_delay = pre_delay
        self.post_delay = post_delay

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TMuxWindow":
        """
        Creates a TMuxWindow instance from a dictionary,
        supporting the flexible format where the first key-value pair
        is the name and command, and 'pre_delay'/'post_delay' are optional keys.
        """
        if not isinstance(data, dict) or not data:
            raise ValueError("Window data must be a non-empty dictionary.")

        # Extract pre_delay and post_delay first, as they're known, separate keys
        pre_delay = float(data.get("pre_delay", 0.0))
        post_delay = float(data.get("post_delay", 0.0))

        # The first key-value pair will be the name and command

        try:
            name = next(iter(data))
            commands = data[name]
        except Exception as e:
            raise ValueError(
                f"Each window entry must have a name (key) and a command (value)."
            )

        if not isinstance(commands, str) and not isinstance(commands, list):
            raise ValueError(
                f"Commands for window '{name}' must be a string or a list of strings.."
            )

        if isinstance(commands, str):
            commands = [commands]

        return cls(
            name=name, commands=commands, pre_delay=pre_delay, post_delay=post_delay
        )


@dataclass
class RunROSVar:
    type: str
    help: str
    default: any = None
    choices: Optional[List[str]] = None


class RunROS:
    """Runs a ROS system using tmux."""

    def __init__(
        self,
        session_name: str,
        windows: List[TMuxWindow],
        base_variables: Dict[str, RunROSVar] = {},
        pre_cmd: str = "",
        post_cmd: str = "",
        post_attach: bool = True,
    ):
        """Constructs a RunROS.

        Args:
            session_name (str): Name of the tmux session.
            windows (List[TMuxWindow]): List of tmux windows to spawn.
            base_variables (Dict[str, RunROSVar]): Base variable values to use if a variable isn't defined in run().
            pre_cmd (str): Command to run before starting the tmux session.
            post_cmd (str): Command to run after the tmux session is killed.
            post_attach (bool): Whether to attach to the tmux session at the end or not.
        """
        self.session_name = session_name
        self.windows = windows
        self.base_variables = base_variables
        self.pre_cmd = pre_cmd
        self.post_cmd = post_cmd
        self.post_attach = post_attach

    def get_shell_type(self) -> str:
        """Returns the type of shell we're using.

        Returns:
            str: Type of shell.
        """
        # Check the SHELL environment variable.
        # Note that $SHELL often points to the "default" login shell, not necessarily
        # the "current" interactive shell if you've switched shells.
        shell_path = os.environ.get("SHELL")
        if shell_path:
            if "bash" in shell_path:
                return "bash"
            elif "zsh" in shell_path:
                return "zsh"
        return "unknown"

    def run(self, variables: Dict[str, str] = {}):
        """Runs the ROS system based on the configured session and windows.

        Args:
            variables (Dict[str, str]): Mapping of variables to values. Each variable mentioned
                in a window's command is replaced by it's value from this dictionary. Variables
                are case-sensitive. When used inside commands, variables must be prefixed by '$'.
        """
        # Populate empty/unset variables in variables with base_variables
        for name in self.base_variables:
            if name not in variables and self.base_variables[name].default != None:
                variables[name] = self.base_variables[name].default

        def hydrate_cmd(cmd: str) -> str:
            for var, value in variables.items():
                if value:
                    # True
                    # If else
                    cmd = re.sub(
                        rf"\${re.escape(var)}\?(?:'(.*?)'|\"(.*?)\"|([^\s]+?)):(?:'(.*?)'|\"(.*?)\"|([^\s]+?))(\s|$)",
                        r"\1\2\3\7",
                        cmd,
                    )
                    # If false
                    cmd = re.sub(
                        rf"!\${re.escape(var)}\?(?:'(.*?)'|\"(.*?)\"|([^\s]+?))(\s|$)",
                        "",
                        cmd,
                    )
                    # If true
                    cmd = re.sub(
                        rf"\${re.escape(var)}\?(?:'(.*?)'|\"(.*?)\"|([^\s]+?))(\s|$)",
                        r"\1\2\3\4",
                        cmd,
                    )
                else:
                    # False
                    # If else
                    cmd = re.sub(
                        rf"\${re.escape(var)}\?(?:'(.*?)'|\"(.*?)\"|([^\s]+?)):(?:'(.*?)'|\"(.*?)\"|([^\s]+?))(\s|$)",
                        r"\4\5\6\7",
                        cmd,
                    )
                    # If false
                    cmd = re.sub(
                        rf"!\${re.escape(var)}\?(?:'(.*?)'|\"(.*?)\"|([^\s]+?))(\s|$)",
                        r"\1\2\3\4",
                        cmd,
                    )
                    # If true
                    cmd = re.sub(
                        rf"\${re.escape(var)}\?(?:'(.*?)'|\"(.*?)\"|([^\s]+?))(\s|$)",
                        "",
                        cmd,
                    )
                cmd = re.sub(
                    rf"\${re.escape(var)}(?=[^\w_]|$)",
                    str(value),
                    cmd,
                )
            return cmd

        def load_process_pid(process_name: str) -> int:
            pid_file_path = f"/tmp/{USER_PREFIX}{process_name}.pid"
            try:
                with open(pid_file_path, "r") as f:
                    return int(f.read())
            except (IOError, ValueError):
                return -1

        def set_process_pid(process_name: str, value: int):
            pid_file_path = f"/tmp/{USER_PREFIX}{process_name}.pid"
            with open(pid_file_path, "w") as f:
                f.write(str(value))

        def try_kill_process(process_name: str):
            pid = load_process_pid(process_name)
            if pid > -1:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            set_process_pid(process_name, -1)

        try_kill_process("session_monitor")

        # Ensure that required variables are set

        if self.pre_cmd:
            print("Running pre-command:")
            subprocess.run(hydrate_cmd(self.pre_cmd), shell=True)
            print("Pre-command finished.")

        # Check if a tmux server is already running
        # `tmux has-session` exits with 0 if a session exists, 1 otherwise
        try:
            subprocess.run(
                ["tmux", "has-session", "-t", self.session_name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"Existing tmux session detected. Killing existing session...")
            subprocess.run(
                ["tmux", "kill-session", "-t", self.session_name], check=True
            )
            time.sleep(0.5)
            print("Existing tmux session killed successfully.")
        except subprocess.CalledProcessError:
            print("No tmux session detected. Proceeding with new session creation.")
        except Exception as e:
            print(f"An error occurred while checking/killing tmux session: {e}")

        window_index = 0

        # Initialize the tmux session with the first window
        if not self.windows:
            print("No windows defined in the configuration. Exiting.")
            return

        # Create remaining windows
        for i, window in enumerate(self.windows):
            if window.pre_delay > 0:
                print(
                    f"Waiting {window.pre_delay} seconds before launching '{window.name}'..."
                )
                time.sleep(window.pre_delay)

            if i == 0:
                # Use new-session -e to set PYTHONPATH for the initial session/window
                # This ensures the server is running *before* the environment is set
                new_session_cmd = [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    self.session_name,
                    "-n",
                    window.name,
                ]
                subprocess.run(new_session_cmd)

                if self.post_cmd:
                    session_monitor = subprocess.Popen(
                        [
                            "bash",
                            "-c",
                            f"""while tmux has-session -t {self.session_name}; do
sleep 1
done
{self.post_cmd}""",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    set_process_pid("session_monitor", session_monitor.pid)

            else:
                subprocess.run(
                    [
                        "tmux",
                        "new-window",
                        "-t",
                        f"{self.session_name}:{window_index}",
                        "-n",
                        window.name,
                    ]
                )

            shell_type = self.get_shell_type()
            commands = []
            conda_prefix = os.environ.get("TORC_CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                commands.append(
                    "source /home/ziyaochen/miniconda3/etc/profile.d/conda.sh "
                    f"&& conda activate '{conda_prefix}'"
                )
                parent_pythonpath = os.environ.get("PYTHONPATH")
                if parent_pythonpath:
                    commands.append(
                        f"export PYTHONPATH={shlex.quote(parent_pythonpath)}:$PYTHONPATH"
                    )
                preserved_env_vars = [
                    "GC6D_ROOT",
                    "CUDA_VISIBLE_DEVICES",
                    "MUJOCO_GL",
                    "PYOPENGL_PLATFORM",
                    "TORC_ROBOT",
                    "TORC_ROBOT_TYPE",
                    "TORC_SCENE_PATH",
                    "TORC_SCENE_NAME",
                    "TORC_GRASP_PLANNER",
                    "TORC_USE_CGN_ZMQ",
                    "TORC_CAPTURE_SELECTED_GRASP",
                    "TORC_CAPTURE_ORIGINAL_PICK1_ALLPTS",
                    "TORC_CGN_ZMQ_ADDRESS",
                    "TORC_CGN_ZMQ_TIMEOUT_MS",
                    "TORC_RENDER_EXECUTION_VIDEO",
                    "TORC_RENDER_CAMERAS",
                    "TORC_RENDER_STRIDE",
                    "TORC_RENDER_FPS",
                    "TORC_RENDER_WIDTH",
                    "TORC_RENDER_HEIGHT",
                    "TORC_RENDER_EXECUTION_FRAMES",
                    "TORC_RENDER_JPEG_QUALITY",
                    "TORC_CUROBO_SRC",
                    "TORC_CONDA_PREFIX",
                    "CUDA_HOME",
                    "CUDA_PATH",
                    "CPATH",
                    "C_INCLUDE_PATH",
                    "CPLUS_INCLUDE_PATH",
                    "LIBRARY_PATH",
                    "LD_LIBRARY_PATH",
                ]
                for env_var in preserved_env_vars:
                    env_value = os.environ.get(env_var)
                    if env_value is not None:
                        commands.append(
                            f"export {env_var}={shlex.quote(env_value)}"
                        )
            curobo_src = os.environ.get("TORC_CUROBO_SRC")
            if curobo_src:
                commands.append(
                    f"export PYTHONPATH={shlex.quote(curobo_src)}:$PYTHONPATH"
                )
            ros_setup = os.environ.get("TORC_ROS_SETUP")
            if not ros_setup:
                candidate = os.path.abspath(os.path.join(ROOT_DIR, "../../devel", f"setup.{shell_type}"))
                if os.path.exists(candidate):
                    ros_setup = candidate
                else:
                    ros_setup = f"/home/ziyaochen/gc6d_lift3d_traj/ros_workspace/devel/setup.{shell_type}"
            commands += [
                f"cd '{ROOT_DIR}' && source '{ros_setup}' && export ROS_PACKAGE_PATH='{os.path.dirname(ROOT_DIR)}':$ROS_PACKAGE_PATH && export PYTHONPATH='{ROOT_DIR}/scripts:{ROOT_DIR}':$PYTHONPATH"
            ] + window.commands
            for processed_command in commands:
                subprocess.run(
                    [
                        "tmux",
                        "send-keys",
                        "-t",
                        f"{self.session_name}:{window_index}",
                        hydrate_cmd(processed_command),
                        "C-m",
                    ]
                )

            if window.post_delay > 0:
                print(
                    f"Waiting {window.post_delay} seconds after launching '{window.name}'..."
                )
                time.sleep(window.post_delay)

            window_index += 1

        # Attach the current terminal to the newly created session
        if self.post_attach:
            subprocess.run(["tmux", "attach-session", "-t", self.session_name])

    @staticmethod
    def from_yaml(filepath: str) -> "RunROS":
        """Uses a RunROS yaml config file to instantiate a RunROS class."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        def load_yaml_config(filepath: str):
            filepath_dir = os.path.dirname(filepath)
            with open(filepath, "r") as f:
                config = yaml.safe_load(f)
            base_config_paths = config.get("base_config", "")
            if not (
                isinstance(base_config_paths, str)
                or isinstance(base_config_paths, list)
            ):
                raise ValueError(
                    "Expected base_config to either be a path the base config, or a list of multiple base config paths."
                )

            if isinstance(base_config_paths, str):
                base_config_paths = [base_config_paths]

            def update_config(config: dict, other_config: dict):
                # dict fields to merge in separately
                dict_fields = ["vars", "defaults"]
                for field in other_config:
                    if field in dict_fields and field in config:
                        config[field].update(other_config[field])
                    else:
                        config[field] = other_config[field]

            final_config = {}
            for path in base_config_paths:
                if not path:
                    continue
                if not os.path.isabs(path):
                    path = os.path.join(filepath_dir, path)
                base_config = load_yaml_config(path)
                update_config(final_config, base_config)
            update_config(final_config, config)

            return final_config

        config = load_yaml_config(filepath)
        session_name = config.get("session_name", "ros_system")
        _vars = config.get("vars", {})
        defaults = config.get("defaults", {})
        pre_cmd = config.get("pre_cmd", "")
        post_cmd = config.get("post_cmd", "")
        post_attach = config.get("post_attach", True)

        if not isinstance(session_name, str):
            raise ValueError("Expected session_name to be a string.")
        if not isinstance(_vars, dict):
            raise ValueError("Expected vars to be a dict.")
        if not isinstance(defaults, dict):
            raise ValueError("Expected defaults to be a dict.")
        if not isinstance(post_attach, bool):
            raise ValueError("Expected post_attach to be a bool.")
        if not isinstance(pre_cmd, str):
            raise ValueError("Expected pre_cmd to be a string.")
        if not isinstance(post_cmd, str):
            raise ValueError("Expected post_cmd to be a string.")

        for name in _vars:
            _type = _vars[name].get("type", None)
            if _type not in ["str", "int", "bool", "float", "file", "enum"]:
                raise ValueError(
                    f"Expected var '{name}' to have a valid type (str, int, bool, float, file, enum) but got type: '{_type}'."
                )
            _py_type = _type
            if _type in ["file", "enum"]:
                _py_type = "str"
            default = _vars[name].get("default", None)
            if default is not None and type(default).__name__ != _py_type:
                raise ValueError(
                    f"Expected the default value of '{name}' to have the same type."
                )
            _help = _vars[name].get("help", None)
            if _help is None:
                raise ValueError(f"Expected var '{name}' to have a help string.")

        for name in defaults:
            if name not in _vars:
                raise ValueError(
                    f"Expected default var '{name}' to be defined in vars."
                )
            default_value = defaults[name]
            if type(default_value).__name__ != _vars[name]["type"]:
                raise ValueError(
                    f"Expected the default value of '{name}' to have the type of '{_vars[name]['type']}'."
                )
            _vars[name]["default"] = default_value

        windows_data = config.get("windows", [])
        assert isinstance(windows_data, list), "Expected windows to be a list."
        tmux_windows = []
        for window_entry in windows_data:
            # Each entry in 'windows' can now be a simple key-value or a dictionary
            # The TMuxWindow.from_dict will handle the parsing
            try:
                tmux_windows.append(TMuxWindow.from_dict(window_entry))
            except ValueError as e:
                print(
                    f"Warning: Skipping invalid window entry: {window_entry}. Error: {e}"
                )

        base_variables = {v: RunROSVar(**_vars[v]) for v in _vars}
        return RunROS(
            session_name=session_name,
            windows=tmux_windows,
            base_variables=base_variables,
            pre_cmd=pre_cmd,
            post_cmd=post_cmd,
            post_attach=post_attach,
        )


class RunROSCLI:
    def run(self):
        """Runs the CLI."""
        parser = argparse.ArgumentParser(
            prog="run_ros.sh",
            description="Runs a ROS system based on a yaml config file.",
            add_help=False,
        )
        parser.add_argument(
            "ros_config", type=str, default="", help="Path to a ros config yaml file."
        )
        if len(sys.argv) == 1:
            parser.print_help()
            return
        args, _ = parser.parse_known_args()

        try:
            run_ros = RunROS.from_yaml(args.ros_config)
        except (FileNotFoundError, ValueError) as e:
            print(f"🛑 Error loading config yaml: {e}")
            return

        parser.add_argument(
            "-h", "--help", action="help", help="Show this help message and exit."
        )

        def to_kebab_case(s: str) -> str:
            if not s.isupper():
                # Handle PascalCase or camelCase
                s = s.strip()
                # Handle consecutive uppercase letters
                s = re.sub(r"([A-Z]+)([A-Z])", r"\1-\2", s)
                # Handle camelCase
                s = re.sub(r"([a-z\d])([A-Z])", r"\1-\2", s)
            # Replace spaces and underscores with hyphens
            s = re.sub(r"[_\s]+", r"-", s)
            return s.lower()

        def to_caps_case(s: str) -> str:
            return to_kebab_case(s).replace("-", "_").upper()

        for name, base_var in run_ros.base_variables.items():
            _type = ""
            if base_var.type in ["file", "enum"]:
                _type = str
            else:
                _type = eval(base_var.type)

            if base_var.type == "bool":
                if base_var.default is None or base_var.default == True:
                    action = "store_true"
                else:
                    action = "store_false"
                parser.add_argument(
                    f"--{to_kebab_case(name)}",
                    action=action,
                    help=base_var.help,
                )
            elif base_var.type == "enum":
                parser.add_argument(
                    f"--{to_kebab_case(name)}",
                    type=_type,
                    default=base_var.default,
                    choices=base_var.choices,
                    required=base_var.default is None,
                    help=base_var.help,
                )
            else:
                parser.add_argument(
                    f"--{to_kebab_case(name)}",
                    type=_type,
                    default=base_var.default,
                    required=base_var.default is None,
                    help=base_var.help,
                )
        args = vars(parser.parse_args())
        del args["ros_config"]
        variables = {}
        longest_var_len = 0
        for arg in args:
            var_name = to_caps_case(arg)
            if len(var_name) > longest_var_len:
                longest_var_len = len(var_name)
            variables[var_name] = args[arg]
            if run_ros.base_variables[var_name].type == "file" and args[arg] not in [
                "",
                "''",
                '""',
            ]:
                abs_path = os.path.abspath(os.path.join(ROOT_DIR, args[arg]))
                if not os.path.exists(abs_path):
                    print(
                        f"🛑 Error for arg '{arg}': File at '{abs_path}' does not exist!"
                    )
                    return

        print("🤖 Running ROS system")
        for name, value in variables.items():
            print(f"    {name}:".ljust(longest_var_len + 5), value)

        run_ros.run(variables=variables)


if __name__ == "__main__":
    RunROSCLI().run()
