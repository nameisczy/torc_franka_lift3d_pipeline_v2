#!/bin/bash

# --- Configuration ---
DEFAULT_PROJECT_NAME="grasp_planning"
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_COMPOSE_FILE=${DOCKER_COMPOSE_FILE:-"$DOCKER_DIR/docker-compose.yaml"}

# --- Helper Functions ---
# ... get_next_project_name()
get_next_project_name() {
    local base_name=$1
    local counter=1
    local new_name
    local is_running
    
    while true; do
        new_name="${base_name}-${counter}"
        
        is_running=$(docker compose ls -q --filter name="$new_name")
        
        if [ -z "$is_running" ]; then
            echo "$new_name"
            return 0
        fi
        
        counter=$((counter + 1))
        if [ "$counter" -gt 100 ]; then
            echo "ERROR: Could not find an available project name after 100 attempts." >&2
            return 1
        fi
    done
}

# ... set_container_names()
set_container_names() {
    PROJECT_NAME="$1"
    CONTAINER_VBNPM="${PROJECT_NAME}-vbnpm_ros-1"
    CONTAINER_CGN="${PROJECT_NAME}-cgn_ros-1"
}

# --- Core Commands ---

# Function to start the Docker Compose project
# Takes PROJECT_NAME as $1 and remaining ARGS as $2 (quoted)
start_project() {
    local PROJECT_NAME="$1"
    local RUN_ARGS="$2"

    set_container_names "$PROJECT_NAME"

    echo "--- Checking for existing project: '$PROJECT_NAME' ---"
    
    IS_RUNNING=$(docker compose ls -q --filter name="$PROJECT_NAME")
    if [ -n "$IS_RUNNING" ]; then
        echo "Docker Compose project '$PROJECT_NAME' is running. Shutting it down..."
        docker compose -p "$PROJECT_NAME" -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
        if [ $? -ne 0 ]; then echo "ERROR: Failed to shut down project '$PROJECT_NAME'."; exit 1; fi
        echo "Project '$PROJECT_NAME' successfully shut down."
    fi

    echo "--- Starting $PROJECT_NAME ---"
    docker compose -p "$PROJECT_NAME" -f "$DOCKER_COMPOSE_FILE" up -d

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to start project '$PROJECT_NAME'."
        exit 1
    fi
    
    # NEW: Status message for forwarded arguments
    if [ -n "$RUN_ARGS" ]; then
        echo "--- Forwarding run arguments: '$RUN_ARGS' ---"
    else
        echo "--- No run arguments provided. Starting ROS Master only. ---"
    fi

    echo -e "\n--- Initializing $CONTAINER_VBNPM (ROS MASTER) ---"
    
    # Init VBNPM (ROS Master)
    docker exec --user user -i "$CONTAINER_VBNPM" /bin/bash <<EOF
VBNPM_TRY=\$(host vbnpm_ros > /dev/null && host vbnpm_ros || host localhost)
VBNPM_IP=\$(echo \$VBNPM_TRY | awk '/has address/ {print \$4}')
CGN_TRY=\$(host cgn_ros > /dev/null && host cgn_ros|| host localhost)
CGN_IP=\$(echo \$CGN_TRY | awk '/has address/ {print \$4}')

ROS_IP=\$VBNPM_IP
ROS_MASTER_URI=http://\$VBNPM_IP:11311

# Overwrite/Create .bashrc entries for ROS environment variables
sed -i '/ROS_IP=/d' ~/.bashrc
sed -i '/ROS_MASTER_URI=/d' ~/.bashrc
echo "export ROS_IP=\$ROS_IP" >> ~/.bashrc
echo "export ROS_MASTER_URI=\$ROS_MASTER_URI" >> ~/.bashrc

echo "vbnpm_ros setup (ROS MASTER):"
echo "ROS_IP: \$ROS_IP"
echo "ROS_MASTER_URI: \$ROS_MASTER_URI"

if command -v tmux &> /dev/null; then
    tmux new-session -d -s docker-runner
    tmux send-keys -t docker-runner "source ~/.bashrc" C-m
    # --- The forwarded arguments are placed here ---
    tmux send-keys -t docker-runner "./run_experiment.sh $RUN_ARGS --docker --base-dir $PROJECT_NAME" C-m
else
    echo "Warning: tmux not found. Could not start roscore automatically."
fi
EOF

    # ... (Init CGN remains unchanged) ...
    echo -e "\n--- Initializing $CONTAINER_CGN (ROS NODE) ---"
    
    docker exec --user user -i "$CONTAINER_CGN" /bin/bash <<EOF
VBNPM_TRY=\$(host vbnpm_ros > /dev/null && host vbnpm_ros || host localhost)
VBNPM_IP=\$(echo \$VBNPM_TRY | awk '/has address/ {print \$4}')
CGN_TRY=\$(host cgn_ros > /dev/null && host cgn_ros|| host localhost)
CGN_IP=\$(echo \$CGN_TRY | awk '/has address/ {print \$4}')

ROS_IP=\$CGN_IP
ROS_MASTER_URI=http://\$VBNPM_IP:11311

# Overwrite/Create .bashrc entries for ROS environment variables
sed -i '/ROS_IP=/d' ~/.bashrc
sed -i '/ROS_MASTER_URI=/d' ~/.bashrc
echo "export ROS_IP=\$ROS_IP" >> ~/.bashrc
echo "export ROS_MASTER_URI=\$ROS_MASTER_URI" >> ~/.bashrc

echo "cgn_ros setup:"
echo "ROS_IP: \$ROS_IP"
echo "ROS_MASTER_URI: \$ROS_MASTER_URI"

if command -v tmux &> /dev/null; then
    tmux new-session -d -s docker-runner
    tmux send-keys -t docker-runner "source ~/.bashrc" C-m
    tmux send-keys -t docker-runner "roslaunch cgn_ros grasp_server.launch" C-m
else
    echo "Warning: tmux not found. Could not start roslaunch automatically."
fi
EOF

    echo -e "\n--- Setup Complete ---"
    echo "Project:    $PROJECT_NAME"
    echo "ROS Master: $CONTAINER_VBNPM"
    echo "ROS Node:   $CONTAINER_CGN"
    echo ""
    echo "To attach to the vbnpm_ros: ./docker_run_ros.sh attach $PROJECT_NAME vbnpm_ros"
    echo "To attach to the cgn_ros:   ./docker_run_ros.sh attach $PROJECT_NAME cgn_ros"
}

# ... stop_project()
stop_project() {
    local PROJECT_NAME="$1"
    echo "--- Shutting down project '$PROJECT_NAME' ---"
    IS_RUNNING=$(docker compose ls -q --filter name="$PROJECT_NAME")
    if [ -z "$IS_RUNNING" ]; then echo "Docker Compose project '$PROJECT_NAME' is not running."; return 0; fi
    # Use -f to ensure we are stopping the correct configuration
    docker compose -p "$PROJECT_NAME" -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
    if [ $? -eq 0 ]; then echo "Project '$PROJECT_NAME' successfully shut down."; else echo "ERROR: Failed to shut down project '$PROJECT_NAME'."; exit 1; fi
}

# NEW: list_projects() - Only lists projects using the specific docker-compose.yaml
list_projects() { 
    echo "--- Projects ---"
    echo " @ compose file: $DOCKER_COMPOSE_FILE"
    # Get all running project names
    local found_projects=$(docker compose ls --format json | jq -r '.[] | "\(.Name) \(.ConfigFiles)"' | grep "$DOCKER_COMPOSE_FILE" | awk '{print $1}')
    if [ ${#found_projects[@]} -eq 0 ]; then
        echo "No running projects found using $DOCKER_COMPOSE_FILE."
    else
        echo "$found_projects"
    fi
}

# NEW: stop_all_projects() - Stops all projects from the specific docker-compose.yaml
stop_all_projects() {
    # Get the list of project names that use our config file
    local found_projects=$(docker compose ls --format json | jq -r '.[] | "\(.Name) \(.ConfigFiles)"' | grep "$DOCKER_COMPOSE_FILE" | awk '{print $1}')
    echo "--- Stop All Projects ---"
    echo -e "Found projects:\n${found_projects[*]}"
    
    local stop_count=0
    if [[ -n "$found_projects" ]]; then
      while IFS= read -r project_name; do
          echo ""
          stop_project "$project_name"
          stop_count=$((stop_count + 1))
      done <<< "$found_projects"
    fi
    
    echo ""
    echo "--- Stop All Complete: $stop_count projects successfully shut down. ---"
}

# NEW: attach_container()
attach_container() {
    local PROJECT_NAME="$1"
    local CONTAINER_SHORT_NAME="$2" # e.g., vbnpm_ros or cgn_ros

    if [ -z "$PROJECT_NAME" ] || [ -z "$CONTAINER_SHORT_NAME" ]; then
        echo "ERROR: 'attach' command requires both PROJECT_NAME and CONTAINER_NAME."
        usage
        exit 1
    fi

    # Check if the project is running
    IS_RUNNING=$(docker compose ls -q --filter name="$PROJECT_NAME")
    if [ -z "$IS_RUNNING" ]; then 
        echo "ERROR: Docker Compose project '$PROJECT_NAME' is not running."
        exit 1
    fi

    # Construct the full container name
    local FULL_CONTAINER_NAME="${PROJECT_NAME}-${CONTAINER_SHORT_NAME}-1"

    # Check if the specific container is running
    if ! docker ps -f name="$FULL_CONTAINER_NAME" --format '{{.Names}}' | grep -q "$FULL_CONTAINER_NAME"; then
        echo "ERROR: Container '$FULL_CONTAINER_NAME' is not running."
        echo "Note: Expected container names for project '$PROJECT_NAME' are:"
        echo " - vbnpm_ros: ${PROJECT_NAME}-vbnpm_ros-1"
        echo " - cgn_ros:   ${PROJECT_NAME}-cgn_ros-1"
        exit 1
    fi

    echo "--- Attaching to $FULL_CONTAINER_NAME ---"
    # Execute the tmux attach command
    docker exec -it "$FULL_CONTAINER_NAME" /bin/bash
}


# ... usage()
usage() {
    echo "Usage: $0 [start|stop|stopall|list|attach] [PROJECT_NAME] [-- ARGS...]"
    echo ""
    echo "Commands:"
    echo "  start [PROJECT_NAME] : Starts the Docker project. If no name is given,"
    echo "                         a new name like '${DEFAULT_PROJECT_NAME}-N' is generated."
    echo "                         The optional PROJECT_NAME MUST come before '--'."
    echo "                         All arguments after '--' are passed to"
    echo "                         './run_experiment.sh' inside the ROS Master container."
    echo "  stop PROJECT_NAME    : Stops and removes containers for the specified project."
    echo "  stopall              : Stops and removes all projects started with this script's config."
    echo "  list                 : Lists all running Docker Compose projects started with this script's config."
    echo "  attach PROJECT_NAME CONTAINER_NAME"
    echo "                       : Attaches to the tmux session inside a running container."
    echo "                         CONTAINER_NAME can be 'vbnpm_ros' or 'cgn_ros'."
    echo ""
    echo "Ex. Start with run args:  $0 start -- trial unstructured --method dg_only"
    echo "Ex. Start custom project: $0 start my_project -- trial unstructured"
    echo "Ex. Stop a project:       $0 stop my_project"
    echo "Ex. Stop all projects:    $0 stopall"
    echo "Ex. Attach to ROS Master: $0 attach my_project vbnpm_ros"
}

# --- Main Logic / Argument Parsing (REFACTORED) ---

# Check for at least one argument (the command)
if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

COMMAND="$1"
# Shift off the command so $1 is now the project name or the first arg
shift

case "$COMMAND" in
    start)
        PROJECT_NAME=""
        FORWARD_ARGS=""
        
        # Look for a project name argument before the '--' delimiter
        if [ -n "$1" ] && [ "$1" != "--" ]; then
            PROJECT_NAME="$1"
            shift # Consume the project name
        fi
        
        # If the next argument is '--', shift it off
        if [ "$1" == "--" ]; then
            shift
            # All remaining arguments are the ones to be forwarded
            # We use "$*" to join all remaining arguments into a single string
            # to be passed as $2 to start_project
            FORWARD_ARGS="$*"
        fi

        # If no project name was explicitly set, generate one
        if [ -z "$PROJECT_NAME" ]; then
            NEW_NAME=$(get_next_project_name "$DEFAULT_PROJECT_NAME")
            if [ $? -ne 0 ]; then exit 1; fi
            PROJECT_NAME="$NEW_NAME"
            echo "No project name specified. Generating new name: '$PROJECT_NAME'"
        fi
        
        # Pass the determined name and the forwarded arguments to the start function
        start_project "$PROJECT_NAME" "$FORWARD_ARGS"
        ;;

    stop)
        PROJECT_ARG="$1" # Now $1 is the argument after 'stop'
        if [ -z "$PROJECT_ARG" ]; then
            echo "ERROR: 'stop' command requires a PROJECT_NAME argument."
            usage
            exit 1
        fi
        stop_project "$PROJECT_ARG"
        ;;

    stopall) # NEW Command
        if [ -n "$1" ]; then
            echo "ERROR: 'stopall' command takes no arguments."
            usage
            exit 1
        fi
        stop_all_projects
        ;;

    list) # REVISED Command
        if [ -n "$1" ]; then
            echo "ERROR: 'list' command takes no arguments."
            usage
            exit 1
        fi
        list_projects
        ;;
        
    attach) # NEW Command
        PROJECT_ARG="$1" # Argument after 'attach'
        CONTAINER_ARG="$2" # Second argument after 'attach'
        
        # Only process first two arguments, discard any others
        # Arguments are checked for existence inside attach_container
        attach_container "$PROJECT_ARG" "$CONTAINER_ARG"
        ;;

    *)
        echo "ERROR: Unknown command '$COMMAND'."
        usage
        exit 1
        ;;
esac

exit 0
