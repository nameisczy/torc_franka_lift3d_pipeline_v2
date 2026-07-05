# Set default values for the first four arguments
arg1="${1:-q}"
arg2="${2:-not}"
arg3="${3:-a}"
arg4="${4:-not}"

# closed-loop: q not a gt
# open-loop: q not a a 

echo "6mux args: $arg1, $arg2, $arg3, $arg4"

# Pass the arguments to the actual script
taskset -c 18-72 ./src/lab_vbnpm/3mux_start_sim.sh "$arg1" "$arg2" "$arg3" "$arg4"
# taskset -c 40-72 ./src/lab_vbnpm/3mux_start_sim.sh q not a a
