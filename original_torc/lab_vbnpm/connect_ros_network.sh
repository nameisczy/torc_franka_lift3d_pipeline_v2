[ $# -ne 2 ] &&
	echo "Usage: source $0 <interface> <hostname or IP>\n" &&
	echo "Available interfaces and corresponding IP for local machine:" &&
	ip a | grep -oP "((?<=^\d:) .+:)|((?<=inet )\d+\.\d+\.\d+\.\d+)" | tr -d '\n' | tr ' ' '\n' | tr ':' ' ' &&
	echo &&
	exit 1

IFNAME="$1"
HOST="$2"
export ROS_MASTER="$(ping -qn -4 -c1 -t1 $HOST | grep -oP '\(\d+\.\d+\.\d+\.\d+\)' | tr -d '()')"
export ROS_IP="$(ip addr show $IFNAME | grep -oP 'inet \K\d+\.\d+\.\d+\.\d+')"
export ROS_MASTER_URI=http://$ROS_MASTER:11311
# export ROS_HOSTNAME=$ROS_IP
echo $ROS_MASTER
echo $ROS_IP
echo $ROS_MASTER_URI
