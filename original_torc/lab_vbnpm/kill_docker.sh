for container_name in "groundedsam" "gpd_ros"; do
  containers_to_stop=$(docker ps -q --filter ancestor="$container_name")
  if [ -n "$containers_to_stop" ]; then
    echo "Stopping containers for ancestor '$container_name'..."
    docker stop "$containers_to_stop"
  else
    echo "No running containers found for ancestor '$container_name'."
  fi
done
