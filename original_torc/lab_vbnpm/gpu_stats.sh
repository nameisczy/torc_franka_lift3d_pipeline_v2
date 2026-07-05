#!/usr/bin/env bash
# Get list of GPU indices
GPU_INDICES=( $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits) )
GPU_MEMORY=($(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits))

UNUSED_GPU_MEM=0
TOTAL_FREE_MEM=0
HAS_MEM_GPUS=()
UNUSED_GPUS=()

for i in "${!GPU_INDICES[@]}"; do
	GPU=${GPU_INDICES[$i]}
	GPU_MEM=${GPU_MEMORY[$i]}

	# Check if the GPU is being used by any process
	USED=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c $(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader | sed -n "$((GPU+1))p"))
	# Get the free memory of this GPU
	MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((GPU+1))p")
	PERCENT_FREE=$(( MEM_FREE * 100 / GPU_MEM ))

	echo $GPU is used?: $USED
	echo free memory: $MEM_FREE
	echo percent free: $PERCENT_FREE

	if [ $USED -eq 0 ]; then
		UNUSED_GPU_MEM=$((UNUSED_GPU_MEM + MEM_FREE))
		UNUSED_GPUS+=($GPU)
	fi
	if [ $PERCENT_FREE -ge 95 ]; then
		TOTAL_FREE_MEM=$((TOTAL_FREE_MEM + MEM_FREE))
		HAS_MEM_GPUS+=($GPU)
	fi
done

echo
echo "Unused GPUs: ${UNUSED_GPUS[@]}"
echo "GPUs 95% free: ${HAS_MEM_GPUS[@]}"
echo "Total Free Memory on Unused GPUs: ${UNUSED_GPU_MEM} MiB"
echo "Total Free Memory safely available: ${TOTAL_FREE_MEM} MiB"
