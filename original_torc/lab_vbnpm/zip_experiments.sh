#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIST_FILE="${ROOT_DIR}/ZIP_EXPERIMENTS"
RUNS_DIR="${ROOT_DIR}/experiments/runs"
OUTPUT_DIR="${ROOT_DIR}/experiments/zips"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_ZIP="${OUTPUT_DIR}/selected_experiments_${TIMESTAMP}.zip"

if [[ ! -f "${LIST_FILE}" ]]; then
	echo "Error: list file not found at ${LIST_FILE}"
	exit 1
fi

if [[ ! -d "${RUNS_DIR}" ]]; then
	echo "Error: runs directory not found at ${RUNS_DIR}"
	exit 1
fi

mkdir -p "${OUTPUT_DIR}"

declare -a zip_paths=()
declare -A seen=()
missing_count=0

while IFS= read -r line || [[ -n "${line}" ]]; do
	path="${line%%#*}"
	path="${path%$'\r'}"
	path="$(echo "${path}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"

	if [[ -z "${path}" ]]; then
		continue
	fi

	if [[ -n "${seen["${path}"]+x}" ]]; then
		continue
	fi
	seen["${path}"]=1

	full_path="${RUNS_DIR}/${path}"
	resolved_path="${full_path}"

	if [[ ! -d "${resolved_path}" && ! -f "${resolved_path}" ]]; then
		mapfile -t matches < <(find "${RUNS_DIR}" \( -type f -o -type d \) -path "*/${path}" 2>/dev/null)
		if [[ ${#matches[@]} -gt 0 ]]; then
			resolved_path="${matches[0]}"
			if [[ ${#matches[@]} -gt 1 ]]; then
				echo "Warn: multiple matches found for '${path}', using: ${resolved_path}"
			fi
		fi
	fi

	rel_path="${resolved_path#${ROOT_DIR}/experiments/}"

	if [[ -d "${resolved_path}" || -f "${resolved_path}" ]]; then
		zip_paths+=("${rel_path}")
	else
		echo "Warn: missing path, skipped: ${full_path}"
		missing_count=$((missing_count + 1))
	fi
done < "${LIST_FILE}"

if [[ ${#zip_paths[@]} -eq 0 ]]; then
	echo "Error: no valid experiment paths found from ${LIST_FILE}"
	exit 1
fi

echo "Creating zip archive with ${#zip_paths[@]} entries (missing ${missing_count})..."

(
	cd "${ROOT_DIR}/experiments"
	zip -r "${OUTPUT_ZIP}" "${zip_paths[@]}"
)

echo "Created zip: ${OUTPUT_ZIP}"
echo "Included entries: ${#zip_paths[@]}"
echo "Missing entries: ${missing_count}"
