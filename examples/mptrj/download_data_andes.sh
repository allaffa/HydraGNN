#!/bin/bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dataset_dir="${1:-${script_dir}/dataset}"
mkdir -p "${dataset_dir}"

wget --continue \
	https://ndownloader.figshare.com/files/41619375 \
	-O "${dataset_dir}/MPtrj_2022.9_full.json"
