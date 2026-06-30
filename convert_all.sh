#!/usr/bin/env bash
# convert_all.sh
#
# Runs dcm2bids once per subject/session for this batch (H7A007, H7A008,
# H7A010, H7A011, H7A012), using the exact raw subfolder for each session
# (some sessions store everything under a numeric Flywheel session label,
# others under "unknown" -- this script already knows which is which).
#
# Recommended: comment out everything except the first line the first time
# you run this, confirm the output looks right (correct files landing in
# the right datatype folders, no config errors), then uncomment the rest
# and let it run end to end.
#
# Usage:
#   chmod +x convert_all.sh
#   ./convert_all.sh "/Users/braveDP/Downloads/flywheel/padula/nest-a" /Users/braveDP/Desktop/NESTA_bids_staging

set -euo pipefail

RAW_ROOT="${1:?Usage: $0 <nest-a raw root> <bids staging output dir>}"
BIDS_OUT="${2:?Usage: $0 <nest-a raw root> <bids staging output dir>}"
CONFIG="$(cd "$(dirname "$0")" && pwd)/dcm2bids_config.json"

mkdir -p "$BIDS_OUT"
if [ ! -f "$BIDS_OUT/dataset_description.json" ]; then
  dcm2bids_scaffold -o "$BIDS_OUT"
fi

run_one () {
  local sub="$1" ses="$2" subdir="$3"
  local src_dir="$RAW_ROOT/$subdir"
  local extract_dir="$BIDS_OUT/tmp_dcm2bids/extracted_raw/${sub}_ses-${ses}"

  echo "=== $sub ses-$ses (raw: $subdir) ==="

  # dcm2niix can't read .dicom.zip archives directly -- unzip everything for
  # this session into a scratch folder first, then point dcm2bids at that
  # instead of the original (untouched) raw export.
  mkdir -p "$extract_dir"
  shopt -s nullglob
  while IFS= read -r -d '' zipfile; do
    unzip -o -q "$zipfile" -d "$extract_dir"
  done < <(find "$src_dir" -iname "*.dicom.zip" -print0)

  if [ -z "$(find "$extract_dir" -iname "*.dcm" -print -quit)" ]; then
    echo "  !! no .dcm files found after unzipping $src_dir -- check the raw path"
    return
  fi

  dcm2bids -d "$extract_dir" -p "$sub" -s "$ses" -c "$CONFIG" -o "$BIDS_OUT" --force_dcm2bids
}

run_one H7A007 T1 "H7A007_T1/unknown"
run_one H7A007 T2 "H7A007_T2/33040"
run_one H7A007 T3 "H7A007_T3/33099"

run_one H7A008 T1 "H7A008_T1/33004"
run_one H7A008 T2 "H7A008_T2/33060"
run_one H7A008 T3 "H7A008_T3/33109"

run_one H7A010 T1 "H7A010_T1/33256"
run_one H7A010 T2 "H7A010_T2/unknown"
run_one H7A010 T3 "H7A010_T3/33405"

run_one H7A011 T1 "H7A011_T1/33308"
run_one H7A011 T2 "H7A011_T2/33350"
run_one H7A011 T3 "H7A011_T3/33388"

run_one H7A012 T1 "H7A012_T1/33518"
run_one H7A012 T2 "H7A012_T2/33543"

echo
echo "All sessions converted. Next:"
echo "  python add_fieldmaps.py \"$RAW_ROOT\" \"$BIDS_OUT\""
