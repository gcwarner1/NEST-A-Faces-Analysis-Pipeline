#!/usr/bin/env bash
# merge_into_nesta_bids.sh
#
# Copies newly-converted sub-* folders from a staging BIDS dataset into the
# master NESTA_bids dataset, then (re)builds participants.tsv from whatever
# sub-* folders exist in the master dataset.
#
# Never overwrites an existing subject -- if a sub-* folder already exists
# in the target, it's skipped and flagged so you can resolve it by hand.
#
# Usage:
#   ./merge_into_nesta_bids.sh /path/to/staging_bids /Users/braveDP/Desktop/NESTA_bids

set -euo pipefail

NEW_BIDS="${1:?Usage: $0 <staging_bids_dataset> <target_NESTA_bids>}"
TARGET_BIDS="${2:?Usage: $0 <staging_bids_dataset> <target_NESTA_bids>}"

if [ ! -d "$NEW_BIDS" ]; then
  echo "Staging dataset not found: $NEW_BIDS"
  exit 1
fi
if [ ! -d "$TARGET_BIDS" ]; then
  echo "Target dataset not found: $TARGET_BIDS"
  exit 1
fi

echo "Copying new subjects from $NEW_BIDS into $TARGET_BIDS ..."
for sub_dir in "$NEW_BIDS"/sub-*; do
  [ -d "$sub_dir" ] || continue
  sub=$(basename "$sub_dir")
  if [ -e "$TARGET_BIDS/$sub" ]; then
    echo "  !! $sub already exists in $TARGET_BIDS -- NOT overwriting. Resolve manually."
    continue
  fi
  cp -R "$sub_dir" "$TARGET_BIDS/$sub"
  echo "  copied $sub"
done

PARTS="$TARGET_BIDS/participants.tsv"
echo "Rebuilding $PARTS ..."
echo -e "participant_id" > "$PARTS.new"
for sub_dir in "$TARGET_BIDS"/sub-*; do
  [ -d "$sub_dir" ] || continue
  echo -e "$(basename "$sub_dir")" >> "$PARTS.new"
done
mv "$PARTS.new" "$PARTS"
n=$(($(wc -l < "$PARTS") - 1))
echo "Wrote $PARTS with $n participants."
echo
echo "participants.tsv only has the participant_id column for now -- add age/sex/group"
echo "columns whenever you have that data handy; the validator only requires participant_id."
echo
echo "Next: run the BIDS validator on $TARGET_BIDS before pointing fMRIPrep at it."
