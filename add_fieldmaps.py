#!/usr/bin/env python3
"""
add_fieldmaps.py

The "spiral high-res fieldmap" series has NO DICOM in ANY subject/session --
only two pre-exported .nii.gz files (a magnitude image and an already-computed
real fieldmap in Hz, confirmed by the value range: roughly -220 to +220).
dcm2bids/dcm2niix has nothing to work with there, so this script handles
fieldmaps on its own, after you've run dcm2bids for everything else:

  1. Copies the two raw .nii.gz files into fmap/, renamed to BIDS
     convention. This is a "Case 3" direct field map (one magnitude image,
     one real fieldmap in Hz, no second echo) -- the correct BIDS suffix for
     the magnitude image in this case is "magnitude" (no trailing number;
     "magnitude1" is only for the two-magnitude phase-difference case, which
     this isn't).
  2. Squeezes the magnitude image down to 3D before writing it -- BIDS
     requires magnitude images to be exactly 3D. In this dataset the raw
     files are shaped (256, 256, 45, 2, 1): a trailing singleton dimension
     plus a genuine dual-echo dimension (2 echoes bundled into one file).
     The script drops the singleton and takes the first echo as the
     magnitude reference (logged explicitly). If a file's shape doesn't
     match this pattern, it stops and warns rather than guessing.
  3. Writes a JSON sidecar for the fieldmap file with "Units": "Hz" (there's
     no DICOM, so there's no sidecar to inherit metadata from -- this is the
     one piece of metadata we know from your protocol).
  4. Computes "IntendedFor" the same way as before: each BOLD run gets
     whichever fieldmap was acquired most recently before it, using the scan
     number embedded in the raw filenames (fieldmap side) and the
     SeriesNumber that dcm2niix already wrote into each func/*_bold.json
     (BOLD side, which DOES have real DICOM).
  5. H7A011_T1 has two fieldmap acquisitions -- they'll come out as
     fmap/..._run-1_... and ..._run-2_..., split by the rule in #4. Spot
     check that one by hand once converted.

This script has the exact raw paths for this batch (H7A007/8/10/11/12)
hardcoded into MANIFEST below, built directly from nestTreeTwo.txt. If you
pull more subjects later, add entries to MANIFEST rather than re-deriving
this from scratch.

Usage:
    python add_fieldmaps.py /Users/braveDP/Downloads/flywheel/padula/nest-a /Users/braveDP/Desktop/NESTA_bids_staging
"""
import json
import sys
import os
import shutil
import glob

try:
    import numpy as np
    import nibabel as nib
except ImportError:
    print("This script needs numpy and nibabel to process the magnitude images.")
    print("Install with: pip install numpy nibabel")
    sys.exit(1)

# sub/ses -> list of raw fieldmap "stems" (path relative to nest-a root,
# without the .nii.gz / _fieldmap.nii.gz suffix). Two entries = two
# fieldmap acquisitions in that session (H7A011_T1 only).
MANIFEST = {
    ("sub-H7A007", "ses-T1"): ["H7A007_T1/unknown/spiral high-res fieldmap/32965_3_1"],
    ("sub-H7A007", "ses-T2"): ["H7A007_T2/33040/spiral high-res fieldmap/33040_3_1"],
    ("sub-H7A007", "ses-T3"): ["H7A007_T3/33099/spiral high-res fieldmap/33099_3_1"],
    ("sub-H7A008", "ses-T1"): ["H7A008_T1/33004/spiral high-res fieldmap/33004_3_1"],
    ("sub-H7A008", "ses-T2"): ["H7A008_T2/unknown/spiral high-res fieldmap/33060_3_1"],
    ("sub-H7A008", "ses-T3"): ["H7A008_T3/33109/spiral high-res fieldmap/33109_3_1"],
    ("sub-H7A010", "ses-T1"): ["H7A010_T1/33256/spiral high-res fieldmap/33256_3_1"],
    ("sub-H7A010", "ses-T2"): ["H7A010_T2/unknown/spiral high-res fieldmap/33335_3_1"],
    ("sub-H7A010", "ses-T3"): ["H7A010_T3/33405/spiral high-res fieldmap/33405_3_1"],
    ("sub-H7A011", "ses-T1"): [
        "H7A011_T1/33308/spiral high-res fieldmap/33308_3_1",
        "H7A011_T1/33308/spiral high-res fieldmap_1/33308_6_1",
    ],
    ("sub-H7A011", "ses-T2"): ["H7A011_T2/33350/spiral high-res fieldmap/33350_3_1"],
    ("sub-H7A011", "ses-T3"): ["H7A011_T3/33388/spiral high-res fieldmap/33388_3_1"],
    ("sub-H7A012", "ses-T1"): ["H7A012_T1/33518/spiral high-res fieldmap/33518_3_1"],
    ("sub-H7A012", "ses-T2"): ["H7A012_T2/33543/spiral high-res fieldmap/33543_3_1"],
}


def scan_number_from_stem(stem):
    # stem looks like ".../33308_6_1" -> scan number is the middle integer (6)
    base = os.path.basename(stem)
    parts = base.split("_")
    return int(parts[-2])


def series_number(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data.get("SeriesNumber", data.get("AcquisitionNumber"))


def main(raw_root, bids_root):
    for (sub, ses), stems in MANIFEST.items():
        ses_dir = os.path.join(bids_root, sub, ses)
        fmap_dir = os.path.join(ses_dir, "fmap")
        func_dir = os.path.join(ses_dir, "func")

        if not os.path.isdir(func_dir):
            print(f"!! {sub}/{ses}: no func/ folder found in {bids_root} -- "
                  f"did dcm2bids run for this session? skipping")
            continue
        os.makedirs(fmap_dir, exist_ok=True)

        multi = len(stems) > 1
        fmap_scan_numbers = []  # (scan_number, magnitude_path, fieldmap_json_path)

        for i, stem in enumerate(sorted(stems, key=scan_number_from_stem), start=1):
            stem_path = os.path.join(raw_root, stem)
            mag_src = stem_path + ".nii.gz"
            fm_src = stem_path + "_fieldmap.nii.gz"

            if not os.path.exists(mag_src) or not os.path.exists(fm_src):
                print(f"!! {sub}/{ses}: expected files not found next to {stem_path}, skipping this fieldmap")
                continue

            run_tag = f"_run-{i}" if multi else ""
            mag_dst = os.path.join(fmap_dir, f"{sub}_{ses}{run_tag}_magnitude.nii.gz")
            fm_dst = os.path.join(fmap_dir, f"{sub}_{ses}{run_tag}_fieldmap.nii.gz")
            fm_json_dst = os.path.join(fmap_dir, f"{sub}_{ses}{run_tag}_fieldmap.json")

            img = nib.load(mag_src)
            data = np.squeeze(img.get_fdata())  # drop trailing singleton dims first

            if data.ndim == 3:
                squeezed = nib.Nifti1Image(data, img.affine, img.header)
                squeezed.header.set_data_shape(data.shape)
                nib.save(squeezed, mag_dst)
                print(f"{sub}/{ses}: squeezed {os.path.basename(mag_src)} "
                      f"from {img.shape} to {data.shape}")
            elif data.ndim == 4 and data.shape[3] == 2:
                # This is the real shape seen in this dataset:
                # (256, 256, 45, 2, 1) -> squeeze -> (256, 256, 45, 2).
                # The trailing 2 is a dual-echo GE field map acquisition
                # bundled into one file. The precomputed real fieldmap
                # (separate _fieldmap.nii.gz) is what actually drives
                # distortion correction -- this magnitude image is mainly a
                # registration reference, so taking the first (shorter TE)
                # echo is a reasonable, low-risk default. Logged explicitly
                # so it's easy to revisit if it turns out to matter.
                vol0 = data[..., 0]
                squeezed = nib.Nifti1Image(vol0, img.affine, img.header)
                squeezed.header.set_data_shape(vol0.shape)
                nib.save(squeezed, mag_dst)
                print(f"{sub}/{ses}: {os.path.basename(mag_src)} had shape "
                      f"{img.shape} (dual-echo) -- took echo 1 of 2, "
                      f"wrote {vol0.shape}")
            else:
                print(f"!! {sub}/{ses}: {mag_src} has shape {img.shape} -- "
                      f"doesn't match the expected dual-echo pattern. "
                      f"NOT auto-picking a volume; copying as-is, fix by hand.")
                shutil.copy2(mag_src, mag_dst)

            shutil.copy2(fm_src, fm_dst)
            print(f"{sub}/{ses}: copied {os.path.basename(mag_dst)}, {os.path.basename(fm_dst)}")

            scan_num = scan_number_from_stem(stem)
            fmap_scan_numbers.append((scan_num, fm_json_dst))

        if not fmap_scan_numbers:
            continue

        fmap_scan_numbers.sort()

        # Assign each BOLD run to nearest-preceding fieldmap
        intended = {j: [] for _, j in fmap_scan_numbers}
        bold_jsons = sorted(glob.glob(os.path.join(func_dir, "*_bold.json")))
        for bj in bold_jsons:
            bn = series_number(bj)
            if bn is None:
                print(f"  ! no SeriesNumber in {bj}, skipping this run")
                continue
            preceding = [(n, j) for n, j in fmap_scan_numbers if n < bn]
            target = max(preceding, default=fmap_scan_numbers[0])[1]
            bold_nii = bj[: -len(".json")] + ".nii.gz"
            rel = f"bids::{sub}/{ses}/func/{os.path.basename(bold_nii)}"
            intended[target].append(rel)

        for _, fm_json_dst in fmap_scan_numbers:
            with open(fm_json_dst, "w") as f:
                json.dump({
                    "Units": "Hz",
                    "IntendedFor": intended[fm_json_dst],
                }, f, indent=4)
            print(f"{sub}/{ses}: {os.path.basename(fm_json_dst)} -> "
                  f"IntendedFor {len(intended[fm_json_dst])} run(s)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
