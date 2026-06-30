# NESTA_bids: converting and merging the new nest-a batch (H7A007–H7A012)

Subjects in this batch: **H7A007, H7A008, H7A010, H7A011, H7A012** (H7A009 not
included in this pull). Session map:

| Subject | Sessions |
|---|---|
| H7A007 | ses-T1, ses-T2, ses-T3 |
| H7A008 | ses-T1, ses-T2, ses-T3 |
| H7A010 | ses-T1, ses-T2, ses-T3 |
| H7A011 | ses-T1, ses-T2, ses-T3 |
| H7A012 | ses-T1, ses-T2 |

## 0. About the fieldmaps

The `spiral high-res fieldmap` series has **no DICOM** in any subject or
session — only the two pre-exported `.nii.gz` files you already have. That's
confirmed: running `dcm2bids_helper` on it produces nothing, because
`dcm2niix` has no DICOM to read. So fieldmaps are excluded entirely from
`dcm2bids_config.json` and handled by `add_fieldmaps.py` instead, which just
copies the two existing files into `fmap/` with BIDS names and writes the
JSON sidecar by hand (`"Units": "Hz"`, confirmed from the value range you
checked: roughly -220 to +220).

## 1. Install tools

You're on `dcm2bids` 3.2.0. The config now sets `"search_method": "re"`
explicitly at the top — without it, 3.x defaults to shell-glob matching
(`fnmatch`), which silently fails on regex syntax like `^...$` and escaped
parentheses. (This is exactly what happened in your first run: every single
series in all 14 sessions failed to match, which is the signature of the
whole matching engine being wrong, not any one pattern being slightly off.)

```bash
pip install nibabel
pip install bids-validator-deno
```

Note: `pip install bids-validator` (no `-deno`) looks plausible but does **not**
install a working CLI — it's a placeholder package on PyPI. The actual BIDS
Validator 2.0 (the current, actively-maintained one, built on Deno rather
than the old Node.js version) is published under `bids-validator-deno`. If
you'd rather use the classic Node-based tool instead, `npm install -g
bids-validator` still works, just isn't the actively maintained one anymore.

## 2. Convert each subject

**If you already ran this once**: the config changed since then (DWI is
now excluded entirely, plus the `TaskName` fix below), so start clean —

```bash
rm -rf /Users/braveDP/Desktop/NESTA_bids_staging
```

Important: `dcm2niix` (which `dcm2bids` calls internally) can't read your
raw `.dicom.zip` files directly — it needs actual unzipped `.dcm` files on
disk. `convert_all.sh` now handles this for you: for each session it unzips
everything into a scratch folder under
`NESTA_bids_staging/tmp_dcm2bids/extracted_raw/`, then runs `dcm2bids`
against the extracted copy. Your original Flywheel export is never
modified. You can delete `tmp_dcm2bids/extracted_raw/` once the whole batch
is converted and validated — it's just working space, not part of the BIDS
dataset.

```bash
chmod +x convert_all.sh
./convert_all.sh "/Users/braveDP/Downloads/flywheel/padula/nest-a" /Users/braveDP/Desktop/NESTA_bids_staging
```

The script comments suggest testing on just the first subject/session
before letting the rest run, which is worth doing — open the file, comment
out everything after the `H7A007 T1` line, run it, and check that the
output landed correctly (right files in `anat/`/`func/`, no config errors)
before uncommenting the rest.

It'll warn about unmatched series (`Localizer`, `GE HOS FOV28*`) — that's
expected, those are intentionally excluded, same as your existing subjects.

If you'd rather run a session manually (e.g. one-off troubleshooting),
unzip it first, then point `dcm2bids` at the extracted folder rather than
the original:

```bash
mkdir -p /tmp/h7a007_t1_extracted
find "/Users/braveDP/Downloads/flywheel/padula/nest-a/H7A007_T1/unknown" -iname "*.dicom.zip" \
    -exec unzip -o -q {} -d /tmp/h7a007_t1_extracted \;

dcm2bids -d /tmp/h7a007_t1_extracted \
         -p H7A007 -s T1 \
         -c dcm2bids_config.json \
         -o /Users/braveDP/Desktop/NESTA_bids_staging
```

## 3. Add the fieldmaps

```bash
python add_fieldmaps.py "/Users/braveDP/Downloads/flywheel/padula/nest-a" \
    /Users/braveDP/Desktop/NESTA_bids_staging
```

Read the printed output. In particular, check `sub-H7A011/ses-T1` by hand —
that's the subject with two fieldmap acquisitions (a mid-session re-shim),
and you'll want to confirm the run/IntendedFor split looks right before
trusting it.

## 4. Validate the staging dataset on its own

```bash
bids-validator-deno /Users/braveDP/Desktop/NESTA_bids_staging
```

One `[ERROR]` is expected and fine to ignore at this stage:
`PARTICIPANT_ID_MISMATCH` on `participants.tsv` — that file still has the
scaffold's placeholder content; `merge_into_nesta_bids.sh` rebuilds it
properly in step 5. Lots of `[WARNING]`s about missing scanner metadata
(`Manufacturer`, `DeviceSerialNumber`, etc.), missing `events.tsv`, and
gzip header info are expected too — these are all "recommended, not
required" fields your existing 6 subjects likely also lack. Fix anything
else it flags now — much easier than after merging.

## 5. Merge into NESTA_bids

```bash
./merge_into_nesta_bids.sh /Users/braveDP/Desktop/NESTA_bids_staging /Users/braveDP/Desktop/NESTA_bids
```

This copies the new `sub-H7A0*` folders in and rebuilds `participants.tsv`
from scratch (it didn't exist before — the validator only requires the
`participant_id` column, so this is enough to pass; add demographic columns
whenever convenient).

## 6. Validate the combined dataset

```bash
bids-validator-deno /Users/braveDP/Desktop/NESTA_bids
```

This is the step that catches subject-ID collisions or task-naming drift
between the old and new batches — don't skip it just because each half
passed individually.

## 7. Run fMRIPrep on just the new subjects

```bash
fmriprep-docker /Users/braveDP/Desktop/NESTA_bids /Users/braveDP/Desktop/NESTA_bids/derivatives participant \
    --participant-label H7A007 H7A008 H7A010 H7A011 H7A012 \
    --fs-license-file /path/to/license.txt \
    --output-spaces MNI152NLin2009cAsym:res-2
```

`--participant-label` keeps this run from re-touching the six subjects
you've already processed. Note this batch will get fieldmap-corrected
susceptibility distortion correction, which the first six subjects did not
— worth flagging in your methods section / lab notes that the pipeline
changed partway through data collection, since it's a real (if well
justified) difference between the two cohorts that future-you or a reviewer
will ask about.

## Decisions made along the way (for your records)

- **`Cue_Reactivity` in `H7A007_T1`**: 3 real attempts converted in as
  `run-1`/`run-2`/`run-3`; a 4th acquisition (`Cue_Reactivity_NEON_Test`,
  a different underlying series, not a Flywheel duplicate) is excluded as a
  pilot/test scan. Since you're not analyzing cue reactivity yet, no attempt
  was picked as "the" run — sort that out before you do.
- **DWI excluded entirely**: per your call, `DTI 2mm b1250 84dir(axial)` is
  no longer matched by anything in the config — it gets the same "No
  Pairing" treatment as `Localizer`/`GE HOS FOV28` and never lands in the
  BIDS dataset at all. This also makes the `dwi`-folder-vs-`anat`-folder
  question moot — there's no `DATATYPE_MISMATCH` to worry about because
  there's no `dwi` file. (Your existing 6 subjects still have DWI in
  `anat/`, but that's now just a difference between batches, not something
  this conversion needs to reconcile.)
- **Stale dcm2niix cache**: `dcm2bids` caches converted output per
  subject/session inside the staging folder and silently reuses it on
  reruns unless told otherwise — `convert_all.sh` now always passes
  `--force_dcm2bids` so a rerun (e.g. after fixing one subject) can't
  accidentally skip reprocessing and reuse broken output from an earlier
  failed attempt.
- **`.dicom.zip` archives**: your raw Flywheel export has every series
  zipped, and `dcm2niix` can't read zip archives directly — this was the
  actual cause of every "NO PAIRING WAS FOUND" warning during testing, not
  the config patterns. `convert_all.sh` now unzips each session into a
  scratch folder before calling `dcm2bids`; your original raw export is
  never modified.
- **Fieldmaps have no DICOM source**: every `spiral high-res fieldmap`
  folder in this batch contains only pre-exported `.nii.gz` files, no
  `.dicom.zip`. `add_fieldmaps.py` copies them directly and writes the JSON
  sidecar itself (`Units: Hz`, confirmed from the data's value range) since
  there's no DICOM for dcm2niix to extract metadata from automatically.
- **Magnitude suffix and dimensionality**: originally wrote the magnitude
  image as `magnitude1`, which is wrong for this fieldmap type (a single
  direct field map only ever has one magnitude image, so BIDS wants the
  suffix `magnitude`, no number — `magnitude1`/`magnitude2` is only for the
  two-echo phase-difference case). Also, the raw magnitude `.nii.gz` files
  carry an extra singleton 4th dimension that BIDS' 3D requirement rejects;
  `add_fieldmaps.py` now squeezes that out via `nibabel` before writing the
  file, and will warn (not guess) if any file's extra dimension turns out
  to hold real data rather than being a singleton.
- **`TaskName` sidecar field**: `dcm2bids` doesn't auto-populate this even
  though the current validator treats it as required, not just
  recommended. Added via `sidecar_changes` in the config for each task.
- **Fieldmap `IntendedFor`**: assigned by nearest-preceding-acquisition, not
  "all runs in session" — needed because `H7A011_T1` has two fieldmaps and
  pointing both at the same runs would make fMRIPrep error out on
  ambiguous fieldmap selection.
