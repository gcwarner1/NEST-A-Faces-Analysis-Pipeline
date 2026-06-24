#!/Users/braveDP/.conda/envs/bin/python

"""
fMRI GLM Contrast Analysis — Conscious Faces Task (task-con)
============================================================
Fits a first-level GLM for each subject / session using fMRIPrep-preprocessed
data (MNI152NLin2009cAsym, res-2), then computes all pairwise contrasts among
the six emotion categories:
    Anger  |  Disgust  |  Fear  |  Happy  |  Neutral  |  Sad

Outputs (written to OUTPUT_DIR):
  • Per-subject/session stat maps  (z-score, effect-size)  as NIfTI
  • A group-level summary (second-level mean) for every contrast
  • A summary HTML report listing every file produced

Usage
-----
Before running be sure to set paths in the CONFIG session and confirm that TR, Smoothing size, High-pass, etc are appropriate for your analysis.
Once all that is verified, run the following command in a terminal.
    python facesAnalysis.py
"""

# ============================================================
#  CONFIG  — edit these paths to match your machine
# ============================================================
BIDS_DIR    = "/Users/braveDP/Desktop/NEST-A/NESTA_bids"
DERIV_DIR   = f"{BIDS_DIR}/derivatives"
EVENTS_FILE = "/Users/braveDP/Desktop/NEST-A/Faces/Conscious/events.tsv"
OUTPUT_DIR  = "/Users/braveDP/Desktop/NEST-A/Outputs/NESTA_faces_GLM"

# Repetition time (seconds) for the conscious faces task
TR = 2.0

# fMRIPrep output space / resolution used in file names
SPACE      = "MNI152NLin2009cAsym"
RESOLUTION = "2"

# High-pass filter cut-off (Hz)
HIGH_PASS = 0.01

# Gaussian spatial smoothing kernel size for first-level analysis
SMOOTHING_SIZE = 6.0

# HRF model
HRF_MODEL = "spm"

# ============================================================
#  IMPORTS
# ============================================================
import os
import glob
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from nilearn.glm.first_level         import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level        import SecondLevelModel
from nilearn.reporting               import make_glm_report
from nilearn                         import image, plotting
from nilearn.datasets                import load_mni152_brain_mask
from nilearn.interfaces.fmriprep     import load_confounds as load_confounds_nilearn

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
#  HELPERS
# ============================================================

EMOTIONS = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]

# All ordered pairwise contrasts A > B  (A - B)
PAIRWISE_CONTRASTS = list(itertools.permutations(EMOTIONS, 2))

# Each emotion vs. all others (one-vs-rest)
OVR_CONTRASTS = {
    e: {cond: (1.0 if cond == e else -1.0 / (len(EMOTIONS) - 1))
        for cond in EMOTIONS}
    for e in EMOTIONS
}


def find_subjects(deriv_dir: str) -> list[str]:
    """Return sorted list of subject IDs that have at least one task-con run."""
    pattern = os.path.join(
        deriv_dir, "sub-*", "ses-*", "func",
        f"*task-con_space-{SPACE}_res-{RESOLUTION}_desc-preproc_bold.nii.gz"
    )
    hits = glob.glob(pattern)
    subjects = sorted({Path(p).name.split("_")[0] for p in hits})
    return subjects


def find_runs(deriv_dir: str, subject: str) -> list[dict]:
    """
    Return list of dicts, one per (subject, session) run, with keys:
        bold, confounds, mask
    """
    pattern = os.path.join(
        deriv_dir, subject, "ses-*", "func",
        f"{subject}_ses-*_task-con_space-{SPACE}_res-{RESOLUTION}_desc-preproc_bold.nii.gz"
    )
    runs = []
    for bold_path in sorted(glob.glob(pattern)):
        p        = Path(bold_path)
        stem     = p.name
        ses      = stem.split("_ses-")[1].split("_")[0]
        func_dir = p.parent

        conf_path = func_dir / f"{subject}_ses-{ses}_task-con_desc-confounds_timeseries.tsv"
        mask_path = func_dir / f"{subject}_ses-{ses}_task-con_space-{SPACE}_res-{RESOLUTION}_desc-brain_mask.nii.gz"

        if not conf_path.exists():
            print(f"  [WARN] No confound file for {subject} ses-{ses} — skipping run.")
            continue

        runs.append(dict(
            subject=subject,
            session=ses,
            bold=str(bold_path),
            confounds=str(conf_path),
            mask=str(mask_path) if mask_path.exists() else None,
        ))
    return runs


def load_confounds_for_run(bold_path: str) -> pd.DataFrame:
    """
    Load motion confounds using nilearn's fMRIPrep interface.
    Returns a DataFrame of 6 motion parameters, NaNs handled automatically.
    """
    confounds, _ = load_confounds_nilearn(
        bold_path,
        strategy=["motion"],
        motion="basic"
    )
    return confounds


def build_contrast_matrix(design_matrix_columns: list[str]) -> dict[str, np.ndarray]:
    """
    Build contrast vectors for:
      1. Each emotion vs. implicit baseline
      2. All pairwise contrasts  (A - B)
      3. Each emotion vs. rest (OVR)
    Returns dict  {contrast_name: contrast_vector}
    """
    cols      = design_matrix_columns
    n         = len(cols)
    contrasts = {}

    # Locate emotion regressors in the design matrix
    emotion_idx = {e: cols.index(e) for e in EMOTIONS if e in cols}

    # 1. Each emotion vs. baseline
    for emo, idx in emotion_idx.items():
        vec = np.zeros(n)
        vec[idx] = 1.0
        contrasts[emo] = vec

    # 2. Pairwise  A > B
    for (a, b) in PAIRWISE_CONTRASTS:
        if a in emotion_idx and b in emotion_idx:
            vec = np.zeros(n)
            vec[emotion_idx[a]] =  1.0
            vec[emotion_idx[b]] = -1.0
            contrasts[f"{a}_gt_{b}"] = vec

    # 3. One-vs-rest
    for emo, weights in OVR_CONTRASTS.items():
        if all(e in emotion_idx for e in EMOTIONS):
            vec = np.zeros(n)
            for e, w in weights.items():
                vec[emotion_idx[e]] = w
            contrasts[f"{emo}_vs_rest"] = vec

    return contrasts


# ============================================================
#  FIRST-LEVEL GLM
# ============================================================

def run_first_level(run_info: dict, events: pd.DataFrame,
                    out_dir: Path) -> dict[str, str]:
    """
    Fit GLM for one (subject, session) run.
    Returns dict of {contrast_name: z_map_path}.
    """
    subj = run_info["subject"]
    ses  = run_info["session"]
    tag  = f"{subj}_ses-{ses}"
    print(f"\n  Fitting GLM: {tag}")

    # --- confounds via nilearn's fMRIPrep interface ---
    confounds = load_confounds_for_run(run_info["bold"])

    # --- GLM ---
    glm = FirstLevelModel(
        t_r=TR,
        hrf_model=HRF_MODEL,
        drift_model="cosine",
        high_pass=HIGH_PASS,
        mask_img=run_info["mask"],
        noise_model="ar1",
        standardize=False,
        smoothing_fwhm=SMOOTHING_SIZE,
        minimize_memory=True,
        n_jobs=1,
    )
    glm.fit(run_info["bold"], events=events, confounds=confounds)

    # --- contrasts ---
    dm_cols   = glm.design_matrices_[0].columns.tolist()
    contrasts = build_contrast_matrix(dm_cols)

    z_maps = {}
    for cname, cvec in contrasts.items():
        try:
            z_map   = glm.compute_contrast(cvec, stat_type="t", output_type="z_score")
            eff_map = glm.compute_contrast(cvec, stat_type="t", output_type="effect_size")

            z_path   = out_dir / f"{tag}_{cname}_z.nii.gz"
            eff_path = out_dir / f"{tag}_{cname}_effect.nii.gz"

            z_map.to_filename(str(z_path))
            eff_map.to_filename(str(eff_path))

            z_maps[cname] = str(z_path)
        except Exception as exc:
            print(f"    [WARN] Contrast '{cname}' failed: {exc}")

    # --- save design matrix plot ---
    dm_fig_path = out_dir / f"{tag}_design_matrix.png"
    dm = glm.design_matrices_[0]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(dm.values.T, aspect="auto", cmap="gray", interpolation="nearest")
    ax.set_yticks(range(len(dm.columns)))
    ax.set_yticklabels(dm.columns, fontsize=7)
    ax.set_xlabel("Scan (TR)")
    ax.set_title(f"Design matrix — {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(dm_fig_path), dpi=150)
    plt.close(fig)
    print(f"    Design matrix saved → {dm_fig_path.name}")
    print(f"    {len(z_maps)} contrast maps saved.")
    return z_maps


# ============================================================
#  SECOND-LEVEL (GROUP) GLM
# ============================================================

def run_second_level(all_z_maps: dict[str, list[str]],
                     out_dir: Path) -> dict[str, str]:
    """
    Compute a simple group mean z-map for every contrast using a
    second-level intercept-only model.
    Returns dict of {contrast_name: group_z_map_path}.
    """
    group_maps = {}
    print("\n--- Second-level (group) analysis ---")

    # Resample MNI mask to match the voxel grid of the first-level maps
    target_img         = image.load_img(list(all_z_maps.values())[0][0])
    mni_mask           = load_mni152_brain_mask(resolution=2)
    mni_mask_resampled = image.resample_to_img(mni_mask, target_img,
                                               interpolation="nearest")

    for cname, z_paths in all_z_maps.items():
        if len(z_paths) < 2:
            print(f"  [SKIP] {cname}: only {len(z_paths)} run(s), need ≥ 2 for group.")
            continue

        # Resample all z-maps to the same grid before concatenating
        resampled = [image.resample_to_img(image.load_img(p), target_img,
                                           interpolation="continuous")
                     for p in z_paths]
        concat = image.concat_imgs(resampled)

        design = pd.DataFrame({"intercept": np.ones(len(z_paths))})

        slm = SecondLevelModel(mask_img=mni_mask_resampled, smoothing_fwhm=None)
        slm.fit(concat, design_matrix=design)

        group_z = slm.compute_contrast(
            second_level_contrast="intercept",
            output_type="z_score"
        )
        out_path = out_dir / f"group_{cname}_z.nii.gz"
        group_z.to_filename(str(out_path))
        group_maps[cname] = str(out_path)
        print(f"  {cname:40s}  n={len(z_paths):3d}  → {out_path.name}")

    return group_maps


# ============================================================
#  QUICK VISUALIZATION
# ============================================================

def save_glass_brains(group_maps: dict[str, str], out_dir: Path,
                      threshold: float = 3.1):
    """
    Save a glass-brain image for each group contrast map.
    Default threshold z >= 3.1 ~ p < 0.001 uncorrected (one-tailed).
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    print(f"\nSaving glass-brain figures (threshold z={threshold}) ...")

    for cname, gmap_path in group_maps.items():
        try:
            fig_path = fig_dir / f"group_{cname}_glassbrain.png"
            plotting.plot_glass_brain(
                gmap_path,
                threshold=threshold,
                colorbar=True,
                plot_abs=False,
                title=f"Group: {cname}  (z>{threshold})",
                output_file=str(fig_path),
            )
            print(f"  Saved: {fig_path.name}")
        except Exception as exc:
            print(f"  [WARN] glass-brain failed for {cname}: {exc}")


# ============================================================
#  HTML SUMMARY REPORT
# ============================================================

def write_html_report(subjects: list[str], all_z_maps: dict[str, list[str]],
                      group_maps: dict[str, str], out_dir: Path):
    html = ["<html><head><title>Faces GLM Report</title>",
            "<style>body{font-family:sans-serif;margin:2em}",
            "h2{color:#2c5f8a} table{border-collapse:collapse}",
            "td,th{border:1px solid #ccc;padding:4px 8px}",
            "th{background:#eef}</style></head><body>",
            "<h1>Conscious Faces Task — GLM Contrast Report</h1>",
            f"<p><b>Subjects:</b> {len(subjects)}</p>",
            f"<p><b>Emotions:</b> {', '.join(EMOTIONS)}</p>",
            "<h2>Group contrast maps</h2><table>",
            "<tr><th>Contrast</th><th>N runs</th><th>File</th></tr>"]
    for cname, gmap in group_maps.items():
        n = len(all_z_maps.get(cname, []))
        html.append(f"<tr><td>{cname}</td><td>{n}</td>"
                    f"<td>{Path(gmap).name}</td></tr>")
    html.append("</table>")

    html.append("<h2>First-level maps (per run)</h2><table>")
    html.append("<tr><th>Contrast</th><th>N runs</th></tr>")
    for cname, paths in all_z_maps.items():
        html.append(f"<tr><td>{cname}</td><td>{len(paths)}</td></tr>")
    html.append("</table>")

    html.append("<h2>Output directory</h2>")
    html.append(f"<p><code>{out_dir}</code></p>")
    html.append("</body></html>")

    report_path = out_dir / "report.html"
    report_path.write_text("\n".join(html))
    print(f"\nHTML report → {report_path}")


# ============================================================
#  MAIN
# ============================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fl_dir = out_dir / "first_level"
    fl_dir.mkdir(exist_ok=True)
    sl_dir = out_dir / "second_level"
    sl_dir.mkdir(exist_ok=True)

    # Load events (same for all runs)
    events = pd.read_csv(EVENTS_FILE, sep="\t")
    print(f"Events loaded: {len(events)} trials, "
          f"conditions: {sorted(events['trial_type'].unique())}")

    # Discover subjects and runs
    subjects = find_subjects(DERIV_DIR)
    if not subjects:
        raise FileNotFoundError(
            f"No preprocessed task-con BOLD files found under {DERIV_DIR}.\n"
            "Check BIDS_DIR / DERIV_DIR paths in the CONFIG section."
        )
    print(f"\nFound {len(subjects)} subject(s): {subjects}")

    # First-level GLM per run, accumulate z-maps by contrast name
    all_z_maps: dict[str, list[str]] = {}
    for subj in subjects:
        runs = find_runs(DERIV_DIR, subj)
        print(f"\n{subj}: {len(runs)} task-con run(s)")
        for run in runs:
            z_maps = run_first_level(run, events, fl_dir)
            for cname, zpath in z_maps.items():
                all_z_maps.setdefault(cname, []).append(zpath)

    if not all_z_maps:
        raise RuntimeError("No contrast maps were produced. Check paths and file names.")

    # Second-level group analysis
    group_maps = run_second_level(all_z_maps, sl_dir)

    # Glass-brain figures
    save_glass_brains(group_maps, out_dir)

    # Summary report
    write_html_report(subjects, all_z_maps, group_maps, out_dir)

    print("\n✓  Analysis complete.")
    print(f"   Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
