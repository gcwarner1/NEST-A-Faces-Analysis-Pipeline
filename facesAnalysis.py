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

# ------------------------------------------------------------
# dACC longitudinal (TMS treatment) analysis config
# ------------------------------------------------------------
# Brainnetome Atlas 246 (2mm — matches the res-2 functional grid)
BN_ATLAS_PATH = "/Users/braveDP/Desktop/BN_Atlas_246_2mm.nii.gz"
BN_ATLAS_LUT  = "/Users/braveDP/Desktop/BN_Atlas_246_LUT.txt"

# dACC = caudodorsal area 24 + dorsal (pregenual) area 32
DACC_REGION_NAMES = ["A24cd", "A32p"]

# Contrast used for the dACC fear-response analysis
DACC_CONTRAST = "Fear_gt_Neutral"

# Study visit (session) order/labels — TMS treatment course
TIMEPOINT_ORDER = ["T1", "T2", "T3", "T4"]
TIMEPOINT_LABELS = {
    "T1": "T1\nBaseline\n(pre-TMS)",
    "T2": "T2\n(mid-TMS)",
    "T3": "T3\n(mid-TMS)",
    "T4": "T4\nPost-TMS\n(end of course)",
}

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
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats as sstats

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
#  dACC LONGITUDINAL FEAR-RESPONSE ANALYSIS (TMS TREATMENT COURSE)
# ============================================================
"""
Builds left / right / bilateral dACC ROI masks from the Brainnetome Atlas 246
(A24cd + A32p, per hemisphere), extracts each subject's mean Fear>Neutral
z-value within those ROIs at every available study visit (ses-T1..T4), and
produces:
  • Three "spaghetti" plots (one per ROI: L, R, bilateral) — one line per
    subject across the timepoints they have data for, with the group
    mean +/- SEM overlaid.
  • A single CSV summarizing group-level descriptives and paired
    timepoint-to-timepoint statistics for each ROI, to help answer
    "is the treatment working?"
"""

def parse_bn_lut(lut_path: str) -> dict[str, int]:
    """Parse the Brainnetome LUT into {region_name: label_id}."""
    labels = {}
    with open(lut_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            name = parts[1]
            labels[name] = idx
    return labels


def get_dacc_label_ids(lut: dict[str, int], hemi: str) -> list[int]:
    """
    hemi: 'L', 'R', or 'bilateral'.
    Returns the Brainnetome label IDs making up the dACC ROI
    (A24cd + A32p) for the requested hemisphere(s).
    """
    suffixes = []
    if hemi in ("L", "bilateral"):
        suffixes.append("_L")
    if hemi in ("R", "bilateral"):
        suffixes.append("_R")

    ids = []
    missing = []
    for region in DACC_REGION_NAMES:
        for suf in suffixes:
            name = f"{region}{suf}"
            if name in lut:
                ids.append(lut[name])
            else:
                missing.append(name)
    if missing:
        print(f"  [WARN] dACC label(s) not found in LUT: {missing}")
    return ids


def build_dacc_masks(atlas_path: str, lut_path: str, target_img) -> dict[str, "image.Nifti1Image"]:
    """
    Resample the Brainnetome atlas onto the grid of `target_img` (a
    first-level z-map, since all subjects/sessions share the same
    MNI152NLin2009cAsym res-2 grid) and build binary dACC masks for
    left, right, and bilateral.
    """
    lut = parse_bn_lut(lut_path)
    atlas_img = image.load_img(atlas_path)
    atlas_resampled = image.resample_to_img(atlas_img, target_img, interpolation="nearest")
    atlas_data = atlas_resampled.get_fdata()

    masks = {}
    for hemi in ("L", "R", "bilateral"):
        label_ids = get_dacc_label_ids(lut, hemi)
        if not label_ids:
            print(f"  [WARN] No dACC labels resolved for hemi='{hemi}' — skipping mask.")
            continue
        mask_data = np.isin(atlas_data, label_ids).astype(np.uint8)
        n_vox = int(mask_data.sum())
        print(f"  dACC[{hemi}] ROI: labels={label_ids}  n_voxels={n_vox}")
        masks[hemi] = image.new_img_like(atlas_resampled, mask_data)
    return masks


def extract_roi_mean(zmap_path: str, mask_img) -> float | None:
    """Mean z-value within a binary ROI mask for one subject/session z-map."""
    z_img = image.load_img(zmap_path)
    mask_data = mask_img.get_fdata().astype(bool)
    if mask_data.sum() == 0:
        return None
    z_data = z_img.get_fdata()
    if z_data.shape != mask_data.shape:
        z_img = image.resample_to_img(z_img, mask_img, interpolation="continuous")
        z_data = z_img.get_fdata()
    return float(np.nanmean(z_data[mask_data]))


def collect_dacc_values(dacc_records: list[dict], masks: dict[str, "image.Nifti1Image"]) -> pd.DataFrame:
    """
    dacc_records: list of {subject, timepoint, zmap_path} for the
    DACC_CONTRAST (one entry per subject/session that has this contrast).
    Returns a tidy long-format DataFrame:
        subject | timepoint | hemi | mean_z
    """
    rows = []
    for rec in dacc_records:
        for hemi, mask_img in masks.items():
            val = extract_roi_mean(rec["zmap_path"], mask_img)
            if val is not None:
                rows.append(dict(subject=rec["subject"], timepoint=rec["timepoint"],
                                 hemi=hemi, mean_z=val))
    return pd.DataFrame(rows)


# ---- publication-style plotting ----------------------------

_HEMI_TITLES = {
    "L": "Left dACC",
    "R": "Right dACC",
    "bilateral": "Bilateral dACC",
}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(width=1.2)


def make_dacc_spaghetti_plot(df_long: pd.DataFrame, hemi: str, out_path: Path):
    """
    One polished spaghetti plot for a single ROI (hemi in 'L'/'R'/'bilateral'):
    a thin line per subject across whichever timepoints they have, with the
    group mean +/- SEM overlaid as a bold trend line.
    """
    sub = df_long[df_long["hemi"] == hemi]
    if sub.empty:
        print(f"  [WARN] No data for dACC[{hemi}] — skipping plot.")
        return

    wide = sub.pivot_table(index="subject", columns="timepoint", values="mean_z")
    wide = wide.reindex(columns=TIMEPOINT_ORDER)
    x = np.arange(len(TIMEPOINT_ORDER))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titleweight": "bold",
    })
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)

    n_subj = wide.shape[0]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(n_subj - 1, 1)) for i in range(n_subj)]

    for color, (subj, row) in zip(colors, wide.iterrows()):
        y = row.values.astype(float)
        ax.plot(x, y, marker="o", markersize=4, linewidth=1.1,
                color=color, alpha=0.55, zorder=2)

    # group mean +/- SEM
    means = wide.mean(axis=0, skipna=True).values
    sems  = wide.sem(axis=0, skipna=True).values
    ns    = wide.count(axis=0).values
    ax.errorbar(x, means, yerr=sems, color="black", linewidth=2.6,
                marker="D", markersize=7, markerfacecolor="white",
                markeredgewidth=2, capsize=5, zorder=5,
                label="Group mean ± SEM")

    for xi, n in zip(x, ns):
        ax.annotate(f"n={int(n)}", xy=(xi, ax.get_ylim()[1]), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="dimgray")

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TIMEPOINT_LABELS.get(tp, tp) for tp in TIMEPOINT_ORDER], fontsize=9.5)
    ax.set_xlim(-0.4, len(TIMEPOINT_ORDER) - 0.6)
    ax.set_ylabel("Fear > Neutral (mean z-score)", fontsize=12)
    ax.set_xlabel("Study visit", fontsize=12)
    ax.set_title(f"{_HEMI_TITLES[hemi]}: Fear vs. Neutral Activity\nAcross TMS Treatment Course",
                fontsize=13)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    _style_axes(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".png", ".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name} (+ .svg)")


# ---- group statistics ---------------------------------------

def _cohens_d_paired(diff: np.ndarray) -> float:
    sd = np.nanstd(diff, ddof=1)
    return float(np.nanmean(diff) / sd) if sd > 0 else np.nan


def compute_dacc_stats(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    For each ROI (L/R/bilateral), compute:
      - group mean +/- SEM and n at each timepoint
      - paired t-test (and Wilcoxon signed-rank as a nonparametric check)
        for each consecutive timepoint transition (T1->T2, T2->T3, T3->T4)
      - paired baseline-vs-end-of-course test (T1 -> T4), the primary
        "did the treatment change fear response" comparison
      - Cohen's d for paired differences
    Returns one tidy DataFrame combining descriptives and contrasts.
    """
    rows = []

    for hemi in df_long["hemi"].unique():
        sub = df_long[df_long["hemi"] == hemi]
        wide = sub.pivot_table(index="subject", columns="timepoint", values="mean_z")
        wide = wide.reindex(columns=TIMEPOINT_ORDER)

        # --- descriptives per timepoint ---
        for tp in TIMEPOINT_ORDER:
            if tp not in wide.columns:
                continue
            vals = wide[tp].dropna().values
            rows.append(dict(
                hemi=hemi, comparison=f"descriptive_{tp}",
                timepoint_A=tp, timepoint_B=None,
                n=len(vals),
                mean_A=np.nanmean(vals) if len(vals) else np.nan,
                sem_A=sstats.sem(vals, nan_policy="omit") if len(vals) else np.nan,
                mean_B=np.nan, sem_B=np.nan, mean_diff=np.nan,
                t_stat=np.nan, p_value=np.nan,
                wilcoxon_stat=np.nan, wilcoxon_p=np.nan,
                cohens_d=np.nan,
            ))

        # --- paired comparisons: consecutive + overall (T1 vs T4) ---
        comparisons = list(zip(TIMEPOINT_ORDER[:-1], TIMEPOINT_ORDER[1:]))
        if "T1" in wide.columns and "T4" in wide.columns:
            comparisons.append(("T1", "T4"))

        for tA, tB in comparisons:
            if tA not in wide.columns or tB not in wide.columns:
                continue
            paired = wide[[tA, tB]].dropna()
            n = len(paired)
            label = f"{tA}_vs_{tB}"
            if n < 2:
                rows.append(dict(
                    hemi=hemi, comparison=label, timepoint_A=tA, timepoint_B=tB,
                    n=n, mean_A=paired[tA].mean() if n else np.nan,
                    sem_A=np.nan, mean_B=paired[tB].mean() if n else np.nan,
                    sem_B=np.nan, mean_diff=np.nan, t_stat=np.nan, p_value=np.nan,
                    wilcoxon_stat=np.nan, wilcoxon_p=np.nan, cohens_d=np.nan,
                ))
                continue

            diff = (paired[tB] - paired[tA]).values
            t_stat, p_val = sstats.ttest_rel(paired[tB], paired[tA])
            try:
                w_stat, w_p = sstats.wilcoxon(paired[tB], paired[tA])
            except ValueError:
                w_stat, w_p = np.nan, np.nan

            rows.append(dict(
                hemi=hemi, comparison=label, timepoint_A=tA, timepoint_B=tB,
                n=n,
                mean_A=paired[tA].mean(), sem_A=sstats.sem(paired[tA]),
                mean_B=paired[tB].mean(), sem_B=sstats.sem(paired[tB]),
                mean_diff=np.mean(diff),
                t_stat=t_stat, p_value=p_val,
                wilcoxon_stat=w_stat, wilcoxon_p=w_p,
                cohens_d=_cohens_d_paired(diff),
            ))

    stats_df = pd.DataFrame(rows)

    # FDR (Benjamini-Hochberg) correction across all real paired tests
    test_mask = stats_df["p_value"].notna()
    if test_mask.sum() > 0:
        pvals = stats_df.loc[test_mask, "p_value"].values
        order = np.argsort(pvals)
        ranked = np.empty_like(order)
        ranked[order] = np.arange(1, len(pvals) + 1)
        m = len(pvals)
        fdr = pvals * m / ranked
        # enforce monotonicity
        fdr_sorted = np.minimum.accumulate(fdr[order][::-1])[::-1]
        fdr_final = np.empty_like(fdr)
        fdr_final[order] = np.clip(fdr_sorted, 0, 1)
        stats_df.loc[test_mask, "p_fdr"] = fdr_final
    else:
        stats_df["p_fdr"] = np.nan

    return stats_df


def run_dacc_analysis(dacc_records: list[dict], out_dir: Path):
    """
    Top-level entry point: builds dACC masks, extracts per-subject/session
    mean Fear>Neutral z-values, saves spaghetti plots and a stats CSV.
    """
    print("\n--- dACC longitudinal fear-response analysis (TMS treatment course) ---")
    if not dacc_records:
        print(f"  [WARN] No '{DACC_CONTRAST}' maps found across subjects/sessions — skipping dACC analysis.")
        return

    dacc_dir = out_dir / "dACC_analysis"
    dacc_dir.mkdir(exist_ok=True)
    fig_dir = dacc_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    if not os.path.exists(BN_ATLAS_PATH):
        print(f"  [WARN] Brainnetome atlas not found at {BN_ATLAS_PATH} — skipping dACC analysis.")
        return
    if not os.path.exists(BN_ATLAS_LUT):
        print(f"  [WARN] Brainnetome LUT not found at {BN_ATLAS_LUT} — skipping dACC analysis.")
        return

    target_img = image.load_img(dacc_records[0]["zmap_path"])
    print("  Building dACC ROI masks (A24cd + A32p) from Brainnetome Atlas 246 (2mm) ...")
    masks = build_dacc_masks(BN_ATLAS_PATH, BN_ATLAS_LUT, target_img)
    if not masks:
        print("  [WARN] No dACC masks could be built — skipping dACC analysis.")
        return

    print(f"  Extracting mean '{DACC_CONTRAST}' z-values for {len(dacc_records)} subject/session map(s) ...")
    df_long = collect_dacc_values(dacc_records, masks)
    if df_long.empty:
        print("  [WARN] No dACC values extracted — skipping plots/stats.")
        return

    raw_csv = dacc_dir / "dACC_fear_gt_neutral_values.csv"
    df_long.sort_values(["hemi", "subject", "timepoint"]).to_csv(raw_csv, index=False)
    print(f"  Raw per-subject values → {raw_csv.name}")

    print("  Generating spaghetti plots ...")
    for hemi in ("L", "R", "bilateral"):
        out_path = fig_dir / f"dACC_{hemi}_fear_gt_neutral_spaghetti.png"
        make_dacc_spaghetti_plot(df_long, hemi, out_path)

    print("  Computing group descriptives and timepoint-to-timepoint statistics ...")
    stats_df = compute_dacc_stats(df_long)
    stats_csv = dacc_dir / "dACC_fear_gt_neutral_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  Stats summary → {stats_csv.name}")


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
    dacc_records: list[dict] = []  # {subject, timepoint, zmap_path} for DACC_CONTRAST
    for subj in subjects:
        runs = find_runs(DERIV_DIR, subj)
        print(f"\n{subj}: {len(runs)} task-con run(s)")
        for run in runs:
            z_maps = run_first_level(run, events, fl_dir)
            for cname, zpath in z_maps.items():
                all_z_maps.setdefault(cname, []).append(zpath)
            if DACC_CONTRAST in z_maps:
                dacc_records.append(dict(
                    subject=run["subject"],
                    timepoint=run["session"],
                    zmap_path=z_maps[DACC_CONTRAST],
                ))

    if not all_z_maps:
        raise RuntimeError("No contrast maps were produced. Check paths and file names.")

    # Second-level group analysis
    group_maps = run_second_level(all_z_maps, sl_dir)

    # Glass-brain figures
    save_glass_brains(group_maps, out_dir)

    # Summary report
    write_html_report(subjects, all_z_maps, group_maps, out_dir)

    # dACC longitudinal fear-response analysis (TMS treatment course)
    run_dacc_analysis(dacc_records, out_dir)

    print("\n✓  Analysis complete.")
    print(f"   Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
