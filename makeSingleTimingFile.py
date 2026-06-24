#!/Users/braveDP/.conda/envs/bin/python

# ========================================
# Generates a single timing file for conscious FACES task which is saved as events.tsv. A spreadsheet with three columns: 
#   1. Onset (time at which stimuli first presented)
#   2. Duration (how long the stimuli was shown)
#   3. Trial_type (Anger, Happy, Neutral, Fear, Disgust, Sad)
#
# Usage:
#   Set the variable inRoot to the path to the directory containing the onset csvs for each trial (e.g. Anger_Onsets.csv)
#   Set the variable outPath to the directory you want to save outputs to
#
#   Run this script using the following terminal command:
#       python makeSingleTimingFile.py
#
# NOTE: To analyze the non-conscious version of the task simply change the inRoot variable to the non-conscious directory
# ========================================

import os, csv, re, sys
import pandas as pd
from pathlib import Path

#set up path to timing CSVs and path to where the events.tsv files wil be saved
inRoot = '/Users/braveDP/Desktop/NEST-A/Faces/Onsets/Con_onsets/'
outPath = '/Users/braveDP/Desktop/NEST-A/Faces/'

path = Path(outPutDir)
path.mkdir(parents=True, exists_ok=True)

# ==========================================
# TASK TIMING PARAMETERS
# ==========================================

STIM_DURATION = 0.5      # 500 ms
ITI = 0.75               # 750 ms
TRIAL_INTERVAL = STIM_DURATION + ITI  # 1.25 s

#how many stimuli are in each block
n_stims = 8

OUTPUT_FILE = outPath+'events.tsv'

events = []

for con in os.listdir(inRoot):
    if con.endswith('.csv'): #skip pesky .DS_STORE and other files
        df = pd.read_csv(inRoot+con)
        conName = con.split('_')[0]
        
        # ==========================================
        # LOAD BLOCK ONSETS
        # ==========================================

        # Ensure sorted by stimulus number
        df = df.sort_values("num").reset_index(drop=True)

        # ==========================================
        # GENERATE EVENTS
        # ==========================================

        for i in range(len(df)):

            block_start_num = int(df.loc[i, "num"])
            block_start_time = float(df.loc[i, "ons"])

            # Create one event per stimulus
            for stim_idx in range(n_stims):
                onset = block_start_time + (stim_idx * TRIAL_INTERVAL)

                events.append({
                    "onset": round(onset, 4),
                    "duration": STIM_DURATION,
                    "trial_type": conName
                })

# ==========================================
# CREATE EVENTS DATAFRAME
# ==========================================

events_df = pd.DataFrame(events)

# Sort by onset
events_df = events_df.sort_values("onset")

# ==========================================
# SAVE AS BIDS EVENTS.TSV
# ==========================================

events_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print(f"Saved {OUTPUT_FILE}")
print()
print(events_df.head(20))