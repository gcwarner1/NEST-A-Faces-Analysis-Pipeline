#!/Users/braveDP/.conda/envs/bin/python

# =========================
# Generates individual events files for each condition in the conscious Faces task. Events files will be named condition_events.tsv and have the following 3 columns:
#   1. Onset (time at which stimuli first presented)
#   2. Duration (how long the stimuli was shown)
#   3. Trial_type (Anger, Happy, Neutral, Fear, Disgust, Sad)
#
# Note that the trial type column will be consistent within each tsv because this produces one file for each condition.
#
# Before running be sure to verify that the inRoot variable is set to the path to the directory containing the all of the onset CSVs and that outPath is set to where you want the outputs saved
#
# Usage:
#   python makeTimingFiles.py
#
# NOTE: To analyze the non-conscious version of the task simply change the inRoot variable to the non-conscious directory
# =========================

import os, csv, re, sys
import pandas as pd
from pathlib import Path

#set up path to timing CSVs and path to where the condition_events.tsv files wil be saved
inRoot = '/Users/braveDP/Desktop/NEST-A/Faces/Con_onsets/'
outPath = '/Users/braveDP/Desktop/NEST-A/Faces/Conscious/'

path = Path(outPutDir)
path.mkdir(parents=True, exists_ok=True)

# ==========================================
# TASK TIMING PARAMETERS
# ==========================================

STIM_DURATION = 0.5      # 500 ms
ITI = 0.75               # 750 ms
TRIAL_INTERVAL = STIM_DURATION + ITI  # 1.25 s

for con in os.listdir(inRoot):
    if con.endswith('.csv'): #skip pesky .DS_STORE and other files
        df = pd.read_csv(inRoot+con)
        conName = con.split('_')[0]
        OUTPUT_FILE = outPath+conName+'_events.tsv'
        
        # ==========================================
        # LOAD BLOCK ONSETS
        # ==========================================

        # Ensure sorted by stimulus number
        df = df.sort_values("num").reset_index(drop=True)

        # ==========================================
        # GENERATE EVENTS
        # ==========================================

        events = []

        for i in range(len(df)):

            block_start_num = int(df.loc[i, "num"])
            block_start_time = float(df.loc[i, "ons"])

            #how many stimuli are in each block
            n_stims =  8

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