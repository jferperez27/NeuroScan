"""
This file will take in the OASIS dataset with all patient information to separate
cognitively normal and AD (Alzheimer's Disease) demented patients. This data
is then saved into a new CSV file for random selection of patients for training.
Ultimately, this new CSV is needed to get the correct MRI files for each patient
using the provided OASIS scripts.

https://github.com/NrgXnat/oasis-scripts

"""

import os
import pandas as pd

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OASIS_DATA_PATH = os.path.join(BASEDIR, "oasis-files", "OASIS3_UDSb4_cdr.csv")
OUTPUT_DIR = os.path.join(BASEDIR, "oasis-files", "processed")

def parse_oasis_data(oasis_csv_path, output_csv_path):
    df = pd.read_csv(oasis_csv_path)

    # Separate cognitively normal and AD patients
    cognitively_normal = df[df['dx1'] == 'Cognitively normal']
    ad_patients = df[df['dx1'] == 'AD Dementia']

    print(f"Total Cognitively Normal Patients: {len(cognitively_normal)}")
    print(f"Total AD Patients: {len(ad_patients)}")

    # Save to new CSV files
    os.makedirs(output_csv_path, exist_ok=True)
    joined = pd.concat([cognitively_normal, ad_patients])
    joined.to_csv(os.path.join(output_csv_path, "parsed_oasis_data.csv"), index=True)


if __name__ == "__main__":
    parse_oasis_data(OASIS_DATA_PATH, OUTPUT_DIR)