import nibabel as nib
import pathlib
import nibabel.processing
import numpy as np
import pathlib
import pandas as pd


def conform_img(img):
    """
    Conform the input NIfTI image to a standard shape and voxel size.

    Restraints:
        out_shape: (128, 128, 128) - for lightweight VRAM usage
        preferred out_shape: (256, 256, 256) - for better resolution, but requires more VRAM
    """
    new_img = nibabel.processing.conform(img, out_shape=(128, 128, 128), voxel_size=(1, 1, 1), order=3)
    return new_img

def squeeze_img(img, file_name):
    """
    Conform an image and add a tensor dimension at the end, then save it to a file.
    """
    conformed_img = conform_img(img)
    conformed_data = conformed_img.get_fdata()
    data_squeezed = np.expand_dims(conformed_data, axis=-1) # tensor dimension
    nib.save(nib.Nifti1Image(np.squeeze(data_squeezed, -1), conformed_img.affine, conformed_img.header), "training_data/" + file_name)

def setup_image(file_path, file_name):
    """
    Load an image from file_path, process it, and save it with file_name.
    """
    pathlib.Path("training_data").mkdir(exist_ok=True)
    img = nib.load(file_path)
    squeeze_img(img, file_name)
    print(f"Processed and saved {file_name}")

def get_training_data(output=True):
    """
    Process all MRI files, return path to training data key CSV.
    """
    parent_dir = pathlib.Path(__file__).resolve().parent.parent
    oasis_data_dir = parent_dir / "oasis-files" / "processed" / "random_sample.csv"

    oasis_data = pd.read_csv(oasis_data_dir)
    key = {}

    basedir = pathlib.Path(__file__).resolve().parent.parent
    mri_dir = basedir / "mri-files"

    for subject in mri_dir.iterdir():
        if subject.is_dir():
            for session in subject.iterdir():
                if session.is_dir():
                    anat_dir = session / "anat"
                    num_files = len(list(anat_dir.glob("*T1w.nii.gz")))
                    if num_files == 0:
                        print(f"No T1w files found for {subject.name} {session.name}")
                    elif num_files == 1:
                        file_name = f"squeezed_{subject.name}_T1w.nii.gz"
                        file_path = list(anat_dir.glob("*T1w.nii.gz"))[0]
                        if output: print(f"Processing {file_path}")
                        setup_image(file_path, file_name)
                        key[subject.name] = file_name
                    else:
                        files = sorted(anat_dir.glob("*T1w.nii.gz"), reverse=True)
                        file_path = files[0]
                        file_name = f"squeezed_{subject.name}_T1w.nii.gz"
                        if output: print(f"Processing {file_path}")
                        setup_image(file_path, file_name)
                        key[subject.name] = file_name
        if output: print(f"-------------------------")

    key_df = pd.DataFrame(list(key.items()), columns=["subject_id", "file_name"])

    key_df["condition"] = "N/A - Placeholder"

    key_len = len(key_df)
    for i in range(key_len):
        row = key_df.iloc[i]
        subject_id = row["subject_id"]

        len_oasis = len(oasis_data)
        for i in range(len_oasis):
            oasis_row = oasis_data.iloc[i]
            if oasis_row["subject_id"] == subject_id.split("-")[1]:
                diagnosis = oasis_row["diagnosis"]
                if diagnosis.lower() == "ad dementia":
                    diagnosis = "AD"
                elif diagnosis.lower() == "cognitively normal":
                    diagnosis = "CN"
                else:
                    print(f"Unrecognized diagnosis {diagnosis} for subject_id {subject_id}")
                key_df.loc[i, "condition"] = diagnosis
                break

    key_csv_path = basedir / "preprocessing" / "training_data" / "training_data_key.csv"
    key_df.to_csv(key_csv_path, index=False)

    return key_csv_path