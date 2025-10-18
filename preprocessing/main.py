import nibabel as nib
import pathlib
import nibabel.processing
import numpy as np
import pathlib

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

