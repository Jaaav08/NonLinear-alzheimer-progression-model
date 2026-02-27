"""
Este script corrige un atlas de parcelación cerebral para asegurar su compatibilidad
espacial con datos de difusión (DWI). El procedimiento incluye la conversión a formato 3D,
el re-muestreo al espacio del DWI y la re-etiquetación de regiones cerebrales para obtener
labels consecutivos.

Este paso es necesario para la construcción posterior de matrices de conectividad
estructural y para la simulación de señales M/EEG mediante modelos neuronales.
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import scipy.ndimage as ndimage


def fix_atlas(atlas_in, dwi_file, atlas_out):
    """
    Ajusta un atlas de parcelación al espacio de un volumen DWI.

    Parámetros
    ----------
    atlas_in : Path
        Ruta al archivo NIfTI del atlas original.
    dwi_file : Path
        Ruta al archivo NIfTI del volumen DWI de referencia.
    atlas_out : Path
        Ruta de salida para el atlas corregido.
    """

    # Cargar atlas y DWI
    atlas_img = nib.load(str(atlas_in))
    atlas_data = atlas_img.get_fdata()

    dwi_img = nib.load(str(dwi_file))
    dwi_shape = dwi_img.shape[:3]

    # Convertir atlas 4D a 3D si es necesario
    if atlas_data.ndim == 4:
        atlas_data = atlas_data[..., 0]

    # Re-muestrear atlas al tamaño del DWI
    if atlas_data.shape != dwi_shape:
        zoom_factors = (
            dwi_shape[0] / atlas_data.shape[0],
            dwi_shape[1] / atlas_data.shape[1],
            dwi_shape[2] / atlas_data.shape[2],
        )
        # order=0 para preservar etiquetas discretas
        atlas_data = ndimage.zoom(atlas_data, zoom_factors, order=0)

    # Convertir a enteros
    atlas_data = atlas_data.astype(np.int32)

    # Re-etiquetar regiones para obtener labels consecutivos (0...N)
    labels = np.unique(atlas_data)
    label_mapping = {old: new for new, old in enumerate(labels)}

    atlas_fixed = np.zeros_like(atlas_data, dtype=np.int32)
    for old_label, new_label in label_mapping.items():
        atlas_fixed[atlas_data == old_label] = new_label

    # Guardar atlas corregido
    fixed_img = nib.Nifti1Image(atlas_fixed, dwi_img.affine)
    nib.save(fixed_img, str(atlas_out))


if __name__ == "__main__":

    BASE = Path(r" ")

    ATLAS_IN = BASE / "parcellation_simple.nii.gz"
    DWI_FILE = BASE / "subj_dwi.nii.gz"
    ATLAS_OUT = BASE / "subj_atlas_fixed.nii.gz"

    fix_atlas(ATLAS_IN, DWI_FILE, ATLAS_OUT)
