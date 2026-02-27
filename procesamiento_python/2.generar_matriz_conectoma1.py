"""
Este script implementa un pipeline completo para la construcción de la matriz
de conectividad estructural a partir de datos de difusión (DWI). Se emplea
deconvolución esférica restringida (CSD) y tractografía determinista utilizando
la librería DIPY.

La matriz de conectividad resultante es utilizada posteriormente como entrada
en modelos neuronales y simulaciones de señales M/EEG.
"""

from pathlib import Path
import time
import numpy as np
import nibabel as nib

from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.reconst.csdeconv import auto_response_ssst, ConstrainedSphericalDeconvModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model, DeterministicMaximumDirectionGetter
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines, length as sl_length


def generate_connectome(dwi_file, bval_file, bvec_file, atlas_file, out_dir):
    """
    Genera una matriz de conectividad estructural a partir de datos DWI.

    Parámetros
    ----------
    dwi_file : Path
        Archivo NIfTI con datos de difusión.
    bval_file : Path
        Archivo bvals.
    bvec_file : Path
        Archivo bvecs.
    atlas_file : Path
        Atlas de parcelación alineado al espacio DWI.
    out_dir : Path
        Directorio de salida.
    """

    start_time = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Cargar DWI y gradientes
    # --------------------------------------------------
    dwi_data, dwi_affine = load_nifti(str(dwi_file))
    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    gtab = gradient_table(bvals, bvecs)

    # --------------------------------------------------
    # Máscara cerebral (b0)
    # --------------------------------------------------
    dwi_b0 = dwi_data[..., bvals < 50]
    b0_mean = np.mean(dwi_b0, axis=3)
    _, mask = median_otsu(b0_mean, numpass=2, autocrop=False)

    # --------------------------------------------------
    # Estimación de respuesta del tejido blanco
    # --------------------------------------------------
    response, _ = auto_response_ssst(gtab, dwi_data, fa_thr=0.7)

    # --------------------------------------------------
    # Modelo CSD
    # --------------------------------------------------
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)

    peaks = peaks_from_model(
        model=csd_model,
        data=dwi_data,
        sphere=default_sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=mask,
        return_sh=True,
        normalize_peaks=True
    )

    # --------------------------------------------------
    # Tractografía determinista
    # --------------------------------------------------
    seeds = utils.seeds_from_mask(mask, dwi_affine, density=1.0)
    stopping_criterion = BinaryStoppingCriterion(mask)

    direction_getter = DeterministicMaximumDirectionGetter.from_shcoeff(
        peaks.shm_coeff,
        max_angle=30,
        sphere=default_sphere
    )

    streamline_generator = LocalTracking(
        direction_getter,
        stopping_criterion,
        seeds,
        dwi_affine,
        step_size=0.5
    )

    streamlines = Streamlines(streamline_generator)

    # Filtrado de fibras cortas
    streamlines = Streamlines(
        [sl for sl in streamlines if sl_length(sl) > 10.0]
    )

    # --------------------------------------------------
    # Matriz de conectividad
    # --------------------------------------------------
    atlas_data, atlas_affine = load_nifti(str(atlas_file))
    atlas_data = atlas_data.astype(int)

    connectivity_matrix = utils.connectivity_matrix(
        streamlines,
        label_volume=atlas_data,
        affine=atlas_affine,
        symmetric=True
    )

    np.savetxt(out_dir / "subj_connectome_dipy.csv",
               connectivity_matrix, delimiter=",")

    elapsed = (time.time() - start_time) / 60
    return connectivity_matrix, elapsed


if __name__ == "__main__":

    BASE = Path(r" ")

    generate_connectome(
        dwi_file=BASE / "subj_dwi.nii.gz",
        bval_file=BASE / "subj_dwi.bval",
        bvec_file=BASE / "subj_dwi.bvec",
        atlas_file=BASE / "subj_atlas_fixed.nii.gz",
        out_dir=Path("output")
    )
