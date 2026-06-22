# XEOS Iterative Segmentation — 3D Slicer Extension

Iterative PET lesion segmentation using plateau-detection thresholding (Pedro method).

## Features
- Load DICOM directly via Slicer's built-in browser
- Otsu-based initial segmentation, editable in Segment Editor
- Iterative threshold search with plateau detection
- Interactive convergence curve — click or drag to override threshold live
- PET-DICOM colormap applied automatically
- Excel export (summary + per-lesion iterations)
- Rotating MIP GIF with 3D-shaded segmentation overlay
- Export PET, CT, and segmentations as NRRD/NIfTI/MHA

## Installation (developer mode)
1. Unzip and place folder somewhere permanent
2. Edit → Application Settings → Modules → Additional module paths → Add the folder
3. Restart Slicer, search "XEOS"

## Author
Yazdan Salimi — salimiyazdan@gmail.com
