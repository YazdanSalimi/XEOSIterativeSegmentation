cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)
project(XEOSIterativeSegmentation)

set(EXTENSION_HOMEPAGE "https://github.com/yazdan-salimi/XEOSIterativeSegmentation")
set(EXTENSION_CATEGORY "Quantification")
set(EXTENSION_CONTRIBUTORS "Yazdan Salimi")
set(EXTENSION_DESCRIPTION
  "Iterative PET lesion segmentation using Otsu initialization and plateau-detection thresholding. Loads DICOM directly in 3D Slicer, interactive convergence curve with click-to-set threshold, MIP GIF export, Excel metrics.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(XEOSIterativeSegmentation)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
