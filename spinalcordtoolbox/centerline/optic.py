import os
import shutil
import nibabel as nib
import sct_utils as sct
import sct_image
from sct_image import orientation
from msct_image import Image
from sct_convert import convert
from datetime import datetime


def centerline2roi(fname_image, folder_output='', verbose=0):
    """
    Tis method converts a binary centerline image to a .roi centerline file

    Args:
        fname_image: filename of the binary centerline image, in RPI orientation
        folder_output: path to output folder where to copy .roi centerline
        verbose: adjusts the verbosity of the logging.

    Returns: filename of the .roi centerline that has been created

    """
    path_data, file_data, ext_data = sct.extract_fname(fname_image)
    fname_output = file_data + '.roi'

    date_now = datetime.now()
    ROI_TEMPLATE = 'Begin Marker ROI\n' \
                   '  Build version="7.0_33"\n' \
                   '  Annotation=""\n' \
                   '  Colour=0\n' \
                   '  Image source="{fname_segmentation}"\n' \
                   '  Created  "{creation_date}" by Operator ID="SCT"\n' \
                   '  Slice={slice_num}\n' \
                   '  Begin Shape\n' \
                   '    X={position_x}; Y={position_y}\n' \
                   '  End Shape\n' \
                   'End Marker ROI\n'

    im = Image(fname_image)
    nx, ny, nz, nt, px, py, pz, pt = im.dim
    coordinates_centerline = im.getNonZeroCoordinates(sorting='z')

    f = open(fname_output, "w")
    sct.printv('\nWriting ROI file...', verbose)

    for coord in coordinates_centerline:
        coord_phys_center = im.transfo_pix2phys([[(nx - 1) / 2.0, (ny - 1) / 2.0, coord.z]])[0]
        coord_phys = im.transfo_pix2phys([[coord.x, coord.y, coord.z]])[0]
        f.write(ROI_TEMPLATE.format(fname_segmentation=fname_image,
                                    creation_date=date_now.strftime("%d %B %Y %H:%M:%S.%f %Z"),
                                    slice_num=coord.z + 1,
                                    position_x=coord_phys_center[0] - coord_phys[0],
                                    position_y=coord_phys_center[1] - coord_phys[1]))

    f.close()

    if os.path.abspath(folder_output) != os.getcwd():
        shutil.copy(fname_output, folder_output)

    return fname_output


def detect_centerline(image_fname, contrast_type,
                      optic_models_path, folder_output,
                      remove_temp_files=False, init_option=None, output_roi=False, verbose=0):
    """This method will use the OptiC to detect the centerline.

    :param image_fname: The input image filename.
    :param init_option: Axial slice where the propagation starts.
    :param contrast_type: The contrast type.
    :param optic_models_path: The path with the Optic model files.
    :param folder_output: The OptiC output folder.
    :param remove_temp_files: Remove the temporary created files.
    :param verbose: Adjusts the verbosity of the logging.

    :returns: The OptiC output filename.
    """

    image_input = Image(image_fname)
    path_data, file_data, ext_data = sct.extract_fname(image_fname)

    sct.printv('Detecting the spinal cord using OptiC', verbose=verbose)
    image_input_orientation = orientation(image_input, get=True, verbose=False)

    # create temp folder
    path_tmp = sct.create_tmp()
    # temp_folder = sct.TempFolder()

    # copy image to temp folder
    sct.printv('\nCopying input data to tmp folder and convert to nii...', verbose)
    convert(image_fname, path_tmp + 'data.nii')

    # go to temp folder
    path_cwd = os.getcwd()
    os.chdir(path_tmp)

    # define output folder
    # TODO
    path_out = path_cwd

    # convert image data type to int16, as required by opencv (backend in OptiC)
    sct_image.main(args=['-i', 'data.nii', '-type', 'int16', '-o', 'data_int16.nii'])

    # reorient the input image to RPI
    sct_image.main(args=['-i', 'data_int16.nii', '-o', 'data_int16_RPI.nii', '-setorient', 'RPI'])

    # open image
    image_rpi_init = Image('data_int16_RPI.nii')
    nxr, nyr, nzr, ntr, pxr, pyr, pzr, ptr = image_rpi_init.dim
    if init_option is not None:
        if init_option > 1:
            init_option /= (nzr - 1)

    # call the OptiC method to generate the spinal cord centerline
    # optic_input = img_filename
    # optic_filename = img_filename + '_optic'

    os.environ["FSLOUTPUTTYPE"] = "NIFTI_PAIR"
    cmd_optic = 'isct_spine_detect -ctype=dpdt -lambda=1 "%s" "%s" "%s"' % \
                (optic_models_path, 'data_int16_RPI.nii', 'data_int16_RPI_optic')
    sct.run(cmd_optic, verbose=0)

    # convert .img and .hdr files to .nii.gz
    # optic_hdr_filename = img_filename + '_optic_ctr.hdr'
    # centerline_optic_RPI_filename = sct.add_suffix(file_data + ext_data,
    #                                                "_centerline_optic_RPI")
    img = nib.load('data_int16_RPI_optic_ctr.hdr')
    nib.save(img, 'data_int16_RPI_centerline_optic_RPI.nii.gz')

    # reorient the output image to initial orientation
    # centerline_optic_filename = sct.add_suffix(file_data + ext_data, "_centerline_optic")
    # cmd_reorient = 'sct_image -i "%s" -o "%s" -setorient "%s" -v 0' % \
    #                ('data_int16_RPI_centerline_optic_RPI.nii.gz',
    #                 'data_int16_RPI_centerline_optic.nii.gz',
    #                 image_input_orientation)
    # sct.run(cmd_reorient, verbose=0)

    sct_image.main(args=['-i', 'data_int16_RPI_centerline_optic_RPI.nii', '-o', 'data_int16_RPI_centerline_optic.nii', '-setorient', image_input_orientation])

    # copy centerline to parent folder
    # folder_output_from_temp = folder_output
    # if not os.path.isabs(folder_output):
    #     folder_output_from_temp = '../' + folder_output

    sct.printv('Copy output to ' + folder_output, verbose=0)
    shutil.copy('data_int16_RPI_centerline_optic.nii', path_out)

    if output_roi:
        fname_roi_centerline = centerline2roi(fname_image=centerline_optic_RPI_filename,
                                              folder_output=folder_output_from_temp,
                                              verbose=verbose)

        # Note: the .roi file is defined in RPI orientation. To be used, it must be applied on the original image with
        # a RPI orientation. For this reason, this script also outputs the input image in RPI orientation
        shutil.copy(reoriented_image_filename_nii, folder_output_from_temp)

    # return to initial folder
    temp_folder.chdir_undo()

    # delete temporary folder
    sct.remove_folder(path_tmp, remove_temp_files)

    return init_option, os.path.join(folder_output, centerline_optic_filename)
