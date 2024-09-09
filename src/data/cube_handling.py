
import numpy as np
import spectral
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import logging
import csv

from src.data import path_handling as PH
from src.utils import spectra_utils as SU
from src import constants as C


def construct_envi_cube(forest_id: str, light_max_power):
    """Constructs an ENVI style hyperspectral image cube out of rendered images.

    Can be used after the scene has been rendered (at least spectral and visibility maps).

    White reference for reflectance calculation is searched automatically from
    available visibility maps. Note that the maps must be named like "Reference 0.00 material...".

    Default RGB bands for ENVI metadata are inferred if in visible range.
    Otherwise first, middle, and last bands are used.

    :return:
    """

    p = PH.path_directory_forest_rend_spectral(forest_id=forest_id)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Rend directory for forest '{forest_id}' not found.")

    frame_name_list = os.listdir(p)
    if len(frame_name_list) < 1:
        raise FileNotFoundError(f"No rendered frames were found from '{p}'.")

    frame_list = []
    for thing in frame_name_list:
        file_path = PH.join(p, thing)
        image_as_array = plt.imread(file_path)
        frame_list.append(image_as_array)

    raw_cube = np.array(frame_list)

    # w = raw_cube.shape[1]
    # h = raw_cube.shape[2]
    # plt.plot(raw_cube[:,int(w/2), int(h/2)])
    # plt.show()

    # Burnt areas have values around 65535
    # Loop white references until the image is not burned
    max_burn = 65000.0

    p = PH.path_file_forest_sun_csv(forest_id=forest_id)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Sun csv file '{p}' not found. Try rerunning forest initialization.")


    # Retrieve band and wavelength info from the sun file.
    wls = []
    bands = []
    irradiances = []
    with open(p) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:

            if "band" in row:
                continue # header row

            bands.append(int(row[0]))
            wls.append(float(row[1]))
            irradiances.append(float(row[2]))

    # bands, wls, irradiances = FU.read_csv(p)
    white = np.array(irradiances) * light_max_power

    # white_mean = max_burn

    # Find available reflectance plate reflectivity based on visibility map file names.
    # reflectivities = []
    # map_names = PH.list_reference_visibility_maps(forest_id=forest_id)
    # for map_name in map_names:
    #     splitted = map_name.split(' ')
    #     reflectivity = float(splitted[1])
    #     if reflectivity > 0.0:
    #         reflectivities.append(reflectivity)
    #
    # reflectivities.sort(reverse=True)

    # logging.info(f"Searching for good white reference plate..")
    # accepted_reflectivity = None
    # for reflectivity in reflectivities:
    #     accepted_reflectivity = reflectivity
    #     mask_path = PH.find_reference_visibility_map(forest_id=forest_id, reflectivity=reflectivity)
    #     mask = plt.imread(mask_path)
    #     mask = mask > 0
    #     white_cube = raw_cube[:,mask] # Flattens the reference plate area pixels..
    #     white_mean = np.mean(white_cube, axis=(1)) #.. so we take the mean only on one axis.
    #     white_mean_max = white_mean.max()
    #     if white_mean_max < max_burn:
    #         break

    # logging.info(f"Accepted white reference with {accepted_reflectivity:.2f} reflectivity producing maximum mean reflectance {white_mean_max:.1f}.")

    # Originally loaded as z,x,y, where z is spectral dimension
    raw_cube = np.swapaxes(raw_cube, 0,2) # swap to y,x,z
    raw_cube = np.swapaxes(raw_cube, 0,1) # swap to x,y,z

    # white_mean = np.expand_dims(white_mean, axis=(1,2))
    # reflectance_cube = np.divide(raw_cube, white, dtype=np.float32) # tällee oli valkokorjaus. ei oikee hyvä
    reflectance_cube = raw_cube
    # refl_max = np.max(reflectance_cube)

    # plt.plot(white)
    # plt.show()


    # w = reflectance_cube.shape[0]
    # h = reflectance_cube.shape[1]
    # plt.plot(reflectance_cube[int(w/2), int(h/2),:])
    # plt.show()

    # Swap axis to arrange the array as expected by spectral.envi
    # reflectance_cube = np.swapaxes(reflectance_cube, 0,2)
    # reflectance_cube = np.swapaxes(reflectance_cube, 0,1)

    # p = PH.path_file_forest_sun_csv(forest_id=forest_id)
    # if not os.path.exists(p):
    #     logging.warning(f"Could not find sun data for wavelength info. The image cube will be saved without it.")


    # Define default RGB bands.
    if SU.is_in_visible(wls=wls):
        nearest_R_idx = SU.find_nearest_idx(wls, C.default_R_wl)
        nearest_G_idx = SU.find_nearest_idx(wls, C.default_G_wl)
        nearest_B_idx = SU.find_nearest_idx(wls, C.default_B_wl)
        default_bands = [bands[nearest_R_idx], bands[nearest_G_idx], bands[nearest_B_idx]]
    else:
        default_bands = [bands[-1], bands[int(len(bands)/2)], bands[0]]

    header_dict = {
        "bands" : reflectance_cube.shape[0],
        "lines": reflectance_cube.shape[1],
        "samples": reflectance_cube.shape[2],
        "data_type": 4,
        # "reference reflectivity": accepted_reflectivity,
        "default bands": default_bands,
        "wavelength": wls,
        "wavelength units": "nm",
    }

    cube_dir_path = PH.path_directory_forest_cube(forest_id)
    if not os.path.exists(cube_dir_path):
        os.makedirs(cube_dir_path)

    p_hdr = PH.path_file_forest_reflectance_header(forest_id=forest_id)
    # SPy wants to know only the path to the header. It will find the image file automatically from the same dir.
    spectral.envi.save_image(hdr_file=p_hdr, image=reflectance_cube, dtype=np.float32, force=True, metadata=header_dict)


def show_simulated_cube(forest_id: str, use_SPy=False, rgb_bands=None, override_path=None):
    """Shows the hyperspectral image cube.

    Use construct_envi_cube() to generate it.

    :param forest_id:
        Forest scene id.
    :return:
        None
    :raises
        FileNotFoundError if the cube does not exist.
    """

    if override_path is not None:
        p_cube = override_path
    else:
        p_cube = PH.path_file_forest_reflectance_header(forest_id=forest_id)

    if not os.path.exists(p_cube):
        raise FileNotFoundError(f"Cannot find spectral cube file from '{p_cube}'. "
                                f"Use construct_envi_cube() to generate the cube from rendered images.")
    data = spectral.open_image(p_cube)

    if rgb_bands is None:
        # Minus 1 because spectral is zero-based and ENVI standard one-based.. apparently.
        bands = [int(band) - 1 for band in data.metadata['default bands']]
    else:
        bands = rgb_bands

    if use_SPy:
        # TODO Would be nice if this worked, but it just flashes on the screen
        view = spectral.imshow(data, bands=bands, title=forest_id)
    else:
        rgb = data.read_bands(bands=bands)

        pixel = data.read_pixel(row=int(data.nrows/2), col=int(data.ncols/2))

        # col_list = list(range(data.ncols))
        # line = data.read_subimage(rows=[400], cols=col_list)
        # line = np.squeeze(line, axis=0)
        plt.close('all') # Close all previous plots before showing this one.

        plt.plot(pixel)
        plt.show()

        plt.figure(figsize=(10,10))
        plt.title(forest_id)
        plt.imshow(rgb, norm=colors.LogNorm())
        # plt.colorbar()
        plt.show()


def calc_BR_intensity(forest_id: str, dont_show=True, low_cut=1000, use_high_cut=False):
    """Calculate the pixels of a reactor whose pixel value exceed given low_cut value.

    NOTE requires existing mask images rendered with blender (or hand crafted).

    :param forest_id:
        Name of the scene
    :param dont_show:
        If False, shows plots interactively to the user.
    :param low_cut:
        Low cut value
    :param use_high_cut:
        Exclude pixels that are burnt out for 16-bit unit.
    """

    p_cube = PH.path_file_forest_reflectance_header(forest_id=forest_id)
    p_mask = PH.path_file_visibility_map(forest_id=forest_id, file_name='mask.tif')
    data = spectral.open_image(p_cube)
    mask = plt.imread(p_mask)
    mask = mask / mask.max()
    bands = [5, 24]
    br = data.read_bands(bands=bands)
    b_masked = br[:,:,0] * mask
    r_masked = br[:,:,1] * mask

    # Debug plot for middle pixel spectrum
    # test_spec = data.read_pixel(row=int(data.nrows/2), col=int(data.ncols/2))
    # plt.plot(test_spec)
    # plt.show()

    count_non_zero = np.count_nonzero(mask) # "region of interest"

    if use_high_cut:
        high_cut = 65535
        count_gt_b = np.count_nonzero((low_cut < b_masked) & (b_masked < high_cut))
        count_gt_r = np.count_nonzero((low_cut < r_masked) & (r_masked < high_cut))
    else:
        count_gt_b = np.count_nonzero(low_cut < b_masked)
        count_gt_r = np.count_nonzero(low_cut < r_masked)

    percent_b = (count_gt_b/count_non_zero)*100
    percent_r = (count_gt_r/count_non_zero)*100

    ### Debugging printout ###
    # percent_roi = (count_non_zero/count_all)*100
    # count_all = b_masked.shape[0] * b_masked.shape[1]
    # print(f"Scene: {forest_id}")
    # print(f"\tPixel count: {count_all}, of which non-zero {count_non_zero} ({percent_roi:.1f} %)")
    # print(f"\tBlue pixel count above low limit ({low_cut}) {count_gt_b} ({percent_b:.2f} %)")
    # print(f"\tRed pixel count above low limit ({low_cut}) {count_gt_r} ({percent_r:.2f} %)")

    zeroed = np.zeros_like(br[:,:,0])
    showable = np.stack((br[:,:,1], zeroed, br[:,:,0]))
    showable_masked = showable * mask
    showable_masked = np.swapaxes(showable_masked,2,0)
    showable_masked = showable_masked / np.max(showable_masked)

    if not dont_show:
        plt.imshow(showable_masked)
        plt.show()
        plt.imshow(mask, cmap='gray')
        plt.show()

    return percent_b, percent_r


def inspect_cube(forest_id: str):
    p_cube = PH.path_file_forest_reflectance_header(forest_id=forest_id)
    if not os.path.exists(p_cube):
        raise FileNotFoundError(f"Cannot find spectral cube file from '{p_cube}'. "
                                f"Use construct_envi_cube() to generate the cube from rendered images.")
    data = spectral.open_image(p_cube)

    # Minus 1 because spectral is zero-based and ENVI standard one-based.. apparently.
    default_bands = [int(band) - 1 for band in data.metadata['default bands']]

    view = spectral.imshow(data, bands=default_bands)