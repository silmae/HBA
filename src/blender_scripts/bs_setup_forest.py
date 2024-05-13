# bpy stands for Blender Python, which is included Blender's own Python environment.
# It exists inside Blender, so don't worry if your IDE flags it as not found.
import bpy
import os
import sys  # to get command line args
import argparse  # to parse options for us and print a nice help message
import logging
import importlib
import csv

blend_dir = os.path.dirname(os.path.abspath(bpy.data.filepath))

if 'scenes' in blend_dir:
    # We are in a copied blend file in HyperBlend/scenes/scene_12345
    script_dir = os.path.abspath(blend_dir + '../../../src/blender_scripts')
    data_dir = os.path.abspath(blend_dir + '../../../src/data')
    forest_dir = os.path.abspath(blend_dir + '../../../src/forest')
    src_dir = os.path.abspath(blend_dir + '../../../src')
else:
    # We are in the template forest blend file
    script_dir = os.path.abspath(blend_dir + '/src/blender_scripts')
    data_dir = os.path.abspath(blend_dir + '/src/data')
    forest_dir = os.path.abspath(blend_dir + '/src/forest')
    src_dir = os.path.abspath(blend_dir + '/src')

# After this is set, any script in /blender_scripts can be imported
if script_dir not in sys.path:
    sys.path.append(script_dir)
if data_dir not in sys.path:
    sys.path.append(data_dir)
if forest_dir not in sys.path:
    sys.path.append(forest_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import forest_constants as FC
import forest_control as control
import forest_utils as FU
from src.data import file_names as FN
from src.data import path_handling as PH
from src import constants as C

importlib.reload(FC)
importlib.reload(FU)
importlib.reload(FN)
importlib.reload(PH)
importlib.reload(control)
importlib.reload(C)

b_context = bpy.context
b_data = bpy.data
b_ops = bpy.ops
b_scene = b_data.scenes["Scene"]


def set_leaf_material(leaf_material_name, band_list, ad_list, sd_list, ai_list, mf_list):
    """Set leaf material for all frames in band list.

    :param leaf_material_name:
        Must be a name that can be found in Blender forest scene file materials.
    :param band_list:
        List of spectral bands (ints) that correspond to animation frames in Blender.
    :param ad_list:
        List of absorbing particle densities (floats).
    :param sd_list:
        List of scattering particle densities (floats).
    :param ai_list:
        List of scatering anisotropies (floats).
    :param mf_list:
        List of mixing factors (floats).
    """

    def set_leaf_material_parameter_per_frame(material_name, parameter, value, frame):
        material = bpy.data.materials[material_name]

        # bpy.data.materials["Reactor content material"].node_tree.nodes["Group"].inputs[0].default_value

        dp = f'nodes["Group"].inputs[{parameter}].default_value'
        material.node_tree.nodes["Group"].inputs[parameter].default_value = value
        material.node_tree.keyframe_insert(dp, frame=frame)

    for i,band in enumerate(band_list):
        set_leaf_material_parameter_per_frame(leaf_material_name, 0, ad_list[i] * DENSITY, band)
        set_leaf_material_parameter_per_frame(leaf_material_name, 1, sd_list[i] * DENSITY, band)
        set_leaf_material_parameter_per_frame(leaf_material_name, 2, ai_list[i], band)
        set_leaf_material_parameter_per_frame(leaf_material_name, 3, mf_list[i], band)
        # set_leaf_material_parameter_per_frame(leaf_material_name, 'Density scale', DENSITY, band)

    _set_leaf_rgb(leaf_material_name=leaf_material_name)


def _set_leaf_rgb(leaf_material_name: str):
    """Read leaf RGB values from a csv and set them to blend file."""

    # logging.error(f"Setting rgb color for '{leaf_material_name}'")

    p = PH.path_file_forest_rgb_csv(forest_id=forest_id)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Leaf RGB color file '{p}' not found. Have you removed it? Try rerunning forest initialization.")

    with open(p) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            if leaf_material_name in row[0]:
                rgba = (float(row[1]), float(row[2]), float(row[3]), 1.) # must have four items so adding 1 for alpha
                # bpy.data.materials[leaf_material_name].node_tree.nodes["Group"].inputs["RGB color"].default_value = rgba
                bpy.data.materials[leaf_material_name].node_tree.nodes["Group"].inputs[4].default_value = rgba
                print(f"Set RGB: {rgba}")


def read_leaf_material_csv(file_name: str):
    """Read spectral leaf material parameters from a CSV file and return as a bunch of lists.

    :return:
         band_list, wl_list, ad_list, sd_list, ai_list, mf_list
    """

    file_name = FN.filename_leaf_material_csv(file_name.rstrip('.csv'))

    p = PH.join(PH.path_directory_forest_scene(forest_id), file_name)

    if not os.path.exists(p):
        raise FileNotFoundError(f"Leaf csv file '{p}' not found. Check your file names given to setup script.")

    band_list = []
    wl_list = []
    ad_list = []
    sd_list = []
    ai_list = []
    mf_list = []
    with open(p) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            try:
                band_list.append(int(row[0]))
                wl_list.append(float(row[1]))
                ad_list.append(float(row[2]))
                sd_list.append(float(row[3]))
                ai_list.append(float(row[4]))
                mf_list.append(float(row[5]))
            except ValueError:
                # this is ok
                # print(f"Material headers: {row}")
                pass

    return band_list, wl_list, ad_list, sd_list, ai_list, mf_list


def set_animation_frames(band_count):
    """Sets start and end frames to blend file.

    Each wavelength band needs one frame.
    """

    for scene in b_data.scenes:
        scene.render.fps = 5 # does not really matter as we don't do a real animation
        scene.frame_start = 1
        scene.frame_end = band_count # can be safely set to zero as Blender will fix it into one


def insert_leaf_data(leaf_material_names: str):
    """Reads spectral leaf material csv file and sets proper parameters to the Blender file.

    :param leaf_material_names:
        Names of the leaf materials (must mach the ones used in the Blender file) as a
        single string that looks like a list of strings like so: "['Leaf material 1', 'Leaf material 2',...]".
    """

    logging.error(f"Setting leaf material parameters for: '{leaf_material_names}'")

    lmn = [id for id in (leaf_material_names.lstrip('[').rstrip(']')).split(', ')]

    for i, leaf_csv_name in enumerate(lmn):

        leaf_csv_name = leaf_csv_name.strip('\'')
        try:
            band_list, wl_list, ad_list, sd_list, ai_list, mf_list = read_leaf_material_csv(leaf_csv_name)
        except FileNotFoundError as e:
            logging.fatal(f"Could not find file '{leaf_csv_name}'. Blend file will not be setup properly.")
            raise

        # We would only need to do this once, but it's ok if we do it again every round.
        set_animation_frames(len(band_list))

        set_leaf_material(leaf_material_name=leaf_csv_name, band_list=band_list, ad_list=ad_list, sd_list=sd_list, ai_list=ai_list, mf_list=mf_list)


def insert_soil_data():
    """Reads soil data csv and inserts its content into the Blender file.
    """

    logging.error(f"Inserting spectral soil reflectance to Blender material keyframes.")

    p = PH.path_file_forest_soil_csv(forest_id=forest_id)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Soil csv file '{p}' not found. Try rerunning forest initialization.")

    # TODO Should we raise an error or just go without setting soil material??

    bands, _, reflectances = FU.read_csv(p)

    for i,band in enumerate(bands):

        socket_name = 'Spectral reflectivity'
        material = bpy.data.materials["Ground material"]
        dp = f'nodes["Group"].inputs["{socket_name}"].default_value'
        material.node_tree.nodes["Group"].inputs[socket_name].default_value = reflectances[i]
        material.node_tree.keyframe_insert(dp, frame=band)


def insert_trunk_data():
    logging.error(f"insert_trunk_data() called, but I am missing the implementation...")


# def init_cameras():
#
#     cameras = b_data.collections[FC.key_collection_cameras].all_objects
#
#     # All cameras will use field of view instead of focal length.
#     for camera in cameras:
#         camera.lens_unit = "FOV"


if __name__ == '__main__':

    DENSITY = C.density_scale
    """ Nasty global that requires some explaining. This is the density of scattering and 
    absorbing particles in the leaf volume shader. It must be set to the same value that 
    was used when running the leaf simulation. It only needs to be changed if the simulated 
    leaf has different thickness. The default leaf has thickness of 2 mm and density of 3000. 
    If the thickness is changed, the density in leaf simulation can be changed from 
    src/leaf_model/leaf_commons.py.
    """

    """Maximum sun power set to 4 W/m2 so that white does not burn. Can be increased 
    if there is no pure white in the scene."""


    # Store arguments passed from blender_control.py
    argv = sys.argv

    if "--" not in argv:
        argv = []  # no arguments for the script
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    # Argument names
    key_scene_id = ['-id', '--scene_id']
    key_leaf_ids = ['-l_ids', '--leaf_ids']

    key_r_kettle= ['-r_k', '--r_kettle']
    key_kettle_type = ['-k', '--kettle_type']
    key_r_lamp = ['-r_l', '--r_lamp']
    key_n_rings = ['-n_r', '--n_rings']
    key_n_lamp_1 = ['-n_1', '--n_lamp_1']
    key_n_lamp_2 = ['-n_2', '--n_lamp_2']
    key_n_lamp_3 = ['-n_3', '--n_lamp_3']
    key_top_cam_height = ['-tch', '--top_cam_height']
    key_light_max_pow = ['-p', '--light_max_power']

    parser = argparse.ArgumentParser()

    parser.add_argument(key_scene_id[0], key_scene_id[1], dest=key_scene_id[1], action="store", required=True, help="Name of the forest scene.")
    parser.add_argument(key_leaf_ids[0], key_leaf_ids[1], dest=key_leaf_ids[1], action="store",
                        required=False, type=str, help="List of available leaf names as a string.")

    parser.add_argument(key_r_kettle[0], key_r_kettle[1], dest=key_r_kettle[1], action="store", required=False, type=float)
    parser.add_argument(key_kettle_type[0], key_kettle_type[1], dest=key_kettle_type[1], action="store", required=False)
    parser.add_argument(key_r_lamp[0], key_r_lamp[1], dest=key_r_lamp[1], action="store", required=False, type=float)
    parser.add_argument(key_n_rings[0], key_n_rings[1], dest=key_n_rings[1], action="store", required=False, type=int)
    parser.add_argument(key_n_lamp_1[0], key_n_lamp_1[1], dest=key_n_lamp_1[1], action="store", required=False, type=int)
    parser.add_argument(key_n_lamp_2[0], key_n_lamp_2[1], dest=key_n_lamp_2[1], action="store", required=False, type=int)
    parser.add_argument(key_n_lamp_3[0], key_n_lamp_3[1], dest=key_n_lamp_3[1], action="store", required=False, type=int)
    parser.add_argument(key_top_cam_height[0], key_top_cam_height[1], dest=key_top_cam_height[1], action="store", required=False, type=float)
    parser.add_argument(key_light_max_pow[0], key_light_max_pow[1], dest=key_light_max_pow[1], action="store", required=False, type=float)

    args = parser.parse_args(argv)
    args_dict = vars(args)

    forest_id = args_dict[key_scene_id[1]]
    leaf_material_names = args_dict[key_leaf_ids[1]]

    r_kettle = args_dict[key_r_kettle[1]]
    kettle_type = args_dict[key_kettle_type[1]]
    r_lamp = args_dict[key_r_lamp[1]]
    n_rings = args_dict[key_n_rings[1]]
    n_lamp_1 = args_dict[key_n_lamp_1[1]]
    n_lamp_2 = args_dict[key_n_lamp_2[1]]
    n_lamp_3 = args_dict[key_n_lamp_3[1]]
    top_cam_height = args_dict[key_top_cam_height[1]]
    light_max_pow = args_dict[key_light_max_pow[1]]


    # Set up the kettle geometry
    bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_20"] = 1.0 # water level %
    if r_kettle is not None:
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_4"] = r_kettle * 2 # kettle diameter
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_5"] = r_kettle * 2 # kettle height
        bpy.data.objects["Submarine"].location[2] = r_kettle # camera height
    if n_rings is not None:
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_8"] = n_rings # lamp ring count
    if r_lamp is not None:
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_12"] = r_lamp * 2 # lamp diameter
    if n_lamp_1 is not None:
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_27"] = n_lamp_1 # n lamps in innermost ring
    if n_lamp_2 is not None:
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_28"] = n_lamp_2 # n lamps in center ring
    if n_lamp_3 is not None:
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_29"] = n_lamp_3 # n lamps in outermost ring

    bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_6"] = 0.0  # raise lid
    bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_7"] = False  # cutout

    bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_14"] = 1.0 # lamp reach %
    bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_26"] = True # Hide lid

    if kettle_type == 'steel':
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_25"] = True
    elif kettle_type == 'glass':
        bpy.data.objects["Kettle"].modifiers["GeometryNodes"]["Input_25"] = False
    else:
        raise AttributeError(f"Kettle type '{kettle_type}' not recognised")

    bpy.data.scenes["Scene"].cycles.samples = 5000

    bpy.data.objects["Top camera"].location[0] = 0.0
    bpy.data.objects["Top camera"].location[1] = 0.0
    if top_cam_height is not None:
        bpy.data.objects["Top camera"].location[2] = top_cam_height

    logging.error(f"Running scene setup for '{PH.path_directory_forest_scene(forest_id)}'")

    insert_leaf_data(leaf_material_names=leaf_material_names)
    # insert_soil_data()
    # insert_trunk_data()

    FU.set_sun_power_hsi(forest_id=forest_id, light_max_power=light_max_pow)
    # FU.apply_forest_control(forest_id=forest_id)

    # FU.print_materials()

    bpy.ops.wm.save_as_mainfile(filepath=PH.path_file_forest_scene(forest_id))

    # TODO how to disable using User preferences?
