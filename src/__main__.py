"""
Entry point of the program.

Look for main at the end of the file to recreate algae experiments used in the paper.
"""

import logging
import os.path
import datetime
import math
import time

import os
import numpy as np
import matplotlib.pyplot as plt

import plotter
import src.leaf_model.nn
from src.leaf_model import surf as SM
from src.forest import forest

from src.data import toml_handling as TH, cube_handling as CH, file_names as FN, path_handling as PH
from src.leaf_model import interface as LI
from src import constants as C

from src.rendering import blender_control as BC

# from src.algae import measurement_spec_24_08_23 as algae
from src.algae import measurement_spec_01_09_23 as M
# from src.algae import measurement_spec_11_04_24 as M
from src.utils import data_utils as DU
from src.leaf_model import training_data


def asym_test():
    import numpy as np
    from src.leaf_model import leaf_commons as LC
    from src.leaf_model.opt import Optimization
    from src.utils import data_utils


    results = []
    n = 10
    const = 0.05
    for i in range(4):

        if i == 0:
            run_name = 'const_r_var_t'
            set_name = f"{run_name}_new"
            nn_name = "nn_default"
            old = False
        if i == 1:
            run_name = 'const_t_var_r'
            set_name = f"{run_name}_new"
            nn_name = "nn_default"
            old = False
        if i == 2:
            run_name = 'const_r_var_t'
            set_name = f"{run_name}_old"
            nn_name = "nn_default_old"
            old = True
        if i == 3:
            run_name = 'const_t_var_r'
            set_name = f"{run_name}_old"
            nn_name = "nn_default_old"
            old = True

        if run_name == 'const_r_var_t':
            r_list = np.ones((n,)) * const
            t_list = np.linspace(const, 1-const, num=n, endpoint=True)
            wls = np.arange(n)
        elif run_name == 'const_t_var_r':
            t_list = np.ones((n,)) * const
            r_list = np.linspace(const, 1-const, num=n, endpoint=True)
            wls = np.arange(n)

        data = data_utils.pack_target(wls=wls, refls=r_list, trans=t_list)

        LC.initialize_directories(set_name=set_name, clear_old_results=True)
        TH.write_target(set_name=set_name, data=data)

        # targets = TH.read_target(set_name=set_name, sample_id=0, resampled=False)
        # o = Optimization(set_name=set_name, diffstep=0.01)
        # o.run_optimization(resampled=False, use_threads=True)

        LI.solve_leaf_material_parameters(set_name=set_name, use_dumb_sampling=True, solver='nn', clear_old_results=True,
                                          plot_resampling=False,solver_model_name=nn_name, old=old)
        set_result = TH.read_set_result(set_name=set_name)
        results.append(set_result)

    # plotter.plot_asym_test_result(set_results=results, dont_show=False, save_thumbnail=True)

    print(f"Done {set_name}")


def plot_asym_test():

    results = []
    for i in range(4):

        if i == 0:
            smthng = 'const_r_var_t'
            set_name = f"{smthng}_new"
        if i == 1:
            smthng = 'const_t_var_r'
            set_name = f"{smthng}_new"
        if i == 2:
            smthng = 'const_r_var_t'
            set_name = f"{smthng}_old"
        if i == 3:
            smthng = 'const_t_var_r'
            set_name = f"{smthng}_old"

        set_result = TH.read_set_result(set_name=set_name)
        results.append(set_result)

    plotter.plot_asym_test_result(set_results=results,dont_show=False, save_thumbnail=True)


def iterative_train():

    # Iterative train manually
    set_name_iter_1 = "train_iter_1v4_algae"
    LI.train_models(set_name=set_name_iter_1, generate_data=True, starting_guess_type='curve',
                    train_points_per_dim=30, similarity_rt=0.25, train_surf=True, train_nn=False, data_generation_diff_step=0.01)
    set_name_iter_2 = "train_iter_2_v4_algae"
    surf_model_name = FN.get_surface_model_save_name(set_name_iter_1)
    LI.train_models(set_name=set_name_iter_2, generate_data=True, starting_guess_type='surf',
                    surface_model_name=surf_model_name, similarity_rt=0.5, train_surf=True, train_nn=False,
                    train_points_per_dim=50, data_generation_diff_step=0.001)
    set_name_iter_3 = "train_iter_3_v4_algae"
    surf_model_name = FN.get_surface_model_save_name(set_name_iter_2)
    LI.train_models(set_name=set_name_iter_3, generate_data=True, starting_guess_type='surf',
                    surface_model_name=surf_model_name, similarity_rt=0.75, train_surf=True, train_nn=False,
                    train_points_per_dim=70, data_generation_diff_step=0.001)
    set_name_iter_4 = "train_iter_4_v4_algae"
    surf_model_name = FN.get_surface_model_save_name(set_name_iter_3)
    LI.train_models(set_name=set_name_iter_4, generate_data=True, starting_guess_type='surf',
                    surface_model_name=surf_model_name, similarity_rt=1.0, train_surf=False, train_nn=True,
                    train_points_per_dim=200, dry_run=False, data_generation_diff_step=0.001, show_plot=True, learning_rate=0.0005)



def algae_leaf(set_name, sample_nr):
    """Solve algae parameters as a leaf (hack so no new code needed)."""

    wls, refl, tran = M.plot_algae(save_thumbnail=True, dont_show=True, ret_sampl_nr=sample_nr)
    wls = np.flip(wls)
    refl = np.flip(refl)
    tran = np.flip(tran)
    tran = np.clip(tran,0,1)
    refl = np.clip(refl,0,1)
    data = DU.pack_target(wls=wls,refls=refl,trans=tran)
    TH.write_target(set_name=set_name, data=data)
    LI.solve_leaf_material_parameters(set_name=set_name,use_dumb_sampling=False, resolution=10,
                                      clear_old_results=True, solver='nn')


def make_kettles(light_max_pow=2000, algae_sample_id=1):
    """

    Ensin Wolfram alphaan ratkastavaksi r_s yhtälöstä: r_g^3 - r_s^3 + (r_g^4 / (n * r_s))
    Sitte lasketaan r_l yhtälöstä: r_g^2 / (n * r_s)
    Lopuksi koodissa voidaan tarkistaa että:
    Vl = 2n pi rl^2 rs
    Vs = Vg + Vl
    Vs = 2 pi rs^3
    Al = 4n pi rl rs

    10 l kattila: r_g = 0.1168
    0.1168^3 - x^3 + (0.1168^4/(4x))
    -> yksi positiivinen juuri r_s = 0.12525
    -> r_l = 0.1168^2 / (4*0.12525) = 0.02723002

    Tarkistus:

    Vl = 2n pi rl^2 rs = 8 pi 0.02723002^2 * 0.12525 = 0.00233407
    Vs = Vg + Vl = 0.010 + 0.00233407 = 0.01233407
    Vs = 2 pi rs^3 = 2 pi 0.12525^3 = 0.0123456 OK
    Al = 4n pi rl rs = 16 pi 0.02723002 * 0.12525 = 0.171433 OK

    100 l kattila: r_g = 0.2515
    0.2515^3 - x^3 + (0.2515^4/(4x))
    r_s = 0.269696
    -> r_l = 0.2515^2 / (4*0.269696) = 0.0619492

    1000 l kattila: r_g = 0.5419
    0.5419^3 - x^3 + (0.5419^4/(8x))
    r_s = 0.562822
    -> r_l = 0.5419^2 / (8*0.562822) = 0.0652195

    :return:
    """

    def calc_V(r):
        V = 2 * math.pi * r**3
        return V

    def calc_A(r):
        A = 4 * math.pi * r**2
        return A

    def check_kettle(r_l, r_s, n, V_g, r_g):
        V_l = 2 * n * math.pi * r_l**2 * r_s
        V_s_check = V_g + V_l
        V_s = 2 * math.pi * r_s**3
        A_l = 4 * n * math.pi * r_l * r_s
        A_g = calc_A(r=r_g)
        V_diff = math.fabs(V_s - V_s_check)
        A_diff = math.fabs(A_l - A_g)
        return V_s, V_diff, A_l, A_diff, V_l

    def calc_r_l(r_g, r_s, n):
        r_l = r_g**2 / (n * r_s)
        return r_l

    def do_kettle_thing(target_vol, r_g, r_s, n, n_rings=1, n1=4, n2=4, n3=4, top_cam_height=2.69728):
        V_g = calc_V(r=r_g)
        A_g = calc_A(r=r_g)
        r_l = calc_r_l(r_g=r_g, r_s=r_s,n=n)
        V_s, V_diff, A_l, A_diff, V_l = check_kettle(r_l=r_l, r_s=r_s, n=n, V_g=V_g, r_g=r_g)
        h_g = 2 * r_g
        h_s = 2 * r_s

        print(f"Target volume {target_vol} l or {target_vol / 1000} m^3: ")
        print(f"\tRadius [m]:      glass = {r_g}, steel {r_s}")
        print(f"\tHeight [m]:      glass = {h_g}, steel {h_s}")
        print(f"\tVolume [m^3]:    glass = {V_g}, steel {V_s} (diff {V_diff})")
        print(f"\tLamp count: {n} (c1: {n1}, c2:{n2}, c3 {n3})")
        print(f"\tLamp area [m^2]: glass = {A_g}, steel {A_l} (diff {A_diff})")
        print(f"\tRod lamp volume [m^3]: {V_l}")
        print(f"\tRod lamp radius [m]: {r_l}")

        material_name = "Reactor content material"
        algae_leaf_set_name = f"algae_sample_{algae_sample_id}_lowres_smoothed"
        algae_leaves = [(algae_leaf_set_name, 0, material_name)]
        sun_file_name = 'AP67_spectra_real_2.txt'
        #
        # # Steel kettle
        forest_id = forest.init(leaves=algae_leaves, rng=rng,
                                custom_forest_id=f"reactor_steel_{target_vol}_s{algae_sample_id}",
                                sun_file_name=sun_file_name)
        BC.setup_forest(forest_id=forest_id, leaf_name_list=[material_name], r_kettle=r_s, kettle_type="steel", r_lamp=r_l,
                        n1=n1, n2=n2, n3=n3, n_rings=n_rings, top_cam_height=top_cam_height, light_max_pow=light_max_pow)
        #
        # # Glass kettle
        forest_id = forest.init(leaves=algae_leaves, rng=rng,
                                custom_forest_id=f"reactor_glass_{target_vol}_s{algae_sample_id}",
                                sun_file_name=sun_file_name)
        BC.setup_forest(forest_id=forest_id, leaf_name_list=[material_name], r_kettle=r_g, kettle_type="glass",
                        top_cam_height=top_cam_height,
                        light_max_pow=light_max_pow)

    # Equation for WA
    # r_g^3 - x^3 + (r_g^4 / (n * x))

    # 10 l kettle:
    # 0.1168^3 - x^3 + (0.1168^4 / (6 * x))
    target_vol = 10
    r_g = 0.1168
    r_s = 0.122677  # from Wolfram Alpha
    n = 6
    top_cam_height = 0.617276
    do_kettle_thing(target_vol=target_vol, r_g=r_g, r_s=r_s, n=n, n1=n, n_rings=1, top_cam_height=top_cam_height)

    # 100 l kettle:
    # 0.2515^3 - x^3 + (0.2515^4 / (12 * x))
    target_vol = 100
    r_g = 0.2515
    r_s = 0.25813  # from Wolfram Alpha
    n = 12
    top_cam_height = 1.26728
    do_kettle_thing(target_vol=target_vol, r_g=r_g, r_s=r_s, n=n, n1=4, n2=8, n_rings=2, top_cam_height=top_cam_height)

    # 1000 l kettle:
    # 0.5419^3 - x^3 + (0.5419^4 / (28 * x))
    target_vol = 1000
    r_g = 0.5419
    r_s = 0.548203 # from Wolfram Alpha
    n = 28
    top_cam_height = 2.69728
    do_kettle_thing(target_vol=target_vol, r_g=r_g, r_s=r_s, n=n, n1=4, n2=8, n3=16, n_rings=3, top_cam_height=top_cam_height)


def render_cubes(light_max_pow, algae_sample_id=1):

    start = time.perf_counter()
    scene_id = f"reactor_steel_10_s{algae_sample_id}"
    BC.render_forest(forest_id=scene_id, render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)
    dur = time.perf_counter() - start
    dur_min = dur / 60
    logging.info(f"Rendering cube took {dur_min:.2f} minutes. Scene '{scene_id}'.")

    start = time.perf_counter()
    scene_id = f"reactor_steel_100_s{algae_sample_id}"
    BC.render_forest(forest_id=scene_id, render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)
    dur = time.perf_counter() - start
    dur_min = dur / 60
    logging.info(f"Rendering cube took {dur_min:.2f} minutes. Scene '{scene_id}'.")

    start = time.perf_counter()
    scene_id = f"reactor_steel_1000_s{algae_sample_id}"
    BC.render_forest(forest_id=scene_id, render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)
    dur = time.perf_counter() - start
    dur_min = dur / 60
    logging.info(f"Rendering cube took {dur_min:.2f} minutes. Scene '{scene_id}'.")

    start = time.perf_counter()
    scene_id = f"reactor_glass_10_s{algae_sample_id}"
    BC.render_forest(forest_id=scene_id, render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)
    dur = time.perf_counter() - start
    dur_min = dur / 60
    logging.info(f"Rendering cube took {dur_min:.2f} minutes. Scene '{scene_id}'.")

    start = time.perf_counter()
    scene_id = f"reactor_glass_100_s{algae_sample_id}"
    BC.render_forest(forest_id=scene_id, render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)
    dur = time.perf_counter() - start
    dur_min = dur / 60
    logging.info(f"Rendering cube took {dur_min:.2f} minutes. Scene '{scene_id}'.")

    start = time.perf_counter()
    scene_id = f"reactor_glass_1000_s{algae_sample_id}"
    BC.render_forest(forest_id=scene_id, render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=scene_id, light_max_power=light_max_pow)
    dur = time.perf_counter() - start
    dur_min = dur / 60
    logging.info(f"Rendering cube took {dur_min:.2f} minutes. Scene '{scene_id}'.")


def show_cubes(algae_sample_id=1):

    CH.show_simulated_cube(forest_id=f"reactor_steel_10_s{algae_sample_id}")
    CH.show_simulated_cube(forest_id=f"reactor_steel_100_s{algae_sample_id}")
    CH.show_simulated_cube(forest_id=f"reactor_steel_1000_s{algae_sample_id}")
    CH.show_simulated_cube(forest_id=f"reactor_glass_10_s{algae_sample_id}")
    CH.show_simulated_cube(forest_id=f"reactor_glass_100_s{algae_sample_id}")
    CH.show_simulated_cube(forest_id=f"reactor_glass_1000_s{algae_sample_id}")


def calc_intensities(dont_show=True, low_cut=1000, use_high_cut=False):
    """Calculate the pixels of a reactor whose pixel value exceed given low_cut value.

    NOTE requires existing mask images rendered with blender (or hand crafted).

    :param dont_show:
        If False, shows plots interactively to the user.
    :param low_cut:
        Low cut value
    :param use_high_cut:
        Exclude pixels that are burnt out for 16-bit unit.
    """

    for volume in [10, 100, 1000]:
        row_glass = f"Glass {volume} L\t& "
        row_steel = f"Steel {volume} L\t& "
        for algae_sample_id in [4,1]:
            b_glass, r_glass = CH.calc_BR_intensity(forest_id=f"reactor_glass_{volume}_s{algae_sample_id}",
                                                    dont_show=dont_show, low_cut=low_cut, use_high_cut=use_high_cut)
            b_steel, r_steel = CH.calc_BR_intensity(forest_id=f"reactor_steel_{volume}_s{algae_sample_id}",
                                                    dont_show=dont_show, low_cut=low_cut, use_high_cut=use_high_cut)
            row_glass += f" {b_glass:.1f} & {r_glass:.1f} &"
            row_steel += f" {b_steel:.1f} & {r_steel:.1f} &"

        row_glass = row_glass.rstrip('&')
        row_steel = row_steel.rstrip('&')
        row_glass += " \\\\"
        row_steel += " \\\\"
        print(row_glass)
        print(row_steel)


if __name__ == '__main__':
    # log to stdout instead of stderr for nice coloring
    # logging.basicConfig(stream=sys.stdout, level='INFO')
    path_dir_logs = "../log"
    if not os.path.exists(path_dir_logs):
        os.makedirs(path_dir_logs)

    log_identifier = str(datetime.datetime.now())
    log_identifier = log_identifier.replace(' ', '_')
    log_identifier = log_identifier.replace(':', '')
    log_identifier = log_identifier.replace('.', '')

    log_file_name = f"{log_identifier}.log"
    log_path = PH.join(path_dir_logs, log_file_name)
    logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[
                            logging.FileHandler(log_path, mode='w'),
                            logging.StreamHandler()
                        ])

    """
    The following code runs the experiments and other results shown in the paper. The experiments should be 
    recreatable at least with some tinkering. Retraining and some other less important parts may not work 
    at all anymore. At least not without deep understanding of the current HyperBlend version and some detective 
    work in git history. The code is in sorry state but there's no time to make it neat. I'll hopefully fix this 
    in future iterations of HyperBlend and make it easier to construct experiments without the need to make drastic 
    modifications to the source code. 
    
    -- Kimmo
    """


    #### Retrain NN for algae ####

    # Might work. Running this takes a long time.
    # iterative_train()

    # Visualization of the training outcome. This was used in the paper though perhaps parameters may differ.
    # LI.visualize_leaf_models(training_set_name=set_name_iter_1, show_plot=True, plot_surf=False, plot_nn=False)


    #### Asymmetric test shown in the paper ####

    # Probably doesnt work without some tinkering and fetching the old NN model from git history
    # asym_test()
    # plot_asym_test()


    #### Plot algae measurements ####

    # Plot spectrophotometer measurements of algae cultures. Note that the import of M
    #   dictates which measurement is plotted. All of the data is included in git repo.

    M.plot_references(dont_show=False)
    M.plot_algae(dont_show=False, show_rough=False)


    #### Validation stuff ####

    material_name = "Reactor content material"
    algae_leaf_set_name = f"validation_sample_1_lowres_2" # give a name
    algae_leaf(set_name=algae_leaf_set_name, sample_nr=1) # solve the slab model
    algae_leaves = [(algae_leaf_set_name, 0, material_name)] # collect stuff to be sent to big sim

    # Not in use for algae scenes, but required by the method.
    rng = np.random.default_rng(4321)
    # initialize the big sim scene. Called forest because history.
    forest_id = forest.init(leaves=algae_leaves, rng=rng,
                            custom_forest_id=f"validation_growth_bottle_low_res_3",
                            copy_forest_id='validation_growth_bottle_low_res_2',
                            sun_file_name="AP67_spectra_real_2.txt")

    light_max_pow = 2000 # light pow for validation experiment
    # forest_id = 'validation_growth_bottle_low_res_3' # use this if not run at one run with previous lines
    BC.setup_forest(forest_id=forest_id, leaf_name_list=[material_name], kettle_type="glass", light_max_pow=light_max_pow)
    BC.render_forest(forest_id=forest_id,render_mode='top', light_max_pow=light_max_pow)
    CH.construct_envi_cube(forest_id=forest_id, light_max_power=light_max_pow)

    #### Design experiment ####

    light_max_pow = 50 # light power for design experiment

    # Call algae_leaf() for the ones you want to run the slab simulation on.

    # Construct bioreactor (kettle) scenes of all types. The import M dictates which measured
    #   algae spectra is used and thus which algae_sample_ids can be used. In this example,
    #   1,2,3,4, and 5 are available. In the paper, 1 and 4 were used.
    make_kettles(light_max_pow=light_max_pow, algae_sample_id=1)
    # Render all cubes. Takes some time, so go get a coffee and a good movie.
    render_cubes(light_max_pow=light_max_pow, algae_sample_id=1)

    # The same for sample 4
    make_kettles(light_max_pow=light_max_pow, algae_sample_id=4)
    render_cubes(light_max_pow=light_max_pow, algae_sample_id=4)

    # Should work and show the cubes. However, I recommend inspecting the cubes using
    #   the CubeInspector software available at https://github.com/silmae/cubeinspector
    # show_cubes(algae_sample_id=1)
    # show_cubes(algae_sample_id=4)

    # Calculate the pixels of a reactor whose pixel value exceed given low_cut value.
    # NOTE requires existing mask images rendered with blender (or hand crafted).
    calc_intensities(dont_show=True, low_cut=1000, use_high_cut=False)
