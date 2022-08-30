"""

Interface for all leaf material related actions.

"""

import numpy as np

import time
import logging

import src.leaf_model.training_data
from src.leaf_model.opt import Optimization
from src.data import file_handling as FH, toml_handling as TH
from src import plotter
from src.leaf_model import nn, surf, leaf_commons as LC


def visualize_leaf_models(show_plot=False):
    """Visualize trained surface and and neural network model against training data.   
    
    The plot is always saved to disk regardless of ``show_plot`` flag.
    
    :param show_plot: 
        If True, show interactive plot. Default is false. 
    """
    
    plotter.plot_trained_leaf_models(show_plot=show_plot, save_thumbnail=True, plot_surf=True, plot_nn=True)


def solve_leaf_material_parameters(set_name: str, resolution=1, solver='nn', clear_old_results=False, nn_name=None,
                                   copyof=None, prospect_parameters: dict = None):
    """Solves leaf material parameters for rendering.
    
    The result is saved to disk: this method does not have a return value.
    
    Note that solvers 'surf' and 'nn' need trained model to work. Pre-trained model are included  
    in the Git repository, but you can train your own using ``train_models()`` method. Solver 'opt' 
    does not need training.
    
    :param set_name: 
        Name of the measurement set.
    :param resolution: 
        Spectral resolution.
    :param solver:
        Solving method either 'opt', 'surf' or 'nn'. Opt is slowest and most accurate (the original method). Surf is 
        fast but not very accurate. NN is fast and fairly accurate. Surf and NN are roughly 200 times faster 
        than opt. Recommended solver is the default 'nn'.  
    :param clear_old_results: 
        If True, clear old results of the set. This is handy for redoing the same set with different method, for 
        example. Note that existing wavelength results are not redone unless first removed.
    :param nn_name: 
        Name of the neural network to use. The default is 'nn_default' but if you have trained your own custom 
        NN, use that name (or rename your NN to 'nn_default.pt'.
    :param copyof: 
        Name of the set to copy. Copies target from existing set (walengths, reflectances, and transmittances).
    :param prospect_parameters: 
        TODO remove and make separate method
    """


    if prospect_parameters is not None:
        logging.info(f"Prospect parameter generation asked but I don't know how yet.")
        # TODO generate new targets and save to disk with given name

    if copyof:
        FH.copy_target(from_set=copyof, to_set=set_name)
    else:
        LC.initialize_directories(set_name=set_name, clear_old_results=clear_old_results)

    ids = FH.list_target_ids(set_name)
    ids.sort()

    if len(ids) < 1:
        raise RuntimeError(f'Could not find any targets for set "{set_name}".')

    for _, sample_id in enumerate(ids):
        FH.create_opt_folder_structure_for_samples(set_name, sample_id)
        logging.info(f'Starting optimization of sample {sample_id}')
        targets = TH.read_target(set_name, sample_id)

        # Spectral resolution
        if resolution != 1:
            targets = targets[::resolution]

        if solver == 'opt':
            o = Optimization(set_name=set_name)
            o.run_optimization(resolution=resolution)
        elif solver == 'surf' or solver == "nn":
            start = time.perf_counter()

            wls = targets[:, 0]
            r_m = targets[:, 1]
            t_m = targets[:, 2]

            if solver == 'surf':
                ad_raw, sd_raw, ai_raw, mf_raw = surf.predict(r_m=r_m, t_m=t_m)
            elif solver == "nn":
                if nn_name:
                    ad_raw, sd_raw, ai_raw, mf_raw = nn.predict(r_m=r_m, t_m=t_m, nn_name=nn_name)
                else:
                    ad_raw, sd_raw, ai_raw, mf_raw = nn.predict(r_m=r_m, t_m=t_m)

            ad, sd, ai, mf = LC._convert_raw_params_to_renderable(ad_raw, sd_raw, ai_raw, mf_raw)
            r, t = LC._material_params_to_RT(set_name, sample_id, wls, ad, sd, ai, mf)

            re = np.abs(r - r_m)
            te = np.abs(t - t_m)
            running_time = (time.perf_counter() - start) / 60.0
            time_process_min = running_time
            time_wall_clock_min = running_time
            sample_result_dict = LC._build_sample_res_dict(wls, r, r_m, re, t, t_m, te, ad_raw, sd_raw, ai_raw, mf_raw, time_process_min, time_wall_clock_min)

            TH.write_sample_result(set_name, sample_result_dict, sample_id)

            plotter.plot_sample_result(set_name, sample_id, dont_show=True, save_thumbnail=True)
        else:
            raise AttributeError(f"Unknown solver '{solver}'. Use one of ['nn','surf','opt'].")

    TH.write_set_result(set_name)
    plotter.plot_set_result(set_name)
    plotter.plot_set_errors(set_name)


def train_models(set_name='training_data', show_plot=False, layer_count=9, layer_width=10, epochs=300, batch_size=2,
                 learning_rate=0.001, patience=30, split=0.1, generate_data=False, train_points_per_dim=100,
                 dry_run=False, train_surf=True, train_nn=True):
    """Train surface model and neural network.
    
    If training data does not yet exist, it must be created by setting ``generate_data=True``. Note that 
    this will take a lot of time as the data generation uses the original optimization method. Depending 
    on value of ``train_points_per_dim`` the generation time varies from tens of minutes to several days. 
    You should generate a few thousand points at least for any accuracy. Models in the repository were 
    trained with 40 000 points (4 days generation time). Use ``dry_run=True`` just to print the number of 
    points that would have been generated.
    
    You can select to train surface model (``train_surf``) and neural network (``train_nn``) separately 
    or just generate the points.
    
    Show plot is safe to be kept at default ``False``. The plots are saved to the disk anyways. 
    
    
    :param set_name:
        Set name of the training data. New training data is generated with this name if  ``generate_data=True``.
        Otherwise, existing data with this name is used.
    :param show_plot: 
        If True, shows interactive plots to user (which halts excecution until window is closed). Regardless 
        of this value, the plots are saved to disk. Default is False.
    :param layer_count: 
        Number of hidden layers in neural network. Omitted if ``train_nn=False``.
    :param layer_width: 
        Width of hidden layers in neural network. Omitted if ``train_nn=False``.
    :param epochs: 
        Maximum number of epochs the neural network is trained. Omitted if ``train_nn=False``.
    :param batch_size: 
        Batch size when training neural network. Omitted if ``train_nn=False``. Smaller values (e.g. 2) yield better 
        accuracy while bigger values (e.g. 32) train faster.
    :param learning_rate: 
        Learning rate of the Adam optimizer. Default value of 0.001 is good and this has very little effect on 
        training results. Feel free to test different values. Omitted if ``train_nn=False``.
    :param patience: 
        Stop NN training if the loss has not improved in this many epochs. Omitted if ``train_nn=False``.
    :param split: 
        Percentage [0,1] of data reserved for testing between epochs. Value between 0.1 and 0.2
        is usually sufficient. Omitted if ``train_nn=False``.
    :param generate_data: 
        If True, new training data is generated with given ``set_name``. Default is False. The training data 
        must exist in order to train the models.
    :param train_points_per_dim: 
         Into how many parts each dimension (R,T) are cut in interval [0,1]. Greater value results in more 
         training points. Good values from 100 to 500. For testing purposes, low values, e.g., 20 can be used.
         Omitted if ``generate_data=False``.
    :param dry_run: 
        Print the number of points that would have been generated, but does not really generate the training points.
        Omitted if ``generate_data=False``.
    :param train_surf: 
        If True, train the surface model. Default is True.
    :param train_nn: 
        If True, train the neural network. Default is True.
    """

    if generate_data:
        src.leaf_model.training_data.generate_train_data(set_name=set_name, cuts_per_dim=train_points_per_dim, dry_run=dry_run)

    if train_surf:
        surf.train(set_name=set_name)
    if train_nn:
        nn.train(show_plot=show_plot, layer_count=layer_count, layer_width=layer_width, epochs=epochs,
                 batch_size=batch_size, learning_rate=learning_rate, patience=patience, split=split, set_name=set_name)
