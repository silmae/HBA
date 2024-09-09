import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sphinx.cmd.quickstart import valid_dir

from src.data import path_handling as PH
from src import constants as C
from src import plotter
from src.algae import algae_utils as utils

dir_algae = "algae_measurement_sets"
dir_algae_measurement_set = '01_09_2023'
path_algae_measurement_set = PH.join(PH.path_directory_project_root(), dir_algae, dir_algae_measurement_set)

colors = ['r', 'g', 'b', 'c', 'm']
span = 10

sample_numbers_01_09_2023 = {
    '4791': "White before",
    '4792': "Emtpy cuvette trans",
    '4793': "Emtpy cuvette refl",
    '4794': "Broth trans",
    '4795': "Broth refl",
    '4796': "DI water trans",
    '4797': "DI water refl",
    '4798': "Algae 1 trans",
    '4799': "Algae 1 refl",
    '4800': "Algae 2 trans",
    '4801': "Algae 2 refl",
    '4802': "Algae 3 trans",
    '4803': "Algae 3 refl",
    '4804': "Algae 4 trans",
    '4805': "Algae 4 refl",
    '4806': "Algae 5 trans",
    '4807': "Algae 5 refl",
    '4808': "Dark after",
    '4809': "White after",
}

def plot_references(dont_show=True, save_thumbnail=True):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plotter.figsize)
    fig.suptitle(f"References", fontsize=plotter.fig_title_font_size)
    ax[0].set_title('White')
    ax[1].set_title('Cuvette refs')

    wls, val_light_before = utils.read_sample(sample_nr='4791', measurement_set_name=path_algae_measurement_set)
    _, val_light_after = utils.read_sample(sample_nr='4809', measurement_set_name=path_algae_measurement_set)
    _, val_dark = utils.read_sample(sample_nr='4808', measurement_set_name=path_algae_measurement_set)
    _, val_et = utils.read_sample(sample_nr='4792', measurement_set_name=path_algae_measurement_set)
    _, val_er = utils.read_sample(sample_nr='4793', measurement_set_name=path_algae_measurement_set)
    _, val_bt = utils.read_sample(sample_nr='4794', measurement_set_name=path_algae_measurement_set)
    _, val_br = utils.read_sample(sample_nr='4795', measurement_set_name=path_algae_measurement_set)
    _, val_dit = utils.read_sample(sample_nr='4796', measurement_set_name=path_algae_measurement_set)
    _, val_dir = utils.read_sample(sample_nr='4797', measurement_set_name=path_algae_measurement_set)

    ax[0].plot(wls, val_light_before, label=sample_numbers_01_09_2023['4791'])
    ax[0].plot(wls, val_light_after, label=sample_numbers_01_09_2023['4809'])
    ax[0].plot(wls, val_et, label='Empty cuvette transmittance')
    # utils._plot_refl_tran_to_axis(ax[0], refl=val_er, tran=val_et, x_values=wls, x_label="Wavelength [nm]",
    #                               invert_tran=True, label='Empty cuvette', color='orange')

    # Normalize with white plug
    # val_et = val_et / val_light_before
    # val_er = val_er / val_light_before
    # val_bt = val_bt  / val_light_before
    # val_br = val_br  / val_light_before
    # val_dit = val_dit  / val_light_before
    # val_dir = val_dir  / val_light_before

    utils._plot_refl_tran_to_axis(ax[1], refl=val_er, tran=val_et, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Empty cuvette', color='orange')
    utils._plot_refl_tran_to_axis(ax[1], refl=val_br, tran=val_bt, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='Broth', color='brown')
    utils._plot_refl_tran_to_axis(ax[1], refl=val_dir, tran=val_dit, x_values=wls, x_label="Wavelength [nm]", invert_tran=True, label='DI water', color='blue')

    _, val_a1r = utils.read_sample(sample_nr='4799', measurement_set_name=path_algae_measurement_set)
    _, val_a2r = utils.read_sample(sample_nr='4801', measurement_set_name=path_algae_measurement_set)
    _, val_a3r = utils.read_sample(sample_nr='4803', measurement_set_name=path_algae_measurement_set)
    _, val_a4r = utils.read_sample(sample_nr='4805', measurement_set_name=path_algae_measurement_set)
    _, val_a5r = utils.read_sample(sample_nr='4807', measurement_set_name=path_algae_measurement_set)

    # Normalize with white plug
    # val_a1r = val_a1r / val_light_before
    # val_a2r = val_a2r / val_light_before
    # val_a3r = val_a3r / val_light_before
    # val_a4r = val_a4r / val_light_before
    # val_a5r = val_a5r / val_light_before

    val_a1r = utils.smooth_data_np_convolve(arr=val_a1r, span=span)
    val_a2r = utils.smooth_data_np_convolve(arr=val_a2r, span=span)
    val_a3r = utils.smooth_data_np_convolve(arr=val_a3r, span=span)
    val_a4r = utils.smooth_data_np_convolve(arr=val_a4r, span=span)
    val_a5r = utils.smooth_data_np_convolve(arr=val_a5r, span=span)

    ax[1].plot(wls, val_a1r, label="Algae 1", color=colors[0])
    ax[1].plot(wls, val_a2r, label="Algae 2", color=colors[1])
    ax[1].plot(wls, val_a3r, label="Algae 3", color=colors[2])
    ax[1].plot(wls, val_a4r, label="Algae 4", color=colors[3])
    ax[1].plot(wls, val_a5r, label="Algae 5", color=colors[4])

    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)
    ax[1].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)

    ylim = [0,1.0]
    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)

    ax[0].legend()
    ax[1].legend()

    if save_thumbnail:
        save_path_image = PH.join(path_algae_measurement_set, f"references.png")
        plt.savefig(save_path_image, dpi=plotter.save_resolution,bbox_inches='tight', pad_inches=0.1)
    if not dont_show:
        plt.show()


def plot_algae(dont_show=True, save_thumbnail=True, ret_sampl_nr=1):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plotter.figsize)
    # fig.suptitle(f"Algae", fontsize=plotter.fig_title_font_size)
    ax[0].set_title('Transmittance')
    ax[1].set_title('Reflectance')

    wls, val_light_before = utils.read_sample(sample_nr='4791', measurement_set_name=path_algae_measurement_set)
    _, val_light_after = utils.read_sample(sample_nr='4809', measurement_set_name=path_algae_measurement_set)
    _, val_dark = utils.read_sample(sample_nr='4808', measurement_set_name=path_algae_measurement_set)
    _, val_et = utils.read_sample(sample_nr='4792', measurement_set_name=path_algae_measurement_set)
    _, val_er = utils.read_sample(sample_nr='4793', measurement_set_name=path_algae_measurement_set)
    _, val_bt = utils.read_sample(sample_nr='4794', measurement_set_name=path_algae_measurement_set)
    _, val_br = utils.read_sample(sample_nr='4795', measurement_set_name=path_algae_measurement_set)
    _, val_dit = utils.read_sample(sample_nr='4796', measurement_set_name=path_algae_measurement_set)
    _, val_dir = utils.read_sample(sample_nr='4797', measurement_set_name=path_algae_measurement_set)

    _, val_a1t = utils.read_sample(sample_nr='4798', measurement_set_name=path_algae_measurement_set)
    _, val_a1r = utils.read_sample(sample_nr='4799', measurement_set_name=path_algae_measurement_set)
    _, val_a2t = utils.read_sample(sample_nr='4800', measurement_set_name=path_algae_measurement_set)
    _, val_a2r = utils.read_sample(sample_nr='4801', measurement_set_name=path_algae_measurement_set)
    _, val_a3t = utils.read_sample(sample_nr='4802', measurement_set_name=path_algae_measurement_set)
    _, val_a3r = utils.read_sample(sample_nr='4803', measurement_set_name=path_algae_measurement_set)
    _, val_a4t = utils.read_sample(sample_nr='4804', measurement_set_name=path_algae_measurement_set)
    _, val_a4r = utils.read_sample(sample_nr='4805', measurement_set_name=path_algae_measurement_set)
    _, val_a5t = utils.read_sample(sample_nr='4806', measurement_set_name=path_algae_measurement_set)
    _, val_a5r = utils.read_sample(sample_nr='4807', measurement_set_name=path_algae_measurement_set)

    # Spctral range
    low = 400
    high = 700
    val_light_before = utils.range(wls=wls, low=low, high=high, val=val_light_before)
    val_light_after = utils.range(wls=wls, low=low, high=high, val=val_light_after)
    val_dark = utils.range(wls=wls, low=low, high=high, val=val_dark)
    val_et = utils.range(wls=wls, low=low, high=high, val=val_et)
    val_er = utils.range(wls=wls, low=low, high=high, val=val_er)
    val_dit = utils.range(wls=wls, low=low, high=high, val=val_dit)
    val_dir = utils.range(wls=wls, low=low, high=high, val=val_dir)
    val_a1t = utils.range(wls=wls, low=low, high=high, val=val_a1t)
    val_a1r = utils.range(wls=wls, low=low, high=high, val=val_a1r)
    val_a2t = utils.range(wls=wls, low=low, high=high, val=val_a2t)
    val_a2r = utils.range(wls=wls, low=low, high=high, val=val_a2r)
    val_a3t = utils.range(wls=wls, low=low, high=high, val=val_a3t)
    val_a3r = utils.range(wls=wls, low=low, high=high, val=val_a3r)
    val_a4t = utils.range(wls=wls, low=low, high=high, val=val_a4t)
    val_a4r = utils.range(wls=wls, low=low, high=high, val=val_a4r)
    val_a5t = utils.range(wls=wls, low=low, high=high, val=val_a5t)
    val_a5r = utils.range(wls=wls, low=low, high=high, val=val_a5r)

    wls = utils.range(wls=wls, low=low, high=high, val=wls)

    # Normalize to lamp white
    final_ref_t = val_light_before
    # Normalize to lamp white
    final_ref_r = val_light_before

    # Reference
    np.clip(val_et, 0., 1.)
    np.clip(val_er, 0., 1.)

    np.clip(val_a1t, 0., 1.)
    np.clip(val_a1r, 0., 1.)
    np.clip(val_a2t, 0., 1.)
    np.clip(val_a2r, 0., 1.)
    np.clip(val_a3t, 0., 1.)
    np.clip(val_a3r, 0., 1.)
    np.clip(val_a4t, 0., 1.)
    np.clip(val_a4r, 0., 1.)
    np.clip(val_a5t, 0., 1.)
    np.clip(val_a5r, 0., 1.)

    val_a1t = np.clip(val_a1t / final_ref_t, 0., 1.)
    val_a1r = np.clip(val_a1r / final_ref_r, 0., 1.)
    val_a2t = np.clip(val_a2t / final_ref_t, 0., 1.)
    val_a2r = np.clip(val_a2r / final_ref_r, 0., 1.)
    val_a3t = np.clip(val_a3t / final_ref_t, 0., 1.)
    val_a3r = np.clip(val_a3r / final_ref_r, 0., 1.)
    val_a4t = np.clip(val_a4t / final_ref_t, 0., 1.)
    val_a4r = np.clip(val_a4r / final_ref_r, 0., 1.)
    val_a5t = np.clip(val_a5t / final_ref_t, 0., 1.)
    val_a5r = np.clip(val_a5r / final_ref_r, 0., 1.)

    sigma = 5
    val_a1t = utils.smooth_data_np_convolve(arr=val_a1t, span=sigma)
    val_a1r = utils.smooth_data_np_convolve(arr=val_a1r, span=sigma)
    val_a2t = utils.smooth_data_np_convolve(arr=val_a2t, span=sigma)
    val_a2r = utils.smooth_data_np_convolve(arr=val_a2r, span=sigma)
    val_a3t = utils.smooth_data_np_convolve(arr=val_a3t, span=sigma)
    val_a3r = utils.smooth_data_np_convolve(arr=val_a3r, span=sigma)
    val_a4t = utils.smooth_data_np_convolve(arr=val_a4t, span=sigma)
    val_a4r = utils.smooth_data_np_convolve(arr=val_a4r, span=sigma)
    val_a5t = utils.smooth_data_np_convolve(arr=val_a5t, span=sigma)
    val_a5r = utils.smooth_data_np_convolve(arr=val_a5r, span=sigma)

    ax[0].plot(wls, val_a1t, label='Sample 1', color=colors[0])
    ax[0].plot(wls, val_a2t, label='Sample 2', color=colors[1])
    ax[0].plot(wls, val_a3t, label='Sample 3', color=colors[2])
    ax[0].plot(wls, val_a4t, label='Sample 4', color=colors[3])
    ax[0].plot(wls, val_a5t, label='Sample 5', color=colors[4])

    ax[1].plot(wls, val_a1r, label='Sample 1', color=colors[0])
    ax[1].plot(wls, val_a2r, label='Sample 2', color=colors[1])
    ax[1].plot(wls, val_a3r, label='Sample 3', color=colors[2])
    ax[1].plot(wls, val_a4r, label='Sample 4', color=colors[3])
    ax[1].plot(wls, val_a5r, label='Sample 5', color=colors[4])

    x_label = 'Wavelength [nm]'
    ax[0].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)
    ax[1].set_xlabel(x_label, fontsize=plotter.axis_label_font_size)

    ylim = [0,1.0]
    ax[0].set_ylim(ylim)
    ax[1].set_ylim([0,0.1])

    ax[0].legend()
    ax[1].legend()

    if save_thumbnail:
        save_path_image = PH.join(path_algae_measurement_set, f"algae.png")
        plt.savefig(save_path_image, dpi=plotter.save_resolution, bbox_inches='tight', pad_inches=0.1)
    if not dont_show:
        plt.show()

    if ret_sampl_nr == 1:
        return wls, val_a1r, val_a1t
    elif ret_sampl_nr == 2:
        return wls, val_a2r, val_a2t
    elif ret_sampl_nr == 3:
        return wls, val_a3r, val_a3t
    elif ret_sampl_nr == 4:
        return wls, val_a4r, val_a4t
    elif ret_sampl_nr == 5:
        return wls, val_a5r, val_a5t
    else:
        logging.warning(f"Unsupported sample number {ret_sampl_nr}")
        pass

