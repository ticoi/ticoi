# %%
import pandas as pd

from ticoi.cube_data_classxr import cube_data_class
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob


def plot_CVV(Coh_vector, title, vmax=1, mask_glacier=None, save=False):
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    if mask_glacier is not None: Coh_vector = np.ma.masked_where(mask_glacier[::-1] == 0, Coh_vector)
    fig, ax = plt.subplots(figsize=(8, 10))
    pcm = ax.pcolor(Coh_vector, cmap=cmap, vmin=0, vmax=vmax)
    plt.title(f'{title}', fontsize=14, pad=10)
    cbar = plt.colorbar(pcm, orientation='horizontal', pad=0.05)
    cbar.set_label('Coherence of Velocity Vector', fontsize=13)
    if save and mask_glacier:
        plt.savefig(f"{cube.filedir}/{cube.filename.split('.')[0]}CVV_masked_glacier.png")
    elif save:
        plt.savefig(f"{cube.filedir}/{cube.filename.split('.')[0]}CVV_vv150.png")
    plt.show()


def plot_diff_CVV(diff, name_save, mask_good_quality=None, mask_glacier=None, save=False):
    cmap1 = matplotlib.cm.get_cmap('RdYlBu_r')
    if mask_good_quality is not None: diff = np.where(mask_good_quality[::-1] > 1200, diff, np.nan)
    if mask_glacier is not None:
        diff_good_quality_masked = np.ma.masked_where(mask_glacier[::-1] == 0, diff)
    else:
        diff_good_quality_masked = diff
    fig, ax = plt.subplots(figsize=(8, 10))
    # ax.pcolor(diff, cmap=cmap, vmin=-1, vmax=1,alpha=0.7)
    pcm = ax.pcolor(diff_good_quality_masked, cmap=cmap1, vmin=-0.5, vmax=0.5)
    plt.title('Difference between CVV of the resulted and original data', fontsize=14, pad=10)
    cbar = plt.colorbar(pcm, orientation='horizontal', pad=0.05)
    cbar.set_label('Difference between CVV of the resulted and original data', fontsize=13)
    if save: plt.savefig(f"{cube.filedir}/{cube.filename.split('.')[0]}_{name_save}_regu.png")
    plt.show()


# %%
if __name__ == '__main__':
    main_path = '/media/charriel/Elements/Donnees_IGE_Alpes/Test_calcul/cubes_argentieres/arg1/long_baseline_only'
    list_path = glob.glob(f'{main_path}/*.nc')
    # cube_name = '/home/charriel/Documents/Bettik/Yukon/GPS_Kask/GPS_lower_detect_temp_all_period/TICOI_20d_spline_cube_lower_all_bas_correction_shift_130.nc'
    list_VVC, list_param = [], []

    # %% Download cube
    for cube_name in list_path:
        print(cube_name)
        cube = cube_data_class()
        cube.load(cube_name)
        # si on doit selectionner juste une partie du cube:

        # %% Compute NCVV
        Coh_vector = cube.NCVV()

        cmap1 = matplotlib.cm.get_cmap('RdYlBu_r')
        fig, ax = plt.subplots(figsize=(8, 10))
        pcm = ax.pcolor(Coh_vector, cmap=cmap1)
        plt.title('VVC', fontsize=14, pad=10)
        cbar = plt.colorbar(pcm, orientation='horizontal', pad=0.05)
        cbar.set_label('VVC', fontsize=13)
        plt.show()

        list_VVC.append(np.nanmean(Coh_vector))
        param = cube_name.split('_')[-1].split('.nc')[0]
        if param == '1' and cube_name.split('_')[-2].split('.nc')[0] == '0':
            param = 0.1
        list_param.append(int(param))

    dataf = pd.DataFrame({'param': list_param, 'VCC': list_VVC})
    dataf.sort_values(by='param', inplace=True)

    VCC_max99 = dataf['VCC'].max() * 0.99
    VCC_max95 = dataf['VCC'].max() * 0.95
    # list_interval_output=[72,60,24,12,36,48]
    plt.style.use('seaborn-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(dataf['param'], dataf['VCC'], linestyle='-', marker='o', color='orange',
             label=f'Velocity observations')

    ax1.plot([dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['param']], [dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['VCC']],
             linestyle='-', marker='o', color='r',
             label=f'99% of maximal VCC', markersize=5)
    ax1.annotate(f"{dataf[dataf['VCC'] > VCC_max99].iloc[0]['param']}",
                 (dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['param'], dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['VCC']))
    ax1.plot([dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['param']], [dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['VCC']],
             linestyle='-', marker='+', color='r',
             label=f'95% of maximal VCC', markersize=10)
    ax1.annotate(f"{dataf[dataf['VCC'] > VCC_max95].iloc[0]['param']}",
                 (dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['param'], dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['VCC']))

    ax1.set_ylabel('VVC', fontsize=25)
    ax1.set_xlabel('Regularization coefficient', fontsize=25)
    plt.subplots_adjust(bottom=0.25)
    ax1.legend(loc='lower left', bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
    fig1.savefig(
        f'{main_path}/Compa_VVC_coef')
    plt.show()

    diff = pd.DataFrame({})
    diff['diff'] = (np.diff(dataf['VCC']) / np.diff(dataf['param']))
    diff['param'] = dataf['param'].values[1:]

    plt.style.use('seaborn-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(diff['param'], diff['diff'] * 100, linestyle='-', marker='o',
             color='orange',
             label=f'Velocity observations')
    treshold = 0.025
    ax1.plot([diff['param'][diff['diff'] < treshold / 100].iloc[0]],
             [diff['diff'][diff['diff'] < treshold / 100].iloc[0] * 100], linestyle='-', marker='o', color='r',
             label=f'<{treshold}', markersize=5)
    ax1.annotate(f"{diff['param'][diff['diff'] < treshold / 100].iloc[0]}", (
        diff['param'][diff['diff'] < treshold / 100].iloc[0],
        diff['diff'][diff['diff'] < treshold / 100].iloc[0] * 100))

    treshold = 0.01
    ax1.plot([diff['param'][diff['diff'] < treshold / 100].iloc[0]],
             [diff['diff'][diff['diff'] < treshold / 100].iloc[0] * 100], linestyle='-', marker='+', color='r',
             label=f'<{treshold}', markersize=5)
    ax1.annotate(f"{diff['param'][diff['diff'] < treshold / 100].iloc[0]}", (
        diff['param'][diff['diff'] < treshold / 100].iloc[0],
        diff['diff'][diff['diff'] < treshold / 100].iloc[0] * 100))
    ax1.set_ylim(-0.02, 0.25)
    ax1.set_ylabel('VVC derivative', fontsize=25)
    ax1.set_xlabel('Regularization coefficient', fontsize=25)
    plt.subplots_adjust(bottom=0.25)
    ax1.legend(loc='lower left', bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
    fig1.savefig(
        f'{main_path}/Compa_VVC_coef_diff2')
    plt.show()

    list_coef = [0.1, 1, 10, 100, 150, 1000, 5000, 10000]
    dataf = dataf[dataf['param'].isin(list_coef)]

    VCC_max99 = dataf['VCC'].max() * 0.99
    VCC_max95 = dataf['VCC'].max() * 0.95
    # list_interval_output=[72,60,24,12,36,48]
    plt.style.use('seaborn-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(dataf['param'], dataf['VCC'], linestyle='-', marker='o', color='orange',
             label=f'Velocity observations')

    ax1.plot([dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['param']], [dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['VCC']],
             linestyle='-', marker='o', color='r',
             label=f'99% of maximal VCC', markersize=5)
    ax1.annotate(f"{dataf[dataf['VCC'] > VCC_max99].iloc[0]['param']}",
                 (dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['param'], dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['VCC']))
    ax1.plot([dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['param']], [dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['VCC']],
             linestyle='-', marker='+', color='r',
             label=f'95% of maximal VCC', markersize=10)
    ax1.annotate(f"{dataf[dataf['VCC'] > VCC_max95].iloc[0]['param']}",
                 (dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['param'], dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['VCC']))

    ax1.set_ylabel('VVC', fontsize=25)
    ax1.set_xlabel('Regularization coefficient', fontsize=25)
    plt.subplots_adjust(bottom=0.25)
    ax1.legend(loc='lower left', bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
    fig1.savefig(
        f'{main_path}/Compa_VVC_coef_selectedlist')
    plt.show()

    diff = pd.DataFrame({})
    diff['diff'] = (np.diff(dataf['VCC']) / np.diff(dataf['param']))
    diff['param'] = dataf['param'].values[1:]

    plt.style.use('seaborn-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(diff['param'], diff['diff'] * 100, linestyle='-', marker='o',
             color='orange',
             label=f'Velocity observations')
    treshold = 0.025
    ax1.plot([diff['param'][diff['diff'] < treshold / 100].iloc[0]],
             [diff['diff'][diff['diff'] < treshold / 100].iloc[0] * 100], linestyle='-', marker='o', color='r',
             label=f'<{treshold}', markersize=5)
    ax1.annotate(f"{diff['param'][diff['diff'] < treshold / 100].iloc[0]}", (
        diff['param'][diff['diff'] < treshold / 100].iloc[0],
        diff['diff'][diff['diff'] < treshold / 100].iloc[0] * 100))

    treshold = 0.01
    ax1.plot([diff['param'][diff['diff'] < treshold / 100].iloc[0]],
             [diff['diff'][diff['diff'] < treshold / 100].iloc[0] * 100], linestyle='-', marker='+', color='r',
             label=f'<{treshold}', markersize=5)
    ax1.annotate(f"{diff['param'][diff['diff'] < treshold / 100].iloc[0]}", (
        diff['param'][diff['diff'] < treshold / 100].iloc[0],
        diff['diff'][diff['diff'] < treshold / 100].iloc[0] * 100))
    ax1.set_ylim(-0.02, 0.25)
    ax1.set_ylabel('VVC derivative', fontsize=25)
    ax1.set_xlabel('Regularization coefficient', fontsize=25)
    plt.subplots_adjust(bottom=0.25)
    ax1.legend(loc='lower left', bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
    fig1.savefig(
        f'{main_path}/Compa_VVC_coef_diff2_selectedlist')
    plt.show()