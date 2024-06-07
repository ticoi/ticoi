import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns

class dataframe_data():
    def __init__(self,dataf:pd.DataFrame=pd.DataFrame()):
        self.dataf = dataf
    def set_temporal_baseline_central_date_offset_bar(self):
        delta = self.dataf['date2'] - self.dataf['date1']  # temporal baseline of the observations
        self.dataf['date_cori'] = np.asarray(self.dataf['date1'] + delta // 2).astype('datetime64[D]')  # central date
        self.dataf['temporal_baseline'] = np.asarray((delta).dt.days).astype('int')  # temporal basline as an integer
        self.dataf['offset_bar'] = delta // 2  # to plot the temporal baseline of the plots

    def set_vx_vy_invert(self, conversion:int):
        self.dataf['result_dx'] = self.dataf['result_dx'] / self.dataf['temporal_baseline'] * conversion
        self.dataf['result_dy'] = self.dataf['result_dy'] / self.dataf['temporal_baseline'] * conversion
        self.dataf = self.dataf.rename(columns={'result_dx': 'vx', 'result_dy': 'vy'})

    def set_vv(self):
        self.dataf['vv'] = np.round(np.sqrt((self.dataf['vx'] ** 2 + self.dataf['vy'] ** 2).astype('float')),
                                    2)  # Compute the magnitude of the velocity

    def set_minmax(self):
        self.vxymin = int(self.dataf['vx'].min())
        self.vxymax = int(self.dataf['vx'].max())
        self.vyymin = int(self.dataf['vy'].min())
        self.vyymax = int(self.dataf['vy'].max())
        if 'vv' in self.dataf.columns:
            self.vvymin = int(self.dataf['vv'].min())
            self.vvymax = int(self.dataf['vv'].max())

class pixel_class():

    def __init__(self,save=False,show=False,figsize = (10,6),unit='m/y',path_save='',A=None,dataobs=None):
        self.dataobs = dataobs
        self.datainvert = None
        self.save = save
        self.path_save = path_save
        self.show = show
        self.figsize = figsize
        self.unit = unit
        self.A = A


    def set_obs_data_from_pandas_df(self, dataf_obs, variables = ['vv']):
        self.dataobs = dataframe_data(dataf_obs)
        self.dataobs.set_temporal_baseline_central_date_offset_bar()
        if 'vv' in variables:  self.dataobs.set_vv()
        self.dataobs.set_minmax()

    def set_ilf_results_from_pandas_df(self, dataf_ilf, conversion=365, variables = ['vv']):

        self.datainvert = dataframe_data(dataf_ilf)
        self.datainvert.set_temporal_baseline_central_date_offset_bar()  #set the temporal baseline,
        self.datainvert.set_vx_vy_invert(conversion) #convert displacement in vx and vy
        if 'vv' in variables:  self.datainvert.set_vv()
        self.datainvert.set_minmax()

    def load(self,dataf, type_data = 'obs',dataformat='df',save=False,show=False,figsize = (10,6),unit='m/y',path_save='',variables = ['vv','vx','vy'],A=None):
        self.__init__(save=save,show=show,figsize=figsize,unit=unit,path_save=path_save,A=A)
        if  isinstance(dataf, list): self.load_two_dataset(dataf,dataformat=dataformat,variables=variables)
        else: self.load_one_dataset(dataf,dataformat=dataformat,type_data=type_data,variables=variables)

    def load_one_dataset(self,dataf, type_data = 'obs',dataformat='df',variables = ['vv','vx','vy']):

        conversion = self.get_conversion()
        if type_data == 'obs':
            if dataformat == 'df': self.set_obs_data_from_pandas_df(dataf, variables=variables)
        elif type_data == 'invert':
            if dataformat == 'df': self.set_ilf_results_from_pandas_df(dataf, conversion=conversion, variables=variables)
        else: raise ValueError ('Please enter invert for inverted results and obs for observation')


    def load_two_dataset(self,list_dataf,dataformat='df',variables=['vv','vx','vy']):
        """
        Must be first observations then inverted results
        :param list_dataf:
        :param dataformat:
        :param save:
        :param show:
        :param figsize:
        :param unit:
        :param path_save:
        :return:
        """
        self.load_one_dataset(list_dataf[0], type_data = 'obs',dataformat=dataformat,variables=variables)
        self.load_one_dataset(list_dataf[1], type_data='invert', dataformat=dataformat,variables=variables)

    def get_dataf_invert_or_obs(self,type_data = 'obs'):
        if self.dataobs is None: return self.datainvert
        elif self.datainvert is None: return self.dataobs
        else:
            if type_data == 'invert': return self.datainvert
            else: return self.dataobs

    def get_conversion(self):
        conversion = 365 if self.unit == 'm/y' else 1
        return conversion

    def plot_vx_vy(self,color='blueviolet',type_data='invert'):

        if type_data == 'invert' : label = 'Results from the inversion'
        else: label = 'Observations'
        data = self.get_dataf_invert_or_obs(type_data)

        fig1, ax1 = plt.subplots(2, 1, figsize=self.figsize)
        ax1[0].set_ylim(data.vxymin, data.vxymax)
        ax1[0].plot(data.dataf['date_cori'], data.dataf['vx'], linestyle='', marker='o', markersize=3,
                    color=color)  # Display the vx components
        ax1[0].errorbar(data.dataf['date_cori'], data.dataf['vx'], xerr=data.dataf['offset_bar'], color='b', alpha=0.5, fmt=',', zorder=1)
        ax1[0].set_ylabel(f'Vx [{self.unit}]', fontsize=16)
        ax1[1].set_ylim(data.vyymin, data.vyymax)
        ax1[1].plot(data.dataf['date_cori'], data.dataf['vy'], linestyle='', marker='o', markersize=3, color=color,
                    label=label)  # Display the vx components
        ax1[1].errorbar(data.dataf['date_cori'], data.dataf['vy'], xerr=data.dataf['offset_bar'], color='b', alpha=0.2, fmt=',', zorder=1)
        ax1[1].set_ylabel(f'Vy [{self.unit}]', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax1[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12)
        if self.show: plt.show()
        if self.save:fig1.savefig(f'{self.path_save}vx_vy_{type_data}.png')
        return ax1, fig1

    def plot_vx_vy_overlayed(self,colors = ['blueviolet','orange']):
        show = copy.copy(self.show)
        save = copy.copy(self.save)
        self.show,self.save  = False, False
        ax1,fig1 = self.plot_vx_vy(color=colors[0],type_data='obs')

        self.show,self.save  = show,save

        ax1[0].set_ylim(self.datainvert.vxymin, self.datainvert.vxymax)
        ax1[0].plot(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vx'], linestyle='', marker='o', markersize=3,
                    color=colors[1])  # Display the vx components
        ax1[0].errorbar(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vx'], xerr=self.datainvert.dataf['offset_bar'], color=colors[1], alpha=0.5, fmt=',', zorder=1)
        ax1[1].set_ylim(self.datainvert.vyymin, self.datainvert.vyymax)
        ax1[1].plot(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vy'], linestyle='', marker='o', markersize=3,
                    color=colors[1],
                    label='Results from the inversion')  # Display the vx components
        ax1[1].errorbar(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vy'], xerr=self.datainvert.dataf['offset_bar'],
                        color='b', alpha=0.2, fmt=',', zorder=1)
        ax1[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=14)

        if self.show: plt.show()
        if self.save:fig1.savefig(f'{self.path_save}vx_vy_overlayed.png')


    def plot_vv(self,color='blueviolet',type_data='invert'):

        if type_data == 'invert' : label = 'Results from the inversion'
        else: label = 'Observations'
        data = self.get_dataf_invert_or_obs(type_data)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_ylim(data .vvymin, data .vvymax)
        ax.set_ylabel(f'Velocity magnitude  [{self.unit}]')
        p = ax.plot(data .dataf['date_cori'], data.dataf['vv'], linestyle='', zorder=1, marker='o', lw=0.7, markersize=2, color=color, label=label)
        ax.errorbar(data .dataf['date_cori'], data.dataf['vv'], xerr=data .dataf['offset_bar'], color='b', alpha=0.2, fmt=',', zorder=1)
        plt.subplots_adjust(bottom=0.2)
        ax.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig.transFigure, fontsize=14)
        if self.show: plt.show(block=False)
        if self.save:fig.savefig(f'{self.path_save}vv_{type_data.png}')
        return ax, fig

    def plot_vv_overlayed(self, colors=['blueviolet', 'orange']):
        show = copy.copy(self.show)
        save = copy.copy(self.save)
        self.show, self.save = False, False
        ax, fig = self.plot_vv(color=colors[0], type_data='obs')
        self.show,self.save  = show,save

        ax.set_ylim(self.datainvert.vvymin, self.datainvert.vvymax)
        p = ax.plot(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vv'], linestyle='', zorder=1, marker='o', lw=0.7,
                    markersize=2, color=colors[1], label=f'Results from the inversion')
        ax.errorbar(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vv'], xerr=self.datainvert.dataf['offset_bar'],
                    color=colors[1], alpha=0.2, fmt=',', zorder=1)
        ax.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig.transFigure, fontsize=14)
        if self.show: plt.show()
        if self.save:fig.savefig(f'{self.path_save}vv_overlayed.png')


    def plot_vx_vy_quality(self,cmap='rainbow',type_data='obs'):
        if 'errorx' not in self.dataobs.dataf.columns:
            return ('There is no error, impossible to plot errors')

        if type_data == 'invert' : label = 'Results from the inversion'
        else: label = 'Observations'
        data = self.get_dataf_invert_or_obs(type_data)

        qualityx = data.dataf['errorx']
        qualityy = data.dataf['errory']

        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        ax[0].set_ylabel(f'Velocity x [{self.unit}]')
        scat = ax[0].scatter(data.dataf['date_cori'], data.dataf['vx'], c=qualityx, s=5, cmap=cmap)
        ax[1].set_ylabel(f'Velocity x [{self.unit}]')
        scat = ax[1].scatter(data.dataf['date_cori'], data.dataf['vy'], c=qualityy, s=5, cmap=cmap)
        legend1 = ax[1].legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0, -0.7), ncol=10,
                               title=f'Error [{self.unit}]',fontsize=14)
        ax[1].add_artist(legend1)
        plt.subplots_adjust(bottom=0.20)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.15, 0), bbox_transform=fig.transFigure, fontsize=14)
        if self.show: plt.show(block=False)
        if self.save:fig.savefig(f'{self.path_save}vxvy_quality_bas_{type_data}.png')

    def plot_xcount_vx_vy(self,cmap='rainbow'):
        if self.datainvert is None: return ('You should this function, once a data of inverted have been loaded')
        if 'xcount_x' not in self.datainvert.dataf.columns:
            return ('There is no error, impossible to plot errors')
        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        ax[0].set_ylabel(f'Velocity x [{self.unit}]')
        ax[0].scatter(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vx'], c=self.datainvert.dataf['xcount_x'], s=4,
                             cmap=cmap, label='Y_contribution')
        ax[1].set_ylabel(f'Velocity x [{self.unit}]')
        scat = ax[1].scatter(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vy'], c=self.datainvert.dataf['xcount_y'], s=4,
                             cmap='rainbow', label='Contribution from observations')
        legend1 = ax[1].legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.1, 0), ncol=5,
                               bbox_transform=fig.transFigure,
                               title='Contribution from observations')
        plt.subplots_adjust(bottom=0.2)
        ax[1].add_artist(legend1)
        if self.show: plt.show(block=False)
        if self.save:fig.savefig(f'{self.path_save}X_dates_contribution_vx_vy.png')

    def plot_xcount_vv(self,cmap='rainbow'):
        if self.datainvert is None: return ('You should this function, once a data of inverted have been loaded')
        if 'xcount_x' not in self.datainvert.dataf.columns:
            return ('There is no error, impossible to plot errors')
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_ylabel(f'Velocity magnitude [{self.unit}]', fontsize=16)
        scat = ax.scatter(self.datainvert.dataf['date_cori'], self.datainvert.dataf['vx'], c=(self.datainvert.dataf['xcount_x'] + self.datainvert.dataf['xcount_x']) / 2, s=4, vmin=0,
                          vmax=100,
                          cmap=cmap, label='Contribution from observations')
        legend1 = ax.legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.1, 0), ncol=5,
                            bbox_transform=fig.transFigure,
                            title='Contribution from observations', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax.add_artist(legend1)
        plt.setp(legend1.get_title(), fontsize=16)
        if self.show: plt.show(block=False)
        if self.save:fig.savefig(f'{self.path_save}X_dates_contribution_vv.png')


    def plot_weights_inversion(self):

        ##WEIGHTS USED IN THE FIRST INVERSION
        fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].set_ylabel(f'Vx [{self.unit}]')
        scat1 = ax[0].scatter(self.dataobs.dataf['date_cori'], self.dataobs.dataf['vx'],
                              c=abs(self.dataobs.dataf['weightinix']), s=5, cmap='plasma_r',
                              edgecolors='k', linewidth=0.1)
        ax[1].set_ylabel(f'Vy [{self.unit}]')
        scat2 = ax[1].scatter(self.dataobs.dataf['date_cori'], self.dataobs.dataf['vx'],
                              c=abs(self.dataobs.dataf['weightiniy']), s=5, cmap='plasma_r',
                              edgecolors='k', linewidth=0.1)
        plt.subplots_adjust(bottom=0.3)
        legend1 = ax[1].legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                               bbox_transform=fig.transFigure,
                               ncol=3, title="Weight ini Vx")
        legend2 = ax[1].legend(*scat2.legend_elements(num=5), loc='lower right', bbox_to_anchor=(0.95, 0),
                               bbox_transform=fig.transFigure,
                               ncol=3, title="Weight ini Vy")
        ax[1].add_artist(legend1)
        ax[1].add_artist(legend2)
        if self.show: plt.show(block=False)
        if self.save: fig.savefig(f'{self.path_save}vx_vy_weightini.png')

        ##WEIGHTS USED IN THE LAST INVERSION

        fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].set_ylabel(f'Vx [{self.unit}]')
        scat1 = ax[0].scatter(self.dataobs.dataf['date_cori'], self.dataobs.dataf['vx'],
                              c=abs(self.dataobs.dataf['weightlastx']), s=5, cmap='plasma_r',
                              edgecolors='k', linewidth=0.1)
        ax[1].set_ylabel(f'Vy [{self.unit}]')
        scat2 = ax[1].scatter(self.dataobs.dataf['date_cori'], self.dataobs.dataf['vx'],
                              c=abs(self.dataobs.dataf['weightlasty']), s=5, cmap='plasma_r',
                              edgecolors='k', linewidth=0.1)
        plt.subplots_adjust(bottom=0.3)
        legend1 = ax[1].legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                               bbox_transform=fig.transFigure,
                               ncol=3, title="Weight ini Vx")
        legend2 = ax[1].legend(*scat2.legend_elements(num=5), loc='lower right', bbox_to_anchor=(0.95, 0),
                               bbox_transform=fig.transFigure,
                               ncol=3, title="Weight ini Vy")
        ax[1].add_artist(legend1)
        ax[1].add_artist(legend2)
        if self.show: plt.show(block=False)
        if self.save: fig.savefig(f'{self.path_save}vx_vy_weightlast.png')


    def plot_residuals(self,log_scale=False):
        if self.A is None:
            print ('Please provide A inside load')
            return ('Please provide A inside load')

        self.dataobs.dataf[self.dataobs.dataf['author'] == 'L. Charrier, J. Mouginot, R.Millan, A.Derkacheva'][
            'author'] = 'IGE'
        dataf = self.dataobs.dataf.replace('L. Charrier, J. Mouginot, R.Millan, A.Derkacheva', 'IGE')
        dataf = dataf.replace('S. Leinss, L. Charrier', 'Leinss')

        dataf["abs_residux"] = abs(dataf["residux"])
        dataf["abs_residuy"] = abs(dataf["residuy"])

        dataf = dataf.rename(columns={"author": 'Author'})

        conversion = self.get_conversion()

        ###RECONSTRUCTION PLOT : reconstruct the observation from AX
        Y_reconstruct_x = np.dot(self.A, self.datainvert.dataf['vx'] * self.datainvert.dataf['temporal_baseline'] / conversion) / self.dataobs.dataf['temporal_baseline'] * conversion
        Y_reconstruct_y = np.dot(self.A, self.datainvert.dataf['vy'] * self.datainvert.dataf['temporal_baseline'] / conversion) / self.dataobs.dataf['temporal_baseline'] * conversion

        show = copy.copy(self.show)
        save = copy.copy(self.save)
        self.show, self.save = False, False
        ax,fig = self.plot_vx_vy(type_data='obs')
        self.show,self.save  = show,save

        # fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].plot(self.dataobs.dataf['date_cori'], Y_reconstruct_x, linestyle='', marker='o', color='r', markersize=3,
                    alpha=0.2)  # Display the vx components
        ax[0].errorbar(self.dataobs.dataf['date_cori'], Y_reconstruct_x, xerr=self.dataobs.dataf['offset_bar'], color='r', alpha=0.2, fmt=',', zorder=1)
        ax[0].set_ylabel(f'Vx [{self.unit}]', fontsize=18)
        ax[1].plot(self.dataobs.dataf['date_cori'], Y_reconstruct_y, linestyle='', marker='o', color='r', markersize=3, alpha=0.2,
                    label='Reconstructed Data')  # Display the vy components
        ax[1].errorbar(self.dataobs.dataf['date_cori'], Y_reconstruct_y, xerr=self.dataobs.dataf['offset_bar'], color='r', alpha=0.3, fmt=',', zorder=1)
        ax[1].legend(bbox_to_anchor=(0.55, -0.3), ncol=3, fontsize=15)
        if self.show: plt.show()
        if self.save: fig.savefig(f'{self.path_save}vx_vy_mismatch.png')

        ###RESIDUALS FROM THE LAST INVERSION
        fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].set_ylabel(f'Vx [{self.unit}]')
        scat1 = ax[0].scatter(self.dataobs.dataf['date_cori'], self.dataobs.dataf['vx'], c=abs(self.dataobs.dataf['residux']), s=5, cmap='plasma_r', edgecolors='k',
                              linewidth=0.1)
        ax[1].set_ylabel(f'Vy [{self.unit}]')
        scat2 = ax[1].scatter(self.dataobs.dataf['date_cori'], self.dataobs.dataf['vy'], c=abs(self.dataobs.dataf['residuy']), s=5, cmap='plasma_r', edgecolors='k',
                              linewidth=0.1)
        plt.subplots_adjust(bottom=0.3)
        legend1 = ax[1].legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                               bbox_transform=fig.transFigure,
                               ncol=3, title="Absolute residual Vx")
        legend2 = ax[1].legend(*scat2.legend_elements(num=5), loc='lower right', bbox_to_anchor=(0.95, 0),
                               bbox_transform=fig.transFigure,
                               ncol=3, title="Absolute residual Vy")
        ax[1].add_artist(legend1)
        ax[1].add_artist(legend2)
        if self.show: plt.show(block=False)
        fig.savefig(f'{self.path_save}vx_vy_final_residual.png')

        ###RESIDUALS FOR VX AND VY, ACCORDING TO THE SENSOR
        ax = sns.catplot(data=dataf, x="sensor", y="abs_residux", hue="Author", kind="box")
        ax.set(xlabel='Sensor', ylabel='Absolute residual vx [m/y]')
        if self.show: plt.show()
        plt.savefig(f'{self.path_save}vx_residual_author_abs.png')

        ax = sns.catplot(data=dataf, x="sensor", y="abs_residuy", hue="Author", kind="box")
        ax.set(xlabel='Sensor', ylabel='Absolute residual vy [m/y]')
        if self.show: plt.show()
        plt.savefig(f'{self.path_save}vy_residual_author_abs.png')

        ###RESIDUALS FROM VX AND VY, ACCORDING TO THE AUTHOR
        ax = sns.catplot(data=dataf, x="sensor", y="residux", hue="Author", kind="box")
        ax.set(xlabel='Sensor', ylabel='Residual vx [m/y]')
        if self.show: plt.show()
        plt.savefig(f'{self.path_save}vx_residual_author.png')

        ax = sns.catplot(data=dataf, x="sensor", y="residuy", hue="Author", kind="box")
        ax.set(xlabel='Sensor', ylabel='Residual vy [m/y]')
        if self.show: plt.show()
        plt.savefig(f'{self.path_save}vy_residual_author.png')

        ###RESIDUALS FROM VX AND VY, ACCORDING TO THE QUALITY INDICATOR
        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(dataf[dataf["Author"] == auth]['weightinix'], dataf[dataf["Author"] == auth]['residux'],
                       linestyle='', marker='o', color=color_list[i], markersize=3)
            ax[1].plot(dataf[dataf["Author"] == auth]['weightiniy'], dataf[dataf["Author"] == auth]['residuy'],
                       linestyle='', marker='o', color=color_list[i], markersize=3, label=auth)
        if log_scale:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        ax[0].set_ylabel(f'Residual vx [{self.unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{self.unit}]', fontsize=16)
        ax[1].set_xlabel(f'Quality indicator', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig.transFigure, fontsize=12,
                     ncol=5)
        if self.show: plt.show()
        fig.savefig(f'{self.path_save}residu_qualitylog.png')

        ###RESIDUALS FROM VX AND VY, ACCORDING TO THE TEMPORAL BASELINE
        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(np.array(dataf['temporal_baseline'])[dataf["Author"] == auth] * 2, dataf[dataf["Author"] == auth]['residux'],
                       linestyle='', marker='o', color=color_list[i], markersize=3)
            ax[1].plot(np.array(dataf['temporal_baseline'])[dataf["Author"] == auth] * 2, dataf[dataf["Author"] == auth]['residuy'],
                       linestyle='', marker='o', color=color_list[i], markersize=3, label=auth)
        if log_scale:
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
        ax[0].set_ylabel(f'Residual vx [{self.unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{self.unit}]', fontsize=16)
        ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig.transFigure, fontsize=12,
                     ncol=5)
        if self.show: plt.show()
        fig.savefig(f'{self.path_save}residu_tempbaseline_log.png')
