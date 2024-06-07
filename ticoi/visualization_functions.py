import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

class dataframe_data():
    def __init__(self,dataf=pd.DataFrame()):
        self.dataf = dataf
    def set_temporal_baseline_central_date_offset_bar(self):
        delta = self.dataf['date2'] - self.dataf['date1']  # temporal baseline of the observations
        self.dataf['date_cori'] = np.asarray(self.dataf['date1'] + delta // 2).astype('datetime64[D]')  # central date
        self.dataf['temporal_baseline'] = np.asarray((delta).dt.days).astype('int')  # temporal basline as an integer
        self.dataf['offset_bar'] = delta // 2  # to plot the temporal baseline of the plots

    def set_vx_vy_invert(self, conversion):
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

    def __init__(self,save=False,show=False,figsize = (10,6),unit='m/y',path_save='',A=None):
        self.save = save
        self.path_save = path_save
        self.show = show
        self.figsize = figsize
        self.unit = unit
        self.A = A


    def load_obs_data_from_pandas_df(self, dataf_obs,variables = ['vv'],pos=1):
        self.data1 = dataframe_data(dataf_obs)
        self.data1.set_temporal_baseline_central_date_offset_bar()
        if 'vv' in variables:  self.data1.set_vv()
        self.data1.set_minmax()

    def load_ilf_results_from_pandas_df(self,dataf_ilf,conversion=365,variables = ['vv'],pos=1):
        if pos == 2:
            self.data2 = dataframe_data(dataf_ilf)
            dataftemp = self.data2
        else:
            self.data1 = dataframe_data(dataf_ilf)
            dataftemp = self.data1

        dataftemp.set_temporal_baseline_central_date_offset_bar()  #set the temporal baseline,
        dataftemp.set_vx_vy_invert(conversion) #convert displacement in vx and vy
        if 'vv' in variables:  dataftemp.set_vv()
        dataftemp.set_minmax()
        if pos == 2: self.data2 = dataftemp
        else: self.data1 = dataftemp

    def load(self,dataf, type_data = 'obs',dataformat='df',save=False,show=False,figsize = (10,6),unit='m/y',path_save='',variables = ['vv'],pos=1,A=None):

        conversion = 365 if unit=='m/y' else 1
        self.__init__(save=save,show=show,figsize=figsize,unit=unit,path_save=path_save,A=A)
        if type_data == 'obs':
            if dataformat == 'df': self.load_obs_data_from_pandas_df(dataf,variables=variables)
        elif type_data == 'invert':
            if dataformat == 'df': self.load_ilf_results_from_pandas_df(dataf,conversion=conversion,variables=variables,pos=pos)
        else: raise ValueError ('Please enter invert for inverted results and obs for observation')


    def load_two_dataset(self,list_dataf,dataformat='df',save=False,show=False,figsize = (10,6),unit='m/y',path_save=''):
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
        self.__init__(save=save,show=show,figsize=figsize,unit=unit,path_save=path_save)
        self.load(list_dataf[0], type_data = 'obs',dataformat=dataformat,save=save,show=show,figsize = figsize,unit=unit,path_save=path_save)
        self.load(list_dataf[1], type_data='invert', dataformat=dataformat,save=save,show=show,figsize = figsize,unit=unit,path_save=path_save,pos=2)

    def plot_vx_vy(self,color='blueviolet',type_data='invert'):

        if type_data == 'invert' : label = 'Results from the inversion'
        else: label = 'Observations'

        fig1, ax1 = plt.subplots(2, 1, figsize=self.figsize)
        ax1[0].set_ylim(self.data1.vxymin,self.data1.vxymax)
        ax1[0].plot(self.data1.dataf['date_cori'], self.data1.dataf['vx'], linestyle='', marker='o', markersize=3,
                    color=color)  # Display the vx components
        ax1[0].errorbar(self.data1.dataf['date_cori'], self.data1.dataf['vx'], xerr=self.data1.dataf['offset_bar'], color='b', alpha=0.5, fmt=',', zorder=1)
        ax1[0].set_ylabel(f'Vx [{self.unit}]', fontsize=16)
        ax1[1].set_ylim(self.data1.vyymin,self.data1.vyymax)
        ax1[1].plot(self.data1.dataf['date_cori'], self.data1.dataf['vy'], linestyle='', marker='o', markersize=3, color=color,
                    label=label)  # Display the vx components
        ax1[1].errorbar(self.data1.dataf['date_cori'], self.data1.dataf['vy'], xerr=self.data1.dataf['offset_bar'], color='b', alpha=0.2, fmt=',', zorder=1)
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

        ax1[0].set_ylim(self.data2.vxymin, self.data2.vxymax)
        ax1[0].plot(self.data2.dataf['date_cori'], self.data2.dataf['vx'], linestyle='', marker='o', markersize=3,
                    color=colors[1])  # Display the vx components
        ax1[0].errorbar(self.data2.dataf['date_cori'], self.data2.dataf['vx'], xerr=self.data2.dataf['offset_bar'], color=colors[1], alpha=0.5, fmt=',', zorder=1)
        ax1[1].set_ylim(self.data2.vyymin, self.data2.vyymax)
        ax1[1].plot(self.data2.dataf['date_cori'], self.data2.dataf['vy'], linestyle='', marker='o', markersize=3,
                    color=colors[1],
                    label='Results from the inversion')  # Display the vx components
        ax1[1].errorbar(self.data2.dataf['date_cori'], self.data2.dataf['vy'], xerr=self.data2.dataf['offset_bar'],
                        color='b', alpha=0.2, fmt=',', zorder=1)
        ax1[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=14)

        if self.show: plt.show()
        if self.save:fig1.savefig(f'{self.path_save}vx_vy_overlayed.png')


    def plot_vv(self,color='blueviolet',type_data='invert'):
        if type_data == 'invert' : label = 'Results from the inversion'
        else: label = 'Observations'
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_ylim(self.data1.vvymin, self.data1.vvymax)
        ax.set_ylabel(f'Velocity magnitude  [{self.unit}]')
        p = ax.plot(self.data1.dataf['date_cori'], self.data1.dataf['vv'], linestyle='', zorder=1, marker='o', lw=0.7, markersize=2,color=color,label=label)
        ax.errorbar(self.data1.dataf['date_cori'], self.data1.dataf['vv'], xerr=self.data1.dataf['offset_bar'], color='b', alpha=0.2, fmt=',', zorder=1)
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

        ax.set_ylim(self.data2.vvymin, self.data2.vvymax)
        p = ax.plot(self.data2.dataf['date_cori'], self.data2.dataf['vv'], linestyle='', zorder=1, marker='o', lw=0.7,
                    markersize=2, color=colors[1], label=f'Results from the inversion')
        ax.errorbar(self.data2.dataf['date_cori'], self.data2.dataf['vv'], xerr=self.data2.dataf['offset_bar'],
                    color=colors[1], alpha=0.2, fmt=',', zorder=1)
        ax.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig.transFigure, fontsize=12)
        if self.show: plt.show()
        if self.save:fig.savefig(f'{self.path_save}vv_overlayed.png')


    def plot_vx_vy_quality(self,cmap='rainbow',type_data='invert'):
        if 'errorx' not in self.data1.dataf.columns:
            return ('There is no error, impossible to plot errors')
        qualityx = self.data1.dataf['errorx']
        qualityy = self.data1.dataf['errory']

        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        ax[0].set_ylabel(f'Velocity x [{self.unit}]')
        scat = ax[0].scatter(self.data1.dataf['date_cori'], self.data1.dataf['vx'], c=qualityx, s=5, cmap=cmap, label=f'Error [{self.unit}]')
        ax[1].set_ylabel(f'Velocity x [{self.unit}]')
        scat = ax[1].scatter(self.data1.dataf['date_cori'], self.data1.dataf['vy'], c=qualityy, s=5, cmap=cmap, label=f'Error [{self.unit}]')
        legend1 = ax[1].legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0, -0.7), ncol=10,
                               title='Confidence')
        ax[1].add_artist(legend1)
        plt.subplots_adjust(bottom=0.20)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.15, 0), bbox_transform=fig.transFigure, fontsize=12)
        if self.show: plt.show(block=False)
        if self.save:fig.savefig(f'{self.path_save}vxvy_quality_bas_{type_data}.png')

    def plot_xcount_vx_vy(self,data,cmap='rainbow'):
        if 'xcount_x' not in self.data.dataf.columns:
            return ('There is no error, impossible to plot errors')
        fig, ax = plt.subplots(2, 1, figsize=self.figsize)
        ax[0].set_ylabel(f'Velocity x [{self.unit}]')
        ax[0].scatter(data.dataf['date_cori'], data.dataf['vx'], c=data.dataf['xcount_x'], s=4,
                             cmap=cmap, label='Y_contribution')
        ax[1].set_ylabel(f'Velocity x [{self.unit}]')
        scat = ax[1].scatter(data.dataf['date_cori'], data.dataf['vy'], c=data.dataf['xcount_y'], s=4,
                             cmap='rainbow', label='Contribution from observations')
        legend1 = ax[1].legend(*scat.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.1, 0), ncol=5,
                               bbox_transform=fig.transFigure,
                               title='Contribution from observations')
        plt.subplots_adjust(bottom=0.2)
        ax[1].add_artist(legend1)
        if self.show: plt.show(block=False)
        if self.save:fig.savefig(f'{self.path_save}X_dates_contribution_vx_vy.png')

    def plot_xcount_vv(self,data,cmap='rainbow'):
        if 'xcount_x' not in self.data.dataf.columns:
            return ('There is no error, impossible to plot errors')
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_ylabel(f'Velocity magnitude [{self.unit}]', fontsize=16)
        scat = ax.scatter(data.dataf['date_cori'], data.dataf['vx'], c=(data.dataf['xcount_x'] + data.dataf['xcount_x']) / 2, s=4, vmin=0,
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

    def plot_residuals(self):
        if self.A is None:
            return ('Please provide A inside load')

        # Reconstruct the observation velocity according to A and the estimated velocities
        Y_reconstruct_x = np.dot(A, result['result_vx'] * delta_r / conversion) / delta * conversion
        Y_reconstruct_y = np.dot(A, result['result_vy'] * delta_r / conversion) / delta * conversion

        # Velocity observation compared to reconstructed velocity from the inversion
        fig1, ax1 = plt.subplots(2, 1, figsize=figsize)
        ax1[0].plot(date_cori, dataf['vx'], linestyle='', marker='o', color='b', markersize=3,
                    alpha=0.3)  # Display the vx components
        ax1[0].errorbar(date_cori, dataf['vx'], xerr=offset_bar, color='b', fmt=',', zorder=1, alpha=0.5)
        ax1[0].plot(date_cori, Y_reconstruct_x, linestyle='', marker='o', color='r', markersize=3,
                    alpha=0.2)  # Display the vx components
        ax1[0].errorbar(date_cori, Y_reconstruct_x, xerr=offset_bar, color='r', alpha=0.2, fmt=',', zorder=1)
        ax1[0].set_ylabel(f'Vx [{unit}]', fontsize=18)
        ax1[1].plot(date_cori, dataf['vy'], linestyle='', marker='o', color='b', markersize=3, alpha=0.3,
                    label='Original data')  # Display the vy components
        ax1[1].errorbar(date_cori, dataf['vy'], xerr=offset_bar, color='b', fmt=',', zorder=1, alpha=0.3)
        ax1[1].plot(date_cori, Y_reconstruct_y, linestyle='', marker='o', color='r', markersize=3, alpha=0.2,
                    label='Reconstructed Data')  # Display the vy components
        ax1[1].errorbar(date_cori, Y_reconstruct_y, xerr=offset_bar, color='r', alpha=0.3, fmt=',', zorder=1)
        ax1[1].set_ylabel(f'Vy [{unit}]', fontsize=18)
        ax1[1].legend(bbox_to_anchor=(0.55, -0.3), ncol=3, fontsize=15)
        if show: plt.show()
        fig1.savefig(f'{path_save}vx_vy_mismatch.png')

        # Plot the residual from the last inversion
        Final_residux = dataf['residux']
        Final_residuy = dataf['residuy']

        fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].set_ylabel(f'Vx [{unit}]')
        scat1 = ax[0].scatter(date_cori, dataf['vx'], c=abs(Final_residux), s=5, cmap='plasma_r', edgecolors='k',
                              linewidth=0.1)
        ax[1].set_ylabel(f'Vy [{unit}]')
        scat2 = ax[1].scatter(date_cori, dataf['vy'], c=abs(Final_residuy), s=5, cmap='plasma_r', edgecolors='k',
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
        if show: plt.show(block=False)
        fig.savefig(f'{path_save}vx_vy_final_residual.png')

        # Plot the first weight used in the inversion
        fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].set_ylabel(f'Vx [{unit}]')
        scat1 = ax[0].scatter(date_cori, dataf['vx'], c=abs(dataf['weightinix']), s=5, cmap='plasma_r',
                              edgecolors='k', linewidth=0.1)
        ax[1].set_ylabel(f'Vy [{unit}]')
        scat2 = ax[1].scatter(date_cori, dataf['vy'], c=abs(dataf['weightiniy']), s=5, cmap='plasma_r',
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
        if show: plt.show(block=False)
        fig.savefig(f'{path_save}vx_vy_weightini.png')

        # Plot the last weight of the inversion
        fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        ax[0].set_ylabel(f'Vx [{unit}]')
        scat1 = ax[0].scatter(date_cori, dataf['vx'], c=abs(dataf['weightlastx']), s=5, cmap='plasma_r',
                              edgecolors='k', linewidth=0.1)
        ax[1].set_ylabel(f'Vy [{unit}]')
        scat2 = ax[1].scatter(date_cori, dataf['vy'], c=abs(dataf['weightlasty']), s=5, cmap='plasma_r',
                              edgecolors='k', linewidth=0.1)
        plt.subplots_adjust(bottom=0.3)
        legend1 = ax[1].legend(*scat1.legend_elements(num=5), loc='lower left', bbox_to_anchor=(0.05, 0),
                               bbox_transform=fig.transFigure,
                               ncol=3, title="Last weight Vx")
        legend2 = ax[1].legend(*scat2.legend_elements(num=5), loc='lower right', bbox_to_anchor=(0.95, 0),
                               bbox_transform=fig.transFigure,
                               ncol=3, title="Last weight Vy")
        ax[1].add_artist(legend1)
        ax[1].add_artist(legend2)
        if show: plt.show(block=False)
        fig.savefig(f'{path_save}vx_vy_weightlast.png')

        # Comparison between authors and sensors of the residual
        dataf[dataf['author'] == 'L. Charrier, J. Mouginot, R.Millan, A.Derkacheva']['author'] = 'IGE'
        dataf = dataf.replace('L. Charrier, J. Mouginot, R.Millan, A.Derkacheva', 'IGE')
        dataf = dataf.replace('S. Leinss, L. Charrier', 'Leinss')

        dataf["abs_residux"] = abs(dataf["residux"])
        dataf["abs_residuy"] = abs(dataf["residuy"])

        dataf = dataf.rename(columns={"author": 'Author'})
        dataf.to_csv(f'{path_save}dataf.csv')

        ax = sns.catplot(data=dataf, x="sensor", y="abs_residux", hue="Author", kind="box")
        ax.set(xlabel='Sensor', ylabel='Absolute residual vx [m/y]')
        if show: plt.show()
        plt.savefig(f'{path_save}vx_residual_author_abs.png')

        ax = sns.catplot(data=dataf, x="sensor", y="abs_residuy", hue="Author", kind="box")
        ax.set(xlabel='Sensor', ylabel='Absolute residual vy [m/y]')
        if show: plt.show()
        plt.savefig(f'{path_save}vy_residual_author_abs.png')

        ax = sns.catplot(data=dataf, x="sensor", y="residux", hue="Author", kind="box")
        ax.set(xlabel='Sensor', ylabel='Residual vx [m/y]')
        if show: plt.show()
        plt.savefig(f'{path_save}vx_residual_author.png')

        ax = sns.catplot(data=dataf, x="sensor", y="residuy", hue="Author", kind="box")
        ax.set(xlabel='Sensor', ylabel='Residual vy [m/y]')
        if show: plt.show()
        plt.savefig(f'{path_save}vy_residual_author.png')

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(dataf[dataf["Author"] == auth]['weightinix'], dataf[dataf["Author"] == auth]['residux'],
                       linestyle='', marker='o', color=color_list[i], markersize=3)
            ax[1].plot(dataf[dataf["Author"] == auth]['weightiniy'], dataf[dataf["Author"] == auth]['residuy'],
                       linestyle='', marker='o', color=color_list[i], markersize=3, label=auth)
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
        ax[1].set_xlabel(f'Quality indicator', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                     ncol=5)
        if show: plt.show()
        fig.savefig(f'{path_save}residu_qualitylog.png')

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(dataf[dataf["Author"] == auth]['weightinix'], dataf[dataf["Author"] == auth]['residux'],
                       linestyle='', marker='o', color=color_list[i], markersize=3)
            ax[1].plot(dataf[dataf["Author"] == auth]['weightiniy'],
                       dataf[dataf["Author"] == auth]['residuy'], linestyle='', marker='o',
                       color=color_list[i], markersize=3, label=auth)
        ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
        ax[1].set_xlabel(f'Quality indicator', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                     ncol=5)
        if show: plt.show()
        fig.savefig(f'{path_save}residu_quality.png')

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2, dataf[dataf["Author"] == auth]['residux'],
                       linestyle='', marker='o', color=color_list[i], markersize=3)
            ax[1].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2, dataf[dataf["Author"] == auth]['residuy'],
                       linestyle='', marker='o', color=color_list[i], markersize=3, label=auth)
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
        ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                     ncol=5)
        if show: plt.show()
        fig.savefig(f'{path_save}residu_tempbaseline_log.png')

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2,
                       abs(dataf[dataf["Author"] == auth]['residux']), linestyle='', marker='o',
                       color=color_list[i],
                       markersize=3)
            ax[1].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2,
                       abs(dataf[dataf["Author"] == auth]['residuy']), linestyle='', marker='o',
                       color=color_list[i], markersize=3, label=auth)
        ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
        ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                     ncol=5)
        if show: plt.show()
        fig.savefig(f'{path_save}residu_tempbaselineabs.png')

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2,
                       dataf[dataf["Author"] == auth]['residux'], linestyle='', marker='o', color=color_list[i],
                       markersize=3)
            ax[1].plot(np.array(offset_bar)[dataf["Author"] == auth] * 2,
                       dataf[dataf["Author"] == auth]['residuy'], linestyle='', marker='o',
                       color=color_list[i], markersize=3, label=auth)
        ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
        ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, fontsize=12,
                     ncol=5)
        if show: plt.show()
        fig.savefig(f'{path_save}residu_tempbaseline.png')

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(dataf[dataf["Author"] == auth]['sensor'],
                       dataf[dataf["Author"] == auth]['residux'], linestyle='', marker='o', color=color_list[i],
                       markersize=3)
            ax[1].plot(dataf[dataf["Author"] == auth]["sensor"],
                       dataf[dataf["Author"] == auth]['residuy'], linestyle='', marker='o',
                       color=color_list[i], markersize=3, label=auth)
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
        ax[1].set_xlabel(f'Temporal baseline [days]', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.1, 0), bbox_transform=fig1.transFigure, fontsize=12,
                     ncol=4)
        if show: plt.show()
        fig.savefig(f'{path_save}residu_sensor_log.png')

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(dataf[dataf["Author"] == auth]['sensor'],
                       abs(dataf[dataf["Author"] == auth]['residux']), linestyle='', marker='o',
                       color=color_list[i],
                       markersize=3)
            ax[1].plot(dataf[dataf["Author"] == auth]["sensor"],
                       abs(dataf[dataf["Author"] == auth]['residuy']), linestyle='', marker='o',
                       color=color_list[i], markersize=3, label=auth)
        ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
        ax[1].set_xlabel(f'Sensor', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.1, 0), bbox_transform=fig1.transFigure, fontsize=12,
                     ncol=4)
        if show: plt.show()
        fig.savefig(f'{path_save}residu_sensorabs.png')

        fig, ax = plt.subplots(2, 1, figsize=figsize)
        color_list = ['b', 'm', 'k', 'g', 'm']
        for i, auth in enumerate(dataf['Author'].unique()):
            ax[0].plot(dataf[dataf["Author"] == auth]['sensor'],
                       dataf[dataf["Author"] == auth]['residux'], linestyle='', marker='o', color=color_list[i],
                       markersize=3)
            ax[1].plot(dataf[dataf["Author"] == auth]["sensor"],
                       dataf[dataf["Author"] == auth]['residuy'], linestyle='', marker='o',
                       color=color_list[i], markersize=3, label=auth)
        ax[0].set_ylabel(f'Residual vx [{unit}]', fontsize=16)
        ax[1].set_ylabel(f'Residual vy [{unit}]', fontsize=16)
        ax[1].set_xlabel(f'Sensor', fontsize=16)
        plt.subplots_adjust(bottom=0.2)
        ax[1].legend(loc='lower left', bbox_to_anchor=(0.1, 0), bbox_transform=fig1.transFigure, fontsize=12,
                     ncol=4)
        if show: plt.show()
        fig.savefig(f'{path_save}residu_sensor.png')

# pixel_object = pixel_class()
# pixel_object.load_obs_data_from_pandas_df()