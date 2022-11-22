import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper")

r_data = pd.read_csv('Rndl_result.csv', index_col=0).set_index(['event_file', 'repeats'])
py_data = pd.read_csv('pyndl_result.csv', index_col=0).set_index(['event_file', 'repeats'])
data = r_data.join(py_data).reset_index()

data_long = pd.wide_to_long(data,  ["wctime"], i=["event_num", "repeats"], j="implementation", sep='-', suffix='.+')
data_long = data_long.reorder_levels([2, 0, 1]).sort_index()  # index by method name

data_long.loc['Rndl2-1thread', 'parallel'] = False
data_long.loc['Rndl2-4thread', 'parallel'] = True
data_long.loc['Rndl1', 'parallel'] = False
data_long.loc['Rndl2-1thread', 'lib'] = 'ndl2'
data_long.loc['Rndl2-4thread', 'lib'] = 'ndl2'
data_long.loc['Rndl1', 'lib'] = 'ndl'


data_long.loc['pyndl_openmp1', 'parallel'] = False
data_long.loc['pyndl_openmp4', 'parallel'] = True
data_long.loc['pyndl_thread1', 'parallel'] = False
data_long.loc['pyndl_thread4', 'parallel'] = True
data_long.loc['pyndl_openmp1', 'lib'] = 'pyndl (ours, openMP)'
data_long.loc['pyndl_openmp4', 'lib'] = 'pyndl (ours, openMP)'
data_long.loc['pyndl_thread1', 'lib'] = 'pyndl (ours, threading)'
data_long.loc['pyndl_thread4', 'lib'] = 'pyndl (ours, threading)'


fg = sns.relplot(data=data_long, kind="line", x="event_num", y="wctime", hue='lib', style='lib', markers=True, dashes=False, errorbar='se', col='parallel', facet_kws=dict(legend_out=False), height=3)
fg.set_axis_labels('events', 'wall-clock time [sec]')   
fg.set(ylim=(0, 30))
sns.move_legend(fg, "upper left", bbox_to_anchor=(.57, .90), frameon=True)
fg.legend.set_title(None)
fg.axes[0, 0].set_title('Single processing')
fg.axes[0, 1].set_title('Parallel processing (2 jobs)')
fg.savefig('benchmark_result.pdf')         
fg.savefig('benchmark_result.png', transparent=True, dpi=600)         
fg.savefig('../paper/benchmark_result.png', transparent=True, dpi=600)         
fg.savefig('../docs/source/_static/benchmark_result.png', transparent=True, dpi=600)         

fg = sns.relplot(data=data_long.loc[['Rndl2-1thread', 'pyndl_openmp1'], :], kind="line", x="event_num", y="wctime", hue='lib', style='lib',
                 markers=True, dashes=False, errorbar='se')
fg.legend.set_title(None)
plt.title('Single processing')
plt.xlabel('events')
plt.ylabel('wall-clock time [sec]')
fg.savefig('single_result.pdf')
fg.savefig('single_result.png', transparent=True, dpi=600)

fg = sns.relplot(data=data_long[data_long.parallel==True], kind="line", x="event_num", y="wctime", hue='lib', style='lib',
                 markers=True, dashes=False, errorbar='se', facet_kws=dict(legend_out=False))
fg.legend.set_title(None)
plt.title('Parallel processing (2 jobs)')
plt.xlabel('events')
plt.ylabel('wall-clock time [sec]')
fg.savefig('parallel_result.pdf')
fg.savefig('parallel_result.png', transparent=True, dpi=600)
