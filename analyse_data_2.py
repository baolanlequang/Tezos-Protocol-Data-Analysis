import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def read_data_by_workload_level(level, df_all_metrics, filter = True):
  # Filter data by wordload level using pandas
  df = df_all_metrics.loc[df_all_metrics['Workload Level'] == level]
  if filter:
    df = df.loc[df['Block Creation Interval'] < 40]
  
  # Select and store data
  tfl = df['TFL']
  tps = df['TPS']
  bpr = df['BPR']
  gini = df['Gini']
  hhi = df['HHI']
  block_size = df['Block Size']
  block_creation_interval = df['Block Creation Interval']
  net_work_size = df['Number of Validating Nodes']
  return {
    'Workload Level': level,
    'BPR': bpr.values,
    'Gini': gini.values,
    'HHI': hhi.values,
    'Transaction Finalization Latency': tfl.values,
    'Transaction Throughput': tps.values,
    'block_size': block_size.values,
    'block_creation_interval': block_creation_interval.values,
    'net_work_size': net_work_size.values
  }

def read_data():
  with open('./All_Metrics_by_Workload.csv', 'r') as file_all_metrics:
    df_all_metrics = pd.read_csv(file_all_metrics, sep=";")

  # Read data when Block Creation Interval less than 40 seconds
  data_level_105_filtered = read_data_by_workload_level(105, df_all_metrics, filter=True)
  
  # Read data for all Block Creation Interval
  data_level_105_no_filter = read_data_by_workload_level(105, df_all_metrics, filter=False)
 
  return {
    'data_level_105_filtered': data_level_105_filtered,
    'data_level_105_no_filter': data_level_105_no_filter
  }

def generate_2d_figure(data, filter=True):
  # DoD data name
  dod_metric_names = ['BPR','Gini','HHI']
  
  # Scalability data name
  scalability_metric_names = ['Transaction Finalization Latency','Transaction Throughput']
  for dod_name in dod_metric_names:
    dod_metric = data[dod_name]
    for scalability_name in scalability_metric_names:
      scalability_metric = data[scalability_name]
      
      # calculate Spearman coefficient
      spearman_res = spearmanr(dod_metric, scalability_metric)
      
      # plot the data
      fig = plt.figure(figsize=(16,9))
      poly = PolynomialFeatures(degree=1)
      plt.xticks(fontsize=14)
      plt.yticks(fontsize=14)
      
      # plot by using the scatter
      plt.scatter(dod_metric, scalability_metric, color='blue')
      plt.xlabel(dod_name, fontsize=18)
      plt.ylabel(scalability_name, fontsize=18)
      
      # draw regression line
      sorted_dod = sorted(dod_metric)
      sorted_scal = []
      for ite in sorted_dod:
        indices = np.where(dod_metric == ite)[0]
        idx = indices[0]
        sorted_scal.append(scalability_metric[idx])
      
      x_poly = poly.fit_transform(np.array(sorted_dod).reshape(-1, 1))

      model = LinearRegression()
      model.fit(x_poly, sorted_scal)

      y_pred = model.predict(x_poly)
      plt.plot(sorted_dod, y_pred, color='red')
      
      # draw the title and save the figure
      if filter:
        fig.suptitle(f'Workload Level: 105%, Block Creation Interval: < 40 seconds\nSpearman: {spearman_res.statistic:.4f}', fontsize=24)
        plt.savefig(f'./correlation_by_workload/pdf/Correlation_{dod_name}_{scalability_name}_105.pdf', dpi=80, bbox_inches='tight')
      else:
        fig.suptitle(f'Workload Level: 105%\nSpearman: {spearman_res.statistic:.4f}', fontsize=24)
        plt.savefig(f'./correlation_by_workload/pdf/Correlation_{dod_name}_{scalability_name}_105_nofilter.pdf', dpi=80, bbox_inches='tight')
        
      plt.close()
  
def generate_3d_figure(data, filter=True):
  # DoD data name
  dod_metric_names = ['BPR','Gini','HHI']
  
  for dod_name in dod_metric_names:
    dod_metric = data[dod_name]
    tfl = data['Transaction Finalization Latency']
    tps = data['Transaction Throughput']
      
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dod_metric, tfl, tps, marker='o')
    ax.set_xlabel(dod_name)
    ax.set_ylabel('Transaction Finalization Latency')
    ax.set_zlabel('Transaction Throughput')
    ax.view_init(elev=20., azim=-35, roll=0)
    if filter:
      ax.set_title(f'Correlation between {dod_name} and Scalability\nWorkload Level: 105%, Block Creation Interval: < 40 seconds')
      plt.savefig(f'./correlation_by_workload/pdf/Correlation_{dod_name}_scal_3d_105.pdf', dpi=80)
    else:
      ax.set_title(f'Correlation between {dod_name} and Scalability\nWorkload Level: 105%')
      plt.savefig(f'./correlation_by_workload/pdf/Correlation_{dod_name}_scal_3d_105_no_filter.pdf', dpi=80)
    
    
    plt.close()

def plot_data():
  # Read data from All_Metrics_by_Workload.csv file
  metrics_data = read_data()
  
  data_level_105_filtered = metrics_data['data_level_105_filtered']
  data_level_105_no_filter = metrics_data['data_level_105_no_filter']
  
  generate_2d_figure(data_level_105_filtered)
  generate_2d_figure(data_level_105_no_filter, filter=False)
  
  generate_3d_figure(data_level_105_filtered)
  generate_3d_figure(data_level_105_filtered, filter=False)
      
  
plot_data()