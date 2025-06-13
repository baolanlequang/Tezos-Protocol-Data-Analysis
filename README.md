# How to run the data analysis

## Requirement
1. Make sure that you have `python` environment

2. Run this command in your terminal to install the required libraries

```bash
pip install -r requirements.txt
```

3. Use this command to run the data analysis

```bash
python analyse_data_2.py
```

The figures will be generated in folder `correlation_by_workload/pdf`

## The content of analysis script

1. Read data from `All_Metrics_by_Workload.csv`

2. Generate analysed data to 2D figures by function `generate_2d_figure`

3. Generate analysed data to 3D figures by function `generate_3d_figure`

4. The function `read_data_by_workload_level` is used to read and process data depending on the wordload level by param `level`. By default, param `filter` is alway `True` to select data that has `Block Creation Interval` less than 40 seconds. If `filter` is `False`, the data from all `Block Creation Interval` will be loaded.

5. The Spearman coefficient is calculated when generate 2D figures. This is done by using `spearmanr` from `scipy`. The data is plotted by scatter plot from `mamatplotlib`

