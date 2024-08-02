import sqlite3
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_table_columns(conn, table="results"):
    smt = "select name from pragma_table_info('{table}');".format(table=table)
    print("smt:", smt)
    columns = conn.execute(smt).fetchall()
    return [c_name for c_name in columns]


def get_df(conn, predict_config):
    from sqlite_ml.sqml import SQML
    sqml = SQML()
    sqml.setup_schema(conn)
    sqml.register_functions(conn)
    sub_smt = ""
    if len(predict_config['features']) > 1:
        sub_smt = "json_object(" + ", ".join(
            re.escape("'") + f[0] + re.escape("'") + ', [' + f[0] + ']' for f in
            predict_config['features'][:-1]) + ',' + re.escape("'") + \
                  predict_config['features'][-1][0] + re.escape("'") + ', [' + predict_config['features'][-1][
                      0] + '])'
    else:
        sub_smt = "json_object(" + ", ".join(
            re.escape("'") + f + re.escape("'") + ', [' + f + '])' for f in predict_config['features'])
    smt_batched = """
    SELECT
      '{table}'.*,
      batch.value AS prediction
    FROM
      '{table}'
      JOIN json_each (
        (
          SELECT
            sqml_predict_batch(
              '{name}',
              json_group_array(
              {features}
              )
            )
          FROM
            '{table}'
        )
      ) batch ON (batch.rowid + 1) = '{table}'.rowid;
        """.format(name=predict_config['ex_name'], table=predict_config['dataset'], features=sub_smt,
                   target=predict_config['target'])

    print("smt:", smt_batched)
    results = conn.execute(smt_batched).fetchall()
    get_table_columns(conn, predict_config['dataset'])
    header = [name[0] for name in get_table_columns(conn, predict_config['dataset'])] + ["prediction"]
    rows =[]
    rows.append(tuple(header))
    for idx, row in enumerate(results):
        rows.append(tuple(row))
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df


def plot_multiple_dataframes(dfs, predict_configs, main_title):
    num_dfs = len(dfs)
    fig, axs = plt.subplots(1, num_dfs, figsize=(10 * num_dfs, 8), sharex=True, sharey=True)

    # Find the global min and max for the target values
    min_target = min(df[predict_config['target']].min() for df, predict_config in zip(dfs, predict_configs))
    max_target = max(df[predict_config['target']].max() for df, predict_config in zip(dfs, predict_configs))

    norm = plt.Normalize(min_target, max_target)

    for i, (df, predict_config) in enumerate(zip(dfs, predict_configs)):
        ax = axs[i] if num_dfs > 1 else axs

        x_label = predict_config['features'][0]
        y_label = predict_config['target']
        prediction_label = "prediction"

        # Scatter plot with color gradient based on normalized target values
        scatter = ax.scatter(df[x_label], df[y_label], c=df[y_label], cmap='viridis', s=100, alpha=0.7,
                             edgecolors='w', label='Data Points', norm=norm)

        # Dotted grey line plot
        ax.plot(df[x_label], df[prediction_label], color="red", linestyle='--', linewidth=2, label='Prediction')

        # Labels for axes
        # only for the case study
        x_label = "mean bandwidth [MiB/s]"
        if "min" in y_label:
            y_label= "min bandwidth [MiB/s]"
        elif "max" in y_label:
            y_label= "max bandwidth [MiB/s]"

        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)

        # Custom ticks
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Adjust legend location to avoid overlap
        ax.legend(loc='upper left', fontsize=20)

        # Grid
        ax.grid(True, linestyle='--', alpha=0.6)

    # Add a single color bar for the entire figure
    #cbar = fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    #cbar.set_label("Normalized Bandwidth", fontsize=20)
    #cbar.ax.tick_params(labelsize=20)

    # Add main title for the entire figure
    fig.suptitle(main_title, fontsize=20, y=1.05)  # Adjust y to ensure title is visible

    # Adjust layout manually
    fig.subplots_adjust(left=0.05, right=0.85, top=0.90, bottom=0.10, wspace=0.1)  # Adjust top margin for title

    plt.show()


def generate_predict_sql(conn, predict_config):
    from sqlite_ml.sqml import SQML
    sqml = SQML()
    sqml.setup_schema(conn)
    sqml.register_functions(conn)
    sub_smt = ""
    if len(predict_config['features']) > 1:
        sub_smt = "json_object(" + ", ".join(
            re.escape("'") + f[0] + re.escape("'") + ', [' + f[0] + ']' for f in
            predict_config['features'][:-1]) + ',' + re.escape("'") + \
                  predict_config['features'][-1][0] + re.escape("'") + ', [' + predict_config['features'][-1][
                      0] + '])'
    else:
        sub_smt = "json_object(" + ", ".join(
            re.escape("'") + f + re.escape("'") + ', [' + f + '])' for f in predict_config['features'])
    smt_batched = """
    SELECT
      '{table}'.*,
      batch.value AS prediction
    FROM
      '{table}'
      JOIN json_each (
        (
          SELECT
            sqml_predict_batch(
              '{name}',
              json_group_array(
              {features}
              )
            )
          FROM
            '{table}'
        )
      ) batch ON (batch.rowid + 1) = '{table}'.rowid;
        """.format(name=predict_config['ex_name'], table=predict_config['dataset'], features=sub_smt,
                   target=predict_config['target'])

    print("smt:", smt_batched)
    results = conn.execute(smt_batched).fetchall()
    get_table_columns(conn, predict_config['dataset'])
    header = [name[0] for name in get_table_columns(conn, predict_config['dataset'])] + ["prediction"]
    rows =[]
    rows.append(tuple(header))
    for idx, row in enumerate(results):
        rows.append(tuple(row))

    df = pd.DataFrame(rows[1:], columns=rows[0])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    # Scatter plot with color gradient based on RDbwmaxMiB

    x_label = predict_config['features'][0]
    y_label = predict_config['target']
    predicat_label = "prediction"

    scatter = ax.scatter(df[x_label], df[y_label], c=df[y_label], cmap='viridis', s=100, alpha=0.7,
                         edgecolors='w', label='Data Points')

    # Line plot
    ax.plot(df[x_label], df[predicat_label], color="blue", linewidth=3, label='Prediction')

    # Labels for axes
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # Custom ticks
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Title and legend
    plt.title('Enhanced 2D Scatter Plot', fontsize=16)
    plt.legend(fontsize=12)

    # Color bar to show the scale of the RDbwmaxMiB values
    cbar = plt.colorbar(scatter)
    cbar.set_label(y_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Grid
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    """
    sns.pairplot(df)
    plt.show()

    plt.figure(figsize=(10, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()
    
    """
"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(df['feature1'], df['feature2'], df['target'], c='r', marker='o')

    # Labels
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Target')

    plt.title('3D Scatter Plot')
    plt.show()
"""


if __name__ == '__main__':
    conn = sqlite3.connect("/Users/flash/Desktop/DBs/scale_3056_database.dat")


    config0 = {"dataset":"cleanedRdWdBw", "features":["WRbwmaxMiB", ], "ex_name": "wrbwmax", "target":"RDbwmaxMiB"}
    # generate_predict_sql(conn, config1)
    config1 = {"dataset":"ReadBwMeanAndMax", "features":["RDbwmeanMiB", ], "ex_name": "predicatReadMax", "target":"RDbwmaxMiB"}
    # generate_predict_sql(conn,config2)
    config2 = {"dataset": "ReadBwMeanAndMin", "features": ["RDbwmeanMiB", ], "ex_name": "predicatReadMin",
               "target": "RDbwminMiB"}
    config3 = {"dataset": "WriteBwMeanAndMax", "features": ["WRbwmeanMiB", ], "ex_name": "WriteBwMeanAndMax",
               "target": "WRbwmaxMiB"}

    config4 = {"dataset": "WriteBwMeanAndMin", "features": ["WRbwmeanMiB", ], "ex_name": "WriteBwMeanAndMin",
               "target": "WRbwminMiB"}

    config5 = {"dataset": "WriteBwMeanProcs", "features": ["j_task", ], "ex_name": "WriteBwMeanProcs",
               "target": "WRbwmeanMiB"}

    config6 = {"dataset": "ReadBwMeanProcs", "features": ["j_task", ], "ex_name": "ReadBwMeanProcs",
               "target": "RDbwmeanMiB"}

    config7 = {"dataset": "IopsBWReadMean", "features": ["RDiopsmeanOPS", ], "ex_name": "BWIopsReadMean",
               "target": "RDbwmeanMiB"}

    config8 = {"dataset": "IopsBWWriteMean", "features": ["WRiopsmeanOPS", ], "ex_name": "IopsBWWriteMean",
               "target": "WRbwmeanMiB"}

    configs = [config1, config2]
    dfs =[]
    for config in configs:
        dfs.append(get_df(conn,config))
    plot_multiple_dataframes(dfs,configs, "Read")

    configs = [ config3, config4]
    dfs =[]
    for config in configs:
        dfs.append(get_df(conn,config))
    plot_multiple_dataframes(dfs,configs, "Write")

    configs = [config5, config6]
    dfs = []
    for config in configs:
        dfs.append(get_df(conn, config))
    plot_multiple_dataframes(dfs, configs, "test")


    configs = [config7, config8]
    dfs = []
    for config in configs:
        dfs.append(get_df(conn, config))
    plot_multiple_dataframes(dfs, configs, "test")


