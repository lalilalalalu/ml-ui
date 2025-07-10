import sqlite3
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from matplotlib.colors import Normalize


def get_table_columns(conn, table="results"):
    smt = "select name from pragma_table_info('{table}');".format(table=table)
    print("smt:", smt)
    columns = conn.execute(smt).fetchall()
    return [c_name for c_name in columns]


def get_df(conn, predict_config):
    print(predict_config)
    from sqlite_ml.sqml import SQML
    sqml = SQML()
    sqml.setup_schema(conn)
    sqml.register_functions(conn)
    sub_smt = ""
    if len(predict_config['features']) > 1:
        print(predict_config['features'])
        sub_smt = "json_object(" + ", ".join(
            re.escape("'") + f + re.escape("'") + ', [' + f + ']' for f in
            predict_config['features'][:-1]
        ) + ',' + re.escape("'") + \
                  predict_config['features'][-1] + re.escape("'") + ', [' + predict_config['features'][-1] + '])'
        print("focus: ", sub_smt)
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


def plot_himeno(df):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Normalize for color mapping
    norm = Normalize(vmin=df['MFLOPs'].min(), vmax=df['MFLOPs'].max())

    # Scatter plot for MFLOPs
    scatter = ax.scatter(
        df['procs'],
        df['MFLOPs'],
        c=df['MFLOPs'],
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='w',
        label='Measured Performance',
        norm=norm
    )

    # Plot prediction as a line
    cmap = plt.get_cmap('viridis')
    for category in df['SS_I'].unique():
        subset = df[df['SS_I'] == category]
        color = cmap(norm(subset['MFLOPs'].mean()))
        ax.plot(
            subset['procs'],
            subset['prediction'],
            label=f"Predicted Performance (Size: {category} x {category} x {category})",
            linestyle="--",
            color=color,
            alpha=0.8
        )

    # Set x-axis to integer ticks
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Label the axes
    ax.set_xlabel('Total Number of Processes (2 Processes per node)', fontsize=20)
    ax.set_ylabel('Performance (MFLOPs)', fontsize=20)


    # Set font size for ticks
    ax.tick_params(axis='both', labelsize=20)

    # Add a legend
    ax.legend(fontsize=15)

    # Show the plot
    plt.show()
    plt.savefig("himeno.png")


if __name__ == '__main__':
    # conn = sqlite3.connect("/Users/flash/Desktop/DBs/scale_3056_database.dat")
    conn = sqlite3.connect("/Users/flash/Desktop/DBs/himeno_result_database_x_750.dat")
    config0 = {"dataset":"himeno", "features":["SS_I", "procs"], "ex_name": "gradientb", "target":"MFLOPs"}
    df = get_df(conn, config0)
    plot_himeno(df)




