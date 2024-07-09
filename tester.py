import sqlite3
from sqlite_ml.sqml import SQML
import re

if __name__ == '__main__':
    conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")
    # setup sqlite-ml extension
    sqml = SQML()
    sqml.setup_schema(conn)
    sqml.register_functions(conn)
    c_names = conn.execute("select name from pragma_table_info('results');").fetchall()
    f_list = [(name[0], idx, True) for idx, name in enumerate(c_names)]
    f_str = " json_array(" + ", ".join('[' + f[0] + ']' for f in f_list[:-1]) + ', ['+ f_list[-1][0] + '])'
    f_str = " json_object(" + ", ".join(re.escape("'")+f[0]+re.escape("'")+', [' + f[0] + ']' for f in f_list[:-1]) + ',' + re.escape("'")+f_list[-1][0] +re.escape("'") +', [' + f_list[-1][0] + '])'
    smt ="""SELECT sqml_predict(
        'mytest',
        (
            SELECT {f_str} FROM resultsMY
            LIMIT 1
        )
    ) AS prediction;""".format(f_str=f_str)
    print(smt)

    new_smt = """
    SELECT
  results.*,
  batch.value AS prediction,
  results.WRbwminMiB = batch.value AS match
FROM
  results
  JOIN json_each (
    (
      SELECT
        sqml_predict_batch(
          'Demo877888',
          json_group_array(
            {f_str}
          )
        )
      FROM
        results
    )
  ) batch ON (batch.rowid + 1) = results.rowid
WHERE match = FALSE;
    """.format(f_str=f_str)

    new_smt2 = """
SELECT
  results.*,
  sqml_predict(
    'Demo877888',
    {f_str}
  ) AS prediction
FROM results
LIMIT 1;""".format(f_str=f_str)

    test_3 = """
    SELECT
       results.*,
       batch.value AS prediction,
       results.WRbwminMiB = batch.value AS match
    FROM
      results
      JOIN json_each (
        (
          SELECT
            sqml_predict_batch(
              'Demo877888',
              json_group_array(
                 json_object('WRbwmaxMiB', [WRbwmaxMiB], 'WRbwmeanMiB', [WRbwmeanMiB], 'WRiopsmaxOPS', [WRiopsmaxOPS], 
                 'WRiopsminOPS', [WRiopsminOPS], 'WRiopsmeanOPS', [WRiopsmeanOPS], 'RDbwmaxMiB', [RDbwmaxMiB], 
                 'RDbwminMiB', [RDbwminMiB], 'RDbwmeanMiB', [RDbwmeanMiB], 'RDiopsmaxOPS', [RDiopsmaxOPS], 
                 'RDiopsminOPS', [RDiopsminOPS],'RDiopsmeanOPS', [RDiopsmeanOPS])
              )
            )
          FROM
            results
        )
      ) batch ON (batch.rowid + 1) = results.rowid
    WHERE match = False;
        """

    print(new_smt)
    results = conn.execute(test_3).fetchall()
    for idx, row in enumerate(results):
        print(tuple(row))
        print("---------%i-----------" %idx)
    #print(testot)

