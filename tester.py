# import sqlite3
# from sqlite_ml.sqml import SQML
# import re

# if __name__ == '__main__':
#     conn = sqlite3.connect("DBs/himeno_result_database_x_750.dat")
#     # setup sqlite-ml extension
#     sqml = SQML()
#     sqml.setup_schema(conn)
#     sqml.register_functions(conn)
#     smtml = """SELECT
#   himeno.*,
#   batch.value AS prediction
# FROM
#   himeno
# JOIN json_each (
#   (
#     SELECT
#       sqml_predict_batch(
#         'gradientb',
#         json_group_array(
#           json_object('SS_I', [SS_I], 'procs', [procs])
#         )
#       )from himeno
#   )
# ) AS batch ON (batch.rowid + 1) = himeno.rowid;
# """
#     print(smtml)
#     results = conn.execute(smtml).fetchall()
#     for idx, row in enumerate(results):
#         print(tuple(row))
#         print("---------%i-----------" %idx)
#     #print(testot)



import sqlite3
from sqlite_ml.sqml import SQML
import re

if __name__ == '__main__':
    conn = sqlite3.connect("DBs/himeno_result_database_x_750.dat")
    # setup sqlite-ml extension
    sqml = SQML()
    sqml.setup_schema(conn)
    sqml.register_functions(conn)
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


    smtml = """SELECT sqml_train(
                                      'Demo2571',
                                      'regression',
                                      'linear_regression',
                                      'BABA6afa23cc-a30e-4193-84e9-3db8c5b259d4',
                                      'WRbwmaxMiB'
                                  )
                                  AS
                                  training;"""
    print(smtml)
    results = conn.execute(smtml).fetchall()
    for idx, row in enumerate(results):
        print(tuple(row))
        print("---------%i-----------" %idx)
    #print(testot)

