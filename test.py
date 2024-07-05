import sqlite3
from sqlite_ml.sqml import SQML
import pytermgui as ptg

conn = sqlite3.connect("/Users/flash/Desktop/real_result_database.dat")
c_names = conn.execute("select name from pragma_table_info('resultsMY');").fetchall()
labels = [(str(name[0])) for name in c_names]
c_names = conn.execute("select name from pragma_table_info('resultsMY');").fetchall()
labels = [str(name) for name in c_names]
container = ptg.Container(labels)
print(labels)
