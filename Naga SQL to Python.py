from mysql.connector import connect
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

mydb = connect(host="localhost", user="root", password="Naga2021", auth_plugin='mysql_native_password', database="naga")

result = pd.read_sql_query("SELECT * FROM predictions", mydb)
print(type(result))
# engine = create_engine("mysql+pymysql://root:Naga2021@localhost:3306/naga")
# result.to_sql("predictions_2", engine)
