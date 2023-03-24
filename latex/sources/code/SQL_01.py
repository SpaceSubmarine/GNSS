import sqlfun
import mysql.connector
from mysql.connector import Error

# Inputs
host = '127.0.0.1'
user = 'marc'
passwd = 'Humo_2022'
db_name = 'DF_2'
tb_name = "tablita"


# Llamar a la función de conexión a la base de datos
connection = sqlfun.db_connection(host, user, passwd)

# Para llamar a la función delete_database
#sqlfun.delete_database(host, user, passwd, db_name)

# Para llamar a la función create_database
sqlfun.create_database(db_name, connection)
#connection = sqlfun.db_connection(host, user, passwd, db_name)
#sqlfun.create_table(tb_name, connection)


