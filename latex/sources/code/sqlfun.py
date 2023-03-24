import mysql.connector
from mysql.connector import Error


# funcion para establecer conexion con el servidor mysql
def db_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
            )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection


# funcion para crear una base de datos
def create_database(db_name, connection):
    connection = connection
    cursorobject = connection.cursor()
    cursorobject.execute("CREATE DATABASE " + db_name)
    print(f"La base de datos '{db_name}' ha sido creada correctamente.")
    return cursorobject


# funcion para eliminar una base de datos:
def delete_database(host, user, passwd, db_name):
    connection = None
    try:
        # Conectar a la base de datos
        connection = mysql.connector.connect(
            host=host,
            user=user,
            passwd=passwd
        )

        # Crear un cursor para ejecutar consultas
        cursor = connection.cursor()

        # Eliminar la base de datos
        cursor.execute(f"DROP DATABASE {db_name}")

        print(f"La base de datos '{db_name}' ha sido eliminada correctamente.")

    except Error as err:
        print(f"Error al eliminar la base de datos: '{err}'")

    finally:
        # Cerrar la conexión
        if connection:
            connection.close()



def create_table(table_name, connection):
    # Selecciona la base de datos
    cursor = connection.cursor()
    cursor.execute("USE nombre_de_la_base_de_datos")

    # Define la consulta para crear la tabla
    query = "CREATE TABLE " + table_name + " (id INT PRIMARY KEY AUTO_INCREMENT, nombre VARCHAR(255), edad INT)"
    '''
    CREATE TABLE nombre_de_tabla (
        id INT PRIMARY KEY AUTO_INCREMENT,
        nombre VARCHAR(255),
        edad INT
    )'''

    # Ejecuta la consulta para crear la tabla
    cursor.execute(query)

    # Agrega una columna a la tabla
    query = '''
    ALTER TABLE nombre_de_tabla
    ADD COLUMN direccion VARCHAR(255)'''
    cursor.execute(query)

    # Cierra el cursor y la conexión a la base de datos
    cursor.close()
    connection.close()

