import re

# Leer la lista de acrónimos desde un archivo de texto
with open('GNSS-acronyms.txt', 'r') as file:
    acronyms = file.read().splitlines()

# Ordenar los acrónimos alfabéticamente
acronyms.sort()

# Escribir los acrónimos en formato LaTeX
for acronym in acronyms:
    # Separar la sigla de la descripción
    parts = re.split('\s+', acronym, maxsplit=1)
    sigla = parts[0]
    descripcion = parts[1] if len(parts) > 1 else ''

    # Escribir el acrónimo en formato LaTeX
    print(f'\\acro{{{sigla}}}{{{descripcion}}}')
