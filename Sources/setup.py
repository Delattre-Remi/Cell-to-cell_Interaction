#entrez "python setup.py build" dans la console pour écécuter
#Un dossier build/exe.win-amd64-3.9 va être créé
#Copier tous les fichier du projet dedans
#exévuter main.exe pour lancer le programme

from cx_Freeze import setup, Executable

#Liste complète des packages utilisés 
packages = [
    "tkinter",
    "cv2",
    "numpy",
    "re",
    "os",
    "matplotlib",
    "functools",
    "PIL",
    "configparser",
    "time",
    "multiprocessing",
    "pandas",
    "csv"
]

options = {
    'build_exe': {    
        'packages':packages,
    },
}

#Adaptez les valeurs des variables "name", "version", "description" au programme.
setup(
    name = "Mon Programme",
    options = options,
    version = "1.0",
    description = 'test',
    executables = [Executable("main.py", base=None)]
)