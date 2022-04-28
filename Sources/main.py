from asyncio.subprocess import Process
from multiprocessing.pool import RUN
from re import L, S
import tkinter as tk
from tkinter import Frame, Toplevel, mainloop, ttk
from tkinter import filedialog
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
import numpy as np
from functools import partial
from cellPresenceDetermination import Analyser
from PIL import Image, ImageTk
from configparser import ConfigParser
from utils import getInitDir
import csv
LARGEFONT = ("Verdana", 35)
RUN_FLAG = True
IMAGE_DISP = True
import time
from threading import Thread
global stats
stats = {
	"numberOfValidLeukemicCells" : [],
	"numberOfValidTCells" : [],
	"numberOfinteractions": [],
	"pourcentageOfValidLeukemicCells": [],
	"pourcentageOfValidTCells": [],
	"pourcentageOfInteractions": []
}

class tkinterApp(tk.Tk):
	def setRenderSuperPosition(self):
		self.renderSuperPosition = not self.renderSuperPosition
	
	def setRenderStep(self):
		self.renderStep = not self.renderStep

	def BrowseBrightfield(self):
		self.folderBrightfield = filedialog.askopenfilename(initialdir = getInitDir(self.folderBrightfield), title = "Select the image with empty cells")
		self.saveParam()
		self.Analyzer = None

	def BrowseLeukemicCell(self):
		self.listeLeukemicCell = []
		self.folderLeukemicCell = filedialog.askdirectory(initialdir = getInitDir(self.folderLeukemicCell),title = "Select the directory which contains the images of Leukemic cells")
		self.saveParam()
		for root, dirs, files in os.walk(self.folderLeukemicCell):
			for file in files:
				fichier = os.path.join(root,file)
				self.listeLeukemicCell.append(fichier)
		
	def BrowseTCell(self):
		self.listeTCell = []
		self.folderTCells = filedialog.askdirectory(initialdir = getInitDir(self.folderTCells),title = "Select the directory which contains the images of T-cells")
		self.saveParam()
		for root, dirs, files in os.walk(self.folderTCells):
			for file in files:
				fichier = os.path.join(root,file)
				self.listeTCell.append(fichier)

	def StartAnalysis(self):
		dif = len(self.listeTCell) - len(self.listeLeukemicCell)
		if dif > 0 : self.listeTCell = self.listeTCell[dif:]
		elif dif < 0 : self.listeLeukemicCell = self.listeLeukemicCell[abs(dif):]

		if(self.Analyzer is None) : self.Analyzer = Analyser(self.folderBrightfield)
		self.Analyzer.setRenderSuperPos(self.renderSuperPosition)
		self.Analyzer.setRenderSteps(self.renderStep)

		if self.renderStep : self.topLevel = AnalysisWindow(self)
		ProcessAnalysis(self.listeTCell, self.listeLeukemicCell, self.Analyzer, self.saveFolder)
		if self.renderStep : self.topLevel.DisplayImage()
		if not RUN_FLAG : self.topLevel.DisplayImagesEnd()

	def ResetImgDir(self):
		self.folderBrightfield = ""
		self.folderLeukemicCell = ""
		self.folderTCells = ""
		self.PopUp("Reset","Images directories have been reset")

	def PopUp(self,titre,message):
		tk.messagebox.showinfo(titre,message)

	def changeTheme(self,text,**kwargs):
		self.theme = text
		self.styleApp.theme_use(text)

	def changeSaveFolder(self,text):
		self.saveFolder = filedialog.askdirectory (initialdir = "/",title = "Selectionner un dossier")
		text.insert(tk.INSERT, self.saveFolder)

	def plotGraphInside(self,x,y,title,win):
		plt.clf()
		figure = Figure(figsize=(3, 3), dpi=100)
		plot = figure.add_subplot(1, 1, 1)
		plot.plot(x, y)
		plt.title(title)
		canvas = FigureCanvasTkAgg(figure, win)
		canvas.get_tk_widget().grid(row=0, column=0)

	def pltGraphOutside(self,x,y,title):
		plt.plot(x,y)
		plt.title(title)
		plt.show()

	def saveParam(self):
		config = ConfigParser(allow_no_value=True)
		config.add_section("Parameters")
		config.set("Parameters","folderBrightfield", self.folderBrightfield)
		config.set("Parameters","folderLeukemicCell", self.folderLeukemicCell)
		config.set("Parameters","folderTCells", self.folderTCells)
		config.set("Parameters","theme",self.theme)
		config.set("Parameters","saveFolder", self.saveFolder)
		with open('config.ini', 'w') as conf : config.write(conf)

	def Close(self):
		config = ConfigParser(allow_no_value=True)
		config.add_section("Parameters")
		config.set("Parameters","folderBrightfield", self.folderBrightfield)
		config.set("Parameters","folderLeukemicCell", self.folderLeukemicCell)
		config.set("Parameters","folderTCells", self.folderTCells)
		config.set("Parameters","theme",self.theme)
		config.set("Parameters","saveFolder", self.saveFolder)
		with open('config.ini', 'w') as conf : config.write(conf)
		self.destroy()

	def saveGraph(self,title):
		plt.savefig(self.saveFolder + title + '.png')

	def resetList(self):
		self.list_of_files=[]

	def PrintFilesDir(self):
		if len(self.list_of_files)>0:
			print(self.list_of_files)
		else:
			print("Error no valid files were found")

	def loadParams(self):
		try: #Loading parameters
			self.config = ConfigParser()
			self.config.read("config.ini")
			self.folderBrightfield = self.config["Parameters"]["folderBrightfield"]
			self.folderLeukemicCell = self.config["Parameters"]["folderLeukemicCell"]
			self.folderTCells = self.config["Parameters"]["folderTCells"]
			self.theme = self.config["Parameters"]["Theme"]
			self.saveFolder = self.config["Parameters"]["saveFolder"]
			print("Loaded parameters : \n\t- folderBrightfield : {} \n\t- folderLeukemicCell : {} \n\t- folderTCells : {}".format(self.folderBrightfield, self.folderLeukemicCell, self.folderTCells))
		except Exception as e: #if the loading fails, default parameters are used
			print("Loading parameters failed, creating new empty ones")
			print(e)
			self.folderBrightfield = ""
			self.folderLeukemicCell = ""
			self.folderTCells = ""
			self.theme = "radiance"
			self.saveFolder = ""
			config = ConfigParser(allow_no_value=True)
			config.add_section("Parameters")
			config.set("Parameters","folderBrightfield", '')
			config.set("Parameters","folderLeukemicCell", '')
			config.set("Parameters","folderTCells", '')
			config.set("Parameters","theme", self.theme)
			config.set("Parameters","saveFolder", '')
			with open('config.ini', 'w') as conf : config.write(conf)

	def __init__(self, *args, **kwargs):		
		# __init__ function for class Tk
		tk.Tk.__init__(self, *args, **kwargs)
		self.listeLeukemicCell = []
		self.listeTCell = []
		self.Analyzer = None
		self.topLevelListe = []
		self.frames = {}
		self.wm_iconbitmap('Image/logo.ico')
		self.loadParams()
		self.DisplayImage = False
		self.renderSuperPosition = False
		self.renderStep = False
		self.listeOfThemes = self.listeOfThemes = os.listdir('Themes')
		self.styleApp = ttk.Style()
		for i in self.listeOfThemes : self.tk.call("source", "Themes/"+i+"/"+i+".tcl")
		self.styleApp.theme_use(self.theme)
		self.dic = {}
		self.interactionPositions = []

		# creating a container
		container = tk.Frame(self)
		container.pack(side = "top", fill = "both", expand = True)

		container.grid_rowconfigure(0, weight = 1)
		container.grid_columnconfigure(0, weight = 1)

		# iterating through a tuple consisting
		# of the different page layouts
		for F in (Menu, Parameters):
			frame = F(container, self)
			# initializing frame of that object from
			# Menu, Statistics, Parameters respectively with
			# for loop
			self.frames[F] = frame
		self.show_frame(Menu)

	# to display the current frame passed as
	# parameter
	def show_frame(self, cont):
		for F in self.frames.values():
			F.grid_forget()
		frame = self.frames[cont]
		frame.grid(row = 0, column = 0, sticky ="WN")

# first window frame Menu

class Menu(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)

		buttonDisplayImages = ttk.Checkbutton(self, text = "Display Images live\n(aprox. 5x slower)",command=partial(controller.setRenderStep))
		buttonDisplayImages.grid(row = 3, column = 2, padx = 10, pady = 10,sticky='w')

		buttonSuperPosition = ttk.Checkbutton(self, text = "create the superposition image\n(aprox. 2x slower)",command=partial(controller.setRenderSuperPosition))
		buttonSuperPosition.grid(row = 4, column = 2, padx = 10, pady = 10,sticky='w')

		buttonReset =  ttk.Button(self, text ="Reset", command = partial(controller.ResetImgDir))
		buttonReset.grid(row = 2, column = 2, padx = 10, pady = 10,sticky='w')

		# buttonStatistics = ttk.Button(self, text ="Statistics", command = partial(controller.show_frame,Statistics))
		# buttonStatistics.grid(row = 2, column = 1, padx = 10, pady = 10,sticky='w')

		buttonMenu = ttk.Button(self,text="Menu", state='disabled')
		buttonMenu.grid(row = 1, column = 1, padx = 10, pady = 10,sticky='w')

		## button to show frame 2 with text layout2
		buttonParameters = ttk.Button(self, text ="Parameters", command = partial(controller.show_frame,Parameters))
		buttonParameters.grid(row = 2, column = 1, padx = 10, pady = 10,sticky='w')

		buttonStartProg = ttk.Button(self,text="Start analysis", command = partial(controller.StartAnalysis))
		buttonStartProg.grid(row=1,column=2,padx=10,pady=10,sticky='w')

		buttonBrightfield = ttk.Button(self,text="Choose the image with no cells",command=partial(controller.BrowseBrightfield))
		buttonBrightfield.grid(row=1,column=3,padx=10,pady=10,sticky='w')

		buttonLeukemicCells = ttk.Button(self,text="Choose the folder containing images with leukemic cells", command=partial(controller.BrowseLeukemicCell))
		buttonLeukemicCells.grid(row=2,column=3,padx=10,pady=10,sticky='w')

		buttonTcells = ttk.Button(self,text="Choose the folder containing images with T-cells",command = partial(controller.BrowseTCell))
		buttonTcells.grid(row=3,column=3,padx=10,pady=10,sticky='w')

		#buttonSaveResult = ttk.Button(self,text="Save analysis")
		#buttonSaveResult.grid(row = 4, column = 5 , padx = 10 , pady = 10 ,sticky='w')

		buttonQuit = ttk.Button(self,text="Quit" , style = 'BW.TButton' ,command=controller.Close)
		buttonQuit.grid(row=3,column = 1, padx = 10 ,pady = 10 ,sticky='w')

# second window frame Statistics
class Statistics(tk.Frame):
	def DisplayImage(self,nameImg,row,column):
		photo = ImageTk.PhotoImage(Image.open("Processed/"+ nameImg).resize((150,150)))
		labelTmp = ttk.Label(self,image = photo)
		labelTmp.Image = photo
		labelTmp.grid(row=row, column = column,padx = 10, pady = 10)

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		#label = ttk.Label(self, text ="Statistics", font = LARGEFONT)
		#label.grid(row = 0, column = 4, padx = 10, pady = 10,sticky='w')

		#buttonPlot= ttk.Button(self, text="Plot graph",command = partial(controller.pltGraphOutside,np.arange(1,4),np.arange(1,4),"VarTheme"))
		#buttonPlot.grid(row = 1, column = 4, padx =10, pady = 10,sticky='w')

		#buttonVarTheme=ttk.Button(self, text="Print files directory", command = controller.PrintFilesDir)
		#buttonVarTheme.grid(row=2, column = 4, padx = 10, pady = 10,sticky='w')

		buttonMenu = ttk.Button(self, text ="Menu",command = partial(controller.show_frame,Menu))
		buttonMenu.grid(row = 1, column = 1, padx = 10, pady = 10,sticky='w')

		# buttonStatistics = ttk.Button(self,text = "Statistics",state='disabled')
		# buttonStatistics.grid(row = 2, column = 1, padx = 10, pady = 10,sticky='w')

		buttonParameters = ttk.Button(self, text ="Parameters",command = partial(controller.show_frame,Parameters))
		buttonParameters.grid(row = 3, column = 1, padx = 10, pady = 10,sticky='w')

		buttonQuit = ttk.Button(self,text="Quit" , style = 'BW.TButton' ,command=controller.Close)
		buttonQuit.grid(row=4,column = 1, padx = 10 ,pady = 10 ,sticky='w')

# third window frame Parameters
class Parameters(tk.Frame):
	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		#label = ttk.Label(self, text ="Parameters", font = LARGEFONT)
		#label.grid(row = 0, column = 4, padx = 10, pady = 10,sticky='w')

		buttonParameters = ttk.Button(self, text ="Parameters",state='disabled')
		buttonParameters.grid(row = 2, column = 1, padx = 10, pady = 10,sticky='w')

		buttonStatistics = ttk.Button(self, text ="Statistics", command = partial(controller.show_frame, Statistics))
		buttonStatistics.grid(row = 2, column = 1, padx = 10, pady = 10,sticky='w')

		buttonMenu = ttk.Button(self, text ="Menu", command = partial(controller.show_frame, Menu))
		buttonMenu.grid(row = 1, column = 1, padx = 10, pady = 10,sticky='w')

		SaveFileFolderSave = tk.Text(self, height=1,width=25)
		SaveFileFolderSave.insert(tk.INSERT, controller.saveFolder)
		SaveFileFolderSave.grid(row = 2, column = 4, padx = 10, pady = 10,sticky='w')

		buttonSaveFolder = ttk.Button(self,text='Change Folder for saved files',command = partial(controller.changeSaveFolder,SaveFileFolderSave))
		buttonSaveFolder.grid(row = 1, column = 4, padx = 10, pady = 10,sticky='w')

		buttonQuit = ttk.Button(self,text="Quit" , style = 'BW.TButton' ,command=controller.Close)
		buttonQuit.grid(row=3,column = 1, padx = 10 ,pady = 10 ,sticky='w')

		label = ttk.Label(self, text ="Change Theme")
		label.grid(row = 3, column = 4, padx = 10, pady = 10,sticky='w')

		self.VarTheme = tk.StringVar(self)
		self.VarTheme.set(controller.listeOfThemes[0])
		buttonTheme = ttk.OptionMenu(self,self.VarTheme,*controller.listeOfThemes, command = controller.changeTheme)
		buttonTheme.grid(row=4,column = 4, padx = 10 ,pady = 10 ,sticky='w')	

class AnalysisWindow(tk.Toplevel):
	def __init__(self, controller):
		super().__init__(controller)
		buttonPlot=ttk.Button(self,text="Plot The number of interaction over time",command=partial(controller.pltGraphOutside,stats["numberOfinteractions"],np.arange(0,len(stats["numberOfinteractions"]),1),"Number of interaction"))
		buttonPlot.grid(row=0,column=3, padx=10,pady=10)
		
	def UpdateImage(self,image,row,column):
		try:
			img = ImageTk.PhotoImage(Image.open("Processed/"+ image).resize((500, 500)))
			labelTmp = ttk.Label(self,image = img)
			labelTmp.Image = img
			labelTmp.grid(row=row, column = column,padx = 10, pady = 10)
		except:
			pass

	def DisplayImage(self):
		if IMAGE_DISP:
			self.UpdateImage("Step_5_leukemic_cells_center_determination.PNG",0,0)
			self.UpdateImage("Step_7_t_cells_center_determination.PNG",1,0)
			#self.UpdateImage("Step_8_cell_superpositioning.PNG",0,1)
			#self.UpdateImage()
		self.after(10, self.DisplayImage)

	def DisplayImagesEnd(self):
		self.UpdateImage("Step_5_leukemic_cells_center_determination.PNG",0,0)
		self.UpdateImage("Step_7_t_cells_center_determination.PNG",1,0)
		self.UpdateImage("Step_8_cell_superpositioning.PNG",0,1)

def AnalysisExt(listeTCell, listeLeukemicCell,Analyzer,saveFolder):
	for imgTcell,imgLcell in zip (listeTCell, listeLeukemicCell):
		IMAGE_DISP = False
		interactionPositions, stat = Analyzer.getInteractionArray(imgLcell,imgTcell)
		stats["numberOfValidLeukemicCells"].append(stat["numberOfValidLeukemicCells"])
		stats["numberOfValidTCells"].append(stat["numberOfValidTCells"])
		stats["numberOfinteractions"].append(stat["numberOfinteractions"])
		stats["pourcentageOfValidLeukemicCells"].append(stat["pourcentageOfValidLeukemicCells"])
		stats["pourcentageOfValidTCells"].append(stat["pourcentageOfValidTCells"])
		stats["pourcentageOfInteractions"].append(stat["pourcentageOfInteractions"])
		IMAGE_DISP = True
	with open (saveFolder + "Statistics.csv" , "w", newline='') as excel:
		writer = csv.writer (excel, delimiter = ";")
		writer.writerow(stats.keys())
		writer.writerows(zip(*stats.values()))
	RUN_FLAG=False

def ProcessAnalysis(listeTCell, listeLeukemicCell,Analyzer,saveFolder):
	process1 = Thread(target=AnalysisExt, args=(listeTCell, listeLeukemicCell,Analyzer,saveFolder))
	process1.start()

if __name__ == "__main__":
	app = tkinterApp()
	app.mainloop()

