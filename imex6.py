# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:20:32 2018

@author: ovv
"""
import shutil
from tkinter import *
from PIL import Image, ImageTk, ExifTags, ImageFile, ImageGrab, ImageDraw
import pandas as pd
import numpy as np
#from tkintertable import TableCanvas
import math

# THESE MUST GO WHEN CONVERTING TO WINDOWS EXE
#import matplotlib.pyplot as plt
#from matplotlib.figure import Figure
#import matplotlib as mpl
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg
###### UNTIL HERE


#import networkx as nx
# import pickle
from tkinter import ttk, filedialog
import os
import io
import string
import random
import glob
#import torch
import copy
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import webbrowser
# from PIL.ExifTags import TAGS, GPSTAGS
from itertools import chain
import torch.utils.data as data
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from sklearn.neighbors import typedefs
#from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import sklearn.utils._cython_blas
#import sklearn.neighbors.quad_tree
# import sklearn.tree
# import sklearn.tree._utils
from scipy.spatial import distance
import umap
from scipy.sparse import lil_matrix
#from sklearn.decomposition import PCA
#for graphs
#import matplotlib
from collections import defaultdict
import functools
import h5py
import h5py.defs
import h5py.utils
import h5py.h5ac
import h5py._proxy
import warnings
#import cv2
import time
#import timm

import rasterfairy #from 2d point cloud to square grid. Need to change import rasterfairy.prime to import rasterfairy.prime as prime in the rasterfairy.py file. You also need to change np.float to float.
import bottleneck as bn
import exifread
#import clip ## only if model used is clip... 

batch_size  = 8
num_workers = 4
num_c = 0
tuu=-1
#image_= []
def numpy_unique_ordered(seq):
            array_unique = np.unique(seq, return_index=True)
            dstack = np.dstack(array_unique)
            dstack.dtype = np.dtype([('v', dstack.dtype), ('i', dstack.dtype)])
            dstack.sort(order='i', axis=1)
            return dstack.flatten()['v'].tolist()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
    #id generator for exporting images with the same name.

#Main app window
class Application(Frame,object):
    # Define settings upon initialization. Here you can specify
    def __init__(self, master):
        self.themainframe = super(Application,self).__init__(master)
        #some initial parameters
        self.refresh = 0            #keeps track of whether the graph has been updated recently or not
        self.im_list = []           #list with images
        #self.df = pd.DataFrame()    #clustering results
        self.df = np.array([])    #clustering results
        self.lijstje = []           #list with buckets
        self.theBuckets = []        #the buckets containing the images
        self.bucketDisp = 0         # is 1 when a bucket is displayed, 0 otherwise.
        self.allimages = []         #if images are Preloaded, this contains the images.
        self.num_clus = []          #the number of the current cluster
        #self.cm = []                #the array with correlations between images
        self.features= []           #the features of all the images extracted with the neural network
        self.catList = []           #similar to lijstje
        self.loaded_imgs = []       #contains all the actual image data, if preloaded
        self.imagex = []            #contains the images currently displayed
        self.preloaded = 0          #keeps track of whether images are preloaded or not
        self.meta_load = 0          #keeps track of whether the metatab has been loaded or not
        self.rectangles = []        #contains the squares to display currently selected images
        self.greentangles = []      #contains the squares to display current images that are already in a bucket
        self.selected_images = []   #list of selected images
        self.grid()
        self.master.title("IMEX")   #title of the app
#        self.image_distance = 10    #distance in pixels between images displayed
        self.focusclick = 0         #keeps track of creating the square on a focused image
        self.rectanglewidth = 3     #width of the red selection rectangles
        self.neuralnet = 'resnet152'#determines the neural network used.
        # self.neuralnet = 'clip'#determines the neural network used.
        self.subcyes = 0            # is 1 when subclustering, 0 otherwise
        self.tsneclick = 0          #keeps track of creating the square on tsne graph
        self.x_test = []            # variable for UMAP/TSNE embedding
        self.wt = 0
        self.selectedcat = []       #needed to make it possible to deselect item in category box
        self.video_cm = []          #in case the user adds videos
        self.zearray_reduced = []   ### array for image map
        self.log = []
        self.represent = []
        self.scrolled_images = 0
        self.totimseen = []
        self.ccluster = []
        self.oview = 0
        self.controlZ = []
        self.czcurrent = 0
        self.ctrlzused = 0
        self.meta_data = pd.DataFrame() # DataFrame to hold metadata
        self.selected_edge = None # Currently selected hyperedge
        self.hyperedge_canvases = [] # to store our hyperedge canvases of the second window
        self.list_of_colors = ['crimson', 'gold', 'royalblue', 'hotpink', 'turqoise', 'purple', 'lime', 'lightyellow' ,'forestgreen','silver']
        #currently available:
            #'resnet18'
            #'resnet152' << prefered option
            #'vgg16'
            #'vgg19'
            #'inceptionV-3'
            #'squeezenet'
            #'alexnet'
            #'densenet161' << UNTESTED
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure('.',background = '#555555',foreground = 'white') #sets background of the app to a dark theme
        style.map('.',background=[('disabled','#555555'),('active','#777777')], relief=[('!pressed','sunken'),('pressed','raised')])
        style.configure('TNotebook.Tab',background='#555555')
        style.map('TNotebook.Tab',background=[('selected','black')])
#

#        style.settings()
#        tabstyle = ttk.Style()
#        mygreen = "#d2ffd2"

#
#        tabstyle.theme_create( "yummy", parent="alt", settings={
#            "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0] } },
#            "TNotebook.Tab": {
#                "configure": {"padding": [5, 1], "background": mygreen },
#                "map":       {"background": [("selected", myred)],
#                              "expand": [("selected", [1, 1, 1, 0])] } } } )
#
#        tabstyle.theme_use("yummy")

        
        self.create_widget()

    def create_widget(self):
        
#        self.notebook = ttk.Notebook()
#         = ttk.Frame(self.notebook)
#        .configure(width=10000, height=10000)        
#        self.notebook.add(text='Cluster viewer')
#        self.notebook.place(x=0,y=1)
#        self.tab2 = ttk.Frame(self.notebook)
#        self.notebook.add(self.tab2,text='Metadata viewer')
        #########################################
        ######## ALL tab1 WIDGETS ###############
        #########################################        
        self.image_distance = 10

        # text displaying some useful tips and a welcome message
        self.communication_label = Message(font=12)
        self.communication_label['background'] = '#FFFFFF'
        self.communication_label['font'] = 12
        self.communication_label.place(x=1150,y=20)
        self.communication_label['width'] = 300
        self.communication_label.configure(text='Hello and welcome to IMEX. Start on the left by selecting a folder with images or by loading a previous session. By right clicking a button, you can see some additional information about what the button does.')
        
        #button explanation text. The user can right-click a button to get this explanation
        def open_folder_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can select the image folder containing the images you want to analyze. IMEX will look through the selected folder and all its subfolders for files with the following extentions: .jpg, .png, .bmp, .gif, .tif')
        def features_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button IMEX will extract the features of the images it found in the selected folder using a neural network. The features are a way to represent the content of an image.')
        def cluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='The features will be used generate clusters based on similarity. Images above the threshold will be placed in the same cluster. If clusters contain a lot of false positives, increasing threshold. If there are too many small clusters, try decreasing the threshold.')
        def load_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Load a previously saved session and continue where you stopped last time.')
        def addcategory_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Add a category bucket to structure the image collection with the name specified in the entry field above. You can add images to the buckets using the \'Add cluster to selected bucket(s)\' button, or the \'Add selection to selected bucket(s)\' button.')
        def showcluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Show a specified cluster number')
        def showbucket_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Show the content of the selected bucket(s). This also allows you to delete images from the buckets and in a future update, use the bucket to find more relevant instances.')
        def export_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Export the content of your buckets to a location of your choosing. A folder will be created for each bucket, and a copy of the original images will be placed in the folders.')
        def showimg_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button shows the content of the current cluster. You can set how many images you see in the entry box above this button.')
        def locateonmap_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Shows where an image is located on the image map.')
        def addcluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can add your current cluster to one or more category buckets that you have selected from the list in the middle of the screen. This adds the entire cluster to the bucket at once. If you want more control, you can make a selection, and use the \'Add selection to selected bucket(s)\' button.')
        def addselection_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can add your selected images to one or more category buckets that you have selected from the list in the middle of the screen.')
        def delete_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can delete selected images from a bucket. To do this you first need to use the \'Show bucket\' button, and it only works if you are viewing a single bucket.')
        def changename_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Change the name of the selected bucket to what you put in the entry field below this button')
        def numim_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Here you can set how many images you want to be shown at the same time. The default is 20. Set to zero if you want to view all images in a cluster or bucket. Note that it can take a moment to load all images when the number is high.')
        def vector_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Press this button to calculate a representative image for each cluster. This allows you to use the \'Show overview\' button, which shows a single representative image for each cluster.')
        def imagemap_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Calculates a map with all images, with similar images closer together. Hold shift and draw a shape with left mouse button to select images within the shape.')
        def macromap_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='A \'zoomed out\' version of the image map, use mousewheel on an image to zoom in to view the other images surrounding this image. If you zoom in enough you end up on the actual image map.')
        def overview_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Show the overview after calculating the overview. Displays a representative image for each cluster, which allows you to quickly see relevant clusters. After you have selected one or more clusters, press the \'Show selected cluster\' button to view the cluster.')
        def selectcluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Press this button to view the cluster you selected from the overview. You can always return to the overview by pressing the \'Show overview\' button again.')
        def focus_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='If only a part of the image is of particular interest, you can click this button to enlarge the image. Next, you can draw a square around the part, and press the \'Query selection\' button.')
        def query_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='All images in your collection will be ranked on similarity to the part you selected after pressing the \'Focus image\' button.')
        def rankexternal_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Select an image on your computer. All images in your collection will be ranked on similarity to the specified image. This image does not have to be part of the image collection.')
        def rankbucket_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='All images in your collection will be ranked on similarity to the images in the selected bucket. Makes use of an SVM.')
        def rank_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='All images in your collection will be ranked on similarity to the selected image.')
        def save_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button will save your current session. The following things are saved: the list of images from the folder you selected, the buckets you created and the images you added to the buckets, and the cluster you are currently at.')
        def preload_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button will load thumbnails of all the images in the folder you selected. This will greatly increase the speed of displaying images, and will result in a more pleasurable experience. However, preloading all images may take some time initially.')
        def nonrelevant_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='If you have added images to the \'NonRelevant\' bucket, you can check this box to make sure the images are not displayed anymore, for example when viewing clusters, or when ranking images.')
        def inbucket_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Check this box to hide all images that you already placed in a bucket. This is useful for example if you are are ranking images, and only want to find new instances of images not yet in a bucket.')
        def filter_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='By selecting one or more buckets from the left list, only images present in all of the selected buckets will be shown. Additionally, by selecting a bucket on the right, images in those buckets will not be shown. The lists function as AND (left) and NOT (right).')
        def sankey_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button a sankey graph can be created. For each bucket, it will show which fraction of the images is also present in other buckets.')
        def sorting_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Sort all clusters from most to least relevant based on the images in the currently viewed bucket')
        def restore_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Undo sorting of clusters, restoring the original order')
        def selectvideo_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='WIP, rudimentary version currently implemented. Select a folder containing videos to add to the image collection. Use \'Extract video\' after. ')
        def exctractvideo_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='WIP, rudimentary version currently implemented. Every key frame for each video will be extracted, then within the video these will be clustered. A representative image of each cluster will be added to the clusters in the image collection.')
        def sourcecluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Go to the source cluster of the currently selected image.')
        def prev_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Go to previously shown images.')
        def next_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Go forward in shown images.')
        def newimgs_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Add new images to existing image collection. Currently only works by adding a premade txt files with image locations.')
        def get_metadata_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Retrieve metadata (EXIF) from all images.')
        def display_metadata__text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Open table with selected metadata in new window.')
        def add_metadata_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Select metadata types to add to the metadata table for viewing.')
        def display_metaimages_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Sort the metadata in the table (e.g., by DateTimeOriginal). Select a row in the table to display images from that point onward.')
        
        #button to load
        self.lb = Button(background='#445544',foreground='white',width='20',relief='solid',bd=1)
        self.lb['text'] ="Load session"
        self.lb['command'] = lambda:[self.load_as(),self.display_hyperedges()]
        self.lb.bind('<Button-3>',load_text)
        self.lb.place(x=200,y=20)


               
        #button to show or hide images in the Non-relevant bucket
        vv = IntVar()
        self.nonR = ttk.Checkbutton(variable = vv)
        self.nonR['text'] ="Check to hide Non-relevant images from displayed results"
        self.nonR.bind('<Button-3>',nonrelevant_text)        
        self.nonR.var = vv
        self.nonR.place(x=400,y=220)
        
        #button to show or hide images already in a bucket
        ww = IntVar()
        self.inbucket = ttk.Checkbutton(variable = ww)
        self.inbucket['text'] ="Check to hide images already in a bucket"
        self.inbucket.bind('<Button-3>',inbucket_text)        
        self.inbucket.var = ww
        self.inbucket.place(x=400,y=240)

        #enter number of images shown
        self.e1 = Entry(background='#777777',foreground = 'white',exportselection=0,width=24)
        self.e1.insert(END, 800) #800 is default, change if needed. 
        self.e1.bind('<Button-3>',numim_text)        
        self.e1.place(x=200,y=250)
        self.var = IntVar()
        self.var.set(0)

        #button to show currently selected bucket
        self.b4 = Button(background='#443344',foreground='white',width='20',relief='solid',bd=1)
        self.b4['text'] ="Show hyperedge"
        self.b4.bind('<Button-3>',showbucket_text)        
        self.b4['command'] = self.showEdge
        self.b4.place(x=400,y=120)


        ### some other stuff ####
        #creates window for statistics and other data
        self.newWindow = Toplevel(self.master)
        self.newWindow.geometry("1400x500")
        self.newWindow.title("Hyperedges")
        self.newWindow.configure(background='#555555')
        #create dataframe for the bucket
        self.theBuckets = {}
        #list of available category buckets
        valores = StringVar()
        valores.set("RelevantItems Non-RelevantItems")
        self.theBuckets["RelevantItems"] = []
        self.theBuckets["Non-RelevantItems"] = []
        #scrollbar for the buckets
        boxscrollbar = Scrollbar(width = 10)
        #listbox containing the names of the buckets
        self.categories = Listbox(width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar.set,exportselection=0)
        self.categories['listvariable'] = valores
        self.categories['selectmode'] = 'extended'               
        self.categories.place(x=555,y=50)    
        #place of the scrollbar
        boxscrollbar.config(command=self.categories.yview)
        boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
        self.categories.bind('<Button-1>', self.deselect_list )
        self.categories.bind("<Double-Button-1>", self.double_click_bucket)
        #filter the self.categories listbox
        self.search_var = StringVar()
        self.search_var.trace_add("write", self.update_the_list)
        self.filter_entry = Entry(background='#777777',foreground = 'white', textvariable=self.search_var, width=30)
        self.filter_entry.place(x=555,y=30)
        self.filter_cats_label = Label(width=20,background='#555555',foreground='white',relief='flat',justify=LEFT, anchor='w')
        self.filter_cats_label['text'] = 'Filter the buckets'
        self.filter_cats_label.place(x=555,y=10)
        
#        #entryform to change name of a bucket
#        self.changename = Entry(background='#777777',foreground = 'white', width=30)
#        self.changename.place(x=0, y=0)
        #button to change name of a bucket
        self.changebutton = Button(background='#443344',foreground='white',width='20',relief='solid',bd=1)
        self.changebutton['text'] = "Change bucket name"
        self.changebutton['command'] = self.change_name
        self.changebutton.bind('<Button-3>',changename_text)
        self.changebutton.place(x=400, y=20)
        
        
        #enter the size of the images displayed. 100 is default
        self.imsize = 100
        self.imsize_entry = Entry(background='#777777',foreground = 'white',exportselection=0)
        self.imsize_entry.insert(END, 100)
        self.imsize_entry.place(x=1360,y=270)
        self.imsize_entry['validate'] = 'focusout'
        self.imsize_entry['validatecommand'] = self.get_imsize     
        
        self.set_imsize_label = Label(background='#555555',foreground='white')
        self.set_imsize_label['text'] = 'Set the image display\n size in pixels'
        self.set_imsize_label['wraplength'] = 200
        self.set_imsize_label['justify'] = CENTER
        self.set_imsize_label.place(x=1360,y=230)
               
        #canvas for the images, which adjusts to screen width. Optimized for windows. For sidebar, like in linux, you may want to decrease screen width
        self.screen_width = root.winfo_screenwidth() #requests screen width
        self.screen_width = self.screen_width-60
        self.screen_height = root.winfo_screenheight() #requests screen height
        self.screen_height = self.screen_height-400
        #yscrollbar = Scrollbar(width = 16, orient="vertical") #scroll  bar for canvas
        #xscrollbar = Scrollbar(width = 16, orient="horizontal") #scroll  bar for canvas
        #yscrollbar = Scrollbar(orient="vertical") #scroll  bar for canvas
        #xscrollbar = Scrollbar(orient="horizontal") #scroll  bar for canvas
        
        #self.c = Canvas(bg='#666666',bd=0, scrollregion=(0, 0, 0, 500), yscrollcommand=yscrollbar.set,xscrollcommand=xscrollbar.set, width =self.screen_width, height =self.screen_height) #canvas size
        self.c = Canvas(bg='#666666',bd=0, scrollregion=(0, 0, 0, 500), width =self.screen_width, height =self.screen_height) #canvas size
        yscrollbar = Scrollbar(orient="vertical", command=self.c.yview)
        xscrollbar = Scrollbar(orient="horizontal", command=self.c.xview)        
        self.c.my_tag = 'c'

        self.c.place(x = 0, y=300)
        #yscrollbar.config(command=self.c.yview)
        #xscrollbar.config(command=self.c.xview)
        
        self.c.configure(yscrollcommand=yscrollbar.set, xscrollcommand=xscrollbar.set)
        
        #yscrollbar.place(in_=self.c,relx=1.0, relheight=1)
        #xscrollbar.place(in_=self.c,relx=1.0, relheight=1)
        yscrollbar.place(in_=self.c,relx=1.0,relheight=1)
        xscrollbar.place(in_=self.c,rely=1.0,relwidth=1)
        
        self.num_im_row = math.floor(self.screen_width / (self.imsize + self.image_distance)) #the total number of images that fit from left to right
        self.num_im_col = math.floor(self.screen_height / (self.imsize + self.image_distance)) #the total number of images that fit from top to bottom
        #binds scrollwheel when scrolling with mouse on canvas
        self.c.bind('<Enter>', self._bound_to_mousewheel)
        self.c.bind('<Leave>', self._unbound_to_mousewheel)
        self.c.bind('<Button-2>', self.open_image3)
        self.c.bind("<Button-1>", self.click_select)
        self.c.bind("<Shift-Button-1>", self.shift_click_select)
        self.c.bind("<Control-Button-1>", self.ctrl_click_select)
        self.c.bind("<Button-3>", self.rank_images)
        self.c.bind("<Double-Button-1>", self.double_click_overview)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.origX = self.c.xview()[0]
        self.origY = self.c.yview()[0]

        
        
        
       
    def confirm_remove_images(self):
        confirm = messagebox.askokcancel("Confirmation", "Make sure to backup the save file first. Are you sure you want to remove images? This action cannot be undone.")
        if confirm:
            self.remove_images()
            
        
        

    #this function ranks a selected image by rightclicking an image. It will sort all images based on correlation
    def rank_images(self, event):
        self.communication_label.configure(text='Calculating the ranking. Please wait.')
        self.communication_label['background'] = '#99CCFF'
        self.communication_label.update_idletasks()
        evex = self.c.canvasx(event.x)   #x location to determine selected image
        evey = self.c.canvasy(event.y)   #y location to determine selected image
        self.bucketDisp = 2
        x_num = math.ceil((evex)/(self.imsize + self.image_distance))-1 #determine the row
        y_num = math.ceil((evey)/(self.imsize + self.image_distance))   #determine the column
        im_num = x_num + self.num_im_row*(y_num-1) #calculate the actual image number using row and column
        im_tag = self.c.gettags(self.imagex[im_num])    #get the actual image id from imagex (imagex is a list of all currently displayed images)
        im_tag = int(float(im_tag[0]))
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'query image')
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(int(im_tag)))        
        with h5py.File(self.hdf_path, 'r') as hdf:
            self.rank_list = np.array(hdf.get('cm')[int(im_tag),:])
        if len(self.video_cm)>0:
            self.rank_list = np.hstack((self.rank_list,self.video_cm[int(im_tag),:]))
        temp_list = np.sort(self.rank_list,0)[::-1] #sorts the correlations
        self.rank_list = np.argsort(self.rank_list,0)[::-1] #sorts the id of all the images based on correlation
        self.c.xview_moveto(self.origX)  #####
        self.c.yview_moveto(self.origY) ######
        self.rank_list = np.asarray(self.rank_list)
        temp_list = np.asarray(temp_list)
        temp_list[np.isnan(temp_list)] = -100   #some images have no correlations due to issues with loading and extracting the image features. Ususally means the image file is damaged
        self.rank_list = self.rank_list[temp_list>-50] # this removes all the broken images
        self.rank_list = np.append(im_tag,self.rank_list) # this adds the selected (queried) image to the image list to be displayed
        self.c.delete("all")
        self.display_images(self.rank_list) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets    
        self.communication_label['background'] = '#FFFFFF'
        self.communication_label.configure(text='Finished calculating. Showing the ranking')



    def deselect_list(self, event):
        if len(self.selectedcat) == 0:
            self.selectedcat = self.categories.curselection()
        if len(self.selectedcat) == 1:
            if self.categories.curselection() == self.selectedcat:
                self.categories.selection_clear(0,END)
                self.selectedcat = []        
        
    def _bound_to_mousewheel(self,event):
        self.c.bind_all("<MouseWheel>", self._on_mousewheel)
            
    def _unbound_to_mousewheel(self, event):
        self.c.unbind_all("<MouseWheel>") 

    def _bound_to_mousewheel_second_window(self,event):
        event.widget.bind_all("<MouseWheel>", self._on_mousewheel_second_window)
            
    def _unbound_to_mousewheel_second_window(self, event):
        event.widget.unbind_all("<MouseWheel>")

    def update_the_list(self,*args):
        search_term = self.search_var.get()

        self.categories.delete(0, END)
        
        for item in self.catList:
                if search_term.lower() in item.lower():
                    self.categories.insert(END, item)
    
    def open_image3(self,event):
        canvas = event.widget
        evex = canvas.canvasx(event.x)
        evey = canvas.canvasy(event.y)
        self.item = canvas.find_overlapping(evex, evey, evex, evey)
        im_tag = int(canvas.gettags(self.item)[0])
        mijn_plaatje = self.im_list[im_tag]
        webbrowser.open(mijn_plaatje)
#
    
    def click_select(self,event):
        if self.bucketDisp == 5:
            self.draw_focus_square(event)
        else:
            self.select_image(event)
    
    def draw_focus_square(self, event):
        self.focusclick = self.focusclick + 1
        if self.focusclick == 2:
            self.evex2 = self.c.canvasx(event.x)
            self.evey2 = self.c.canvasy(event.y)
            self.focusclick = 0
            if self.squares is not None:
                self.c.delete(self.squares)
            self.squares = self.c.create_rectangle(self.evex1, self.evey1, self.evex2, self.evey2)
        else:
            self.evex1 = self.c.canvasx(event.x)
            self.evey1 = self.c.canvasy(event.y)
    
    def select_image(self, event):
        canvas = event.widget
        evex = canvas.canvasx(event.x)
        evey = canvas.canvasy(event.y)
        self.selected_images = []
        # Traverse up to find the canvas
        while not isinstance(canvas, Canvas):
            canvas = canvas.master
        # Now you can access the canvas and its tags
        self.canvas_tag = getattr(canvas, 'my_tag', None)
        self.item = event.widget.find_overlapping(evex, evey, evex, evey)
        
        if self.canvas_tag is not None:
            self.c.delete('rect_tag')
            for canv in self.hyperedge_canvases:
                canv.delete('rect_tag')
            bbox = canvas.bbox(self.item)
            
            # Draw a rectangle around the image to highlight it
            rect = canvas.create_rectangle(
                bbox,
                outline='red',
                width=2,
                tags='rect_tag'
            )
            self.rectangles = [rect]
            self.selected_images.append(canvas.gettags(self.item)[0])

    # this allows you to select multiple adjacent images using the shift key + mouse button 1
    def shift_click_select(self,event):
        canvas = event.widget
        evex = canvas.canvasx(event.x)
        evey = canvas.canvasy(event.y)
        
        # Traverse up to find the canvas
        while not isinstance(canvas, Canvas):
            canvas = canvas.master
        # Now you can access the canvas and its tags
        self.canvas_tag2 = getattr(canvas, 'my_tag', None)
        self.item2 = event.widget.find_overlapping(evex, evey, evex, evey)
        
        if self.canvas_tag2 == self.canvas_tag:
            self.selected_images = []
            self.rectangles = []
            maxit = np.max([self.item[0],self.item2[0]]) + 1
            minit = np.min([self.item[0],self.item2[0]])
            for idx in range(minit,maxit):
                bbox = canvas.bbox(idx)
                rect = canvas.create_rectangle(
                    bbox,
                    outline='red',
                    width=2,
                    tags='rect_tag'
                )
                self.rectangles.append(rect)
                self.selected_images.append(canvas.gettags(idx)[0])

    #this function allows you to select multiple non-adjacent images by hold down control + left click
    def ctrl_click_select(self,event):
        canvas = event.widget
        evex = canvas.canvasx(event.x)
        evey = canvas.canvasy(event.y)
        # Traverse up to find the canvas
        while not isinstance(canvas, Canvas):
            canvas = canvas.master
        # Now you can access the canvas and its tags
        self.canvas_tag = getattr(canvas, 'my_tag', None)
        self.item = event.widget.find_overlapping(evex, evey, evex, evey)
        if canvas.gettags(self.item)[0] in self.selected_images:
            item_to_remove = self.selected_images.index(canvas.gettags(self.item)[0])
            canvas.delete(self.rectangles[item_to_remove])
            del self.selected_images[item_to_remove]
            del self.rectangles[item_to_remove]
            
        else:
            self.item = event.widget.find_overlapping(evex, evey, evex, evey)
            
            if self.canvas_tag is not None:
                bbox = canvas.bbox(self.item)
                
                # Draw a rectangle around the image to highlight it
                rect = canvas.create_rectangle(
                    bbox,
                    outline='red',
                    width=2,
                    tags='rect_tag'
                )
                self.rectangles.append(rect)
                self.selected_images.append(canvas.gettags(self.item)[0])

        
    
    def double_click_overview(self,event):
        if self.oview == 1:
            self.show_selected_cluster()

    def double_click_bucket(self, event):
        self.showBucket()

    # def remove_elements(self, elements):
    #     xx = list(self.ccluster)
    #     self.ccluster = np.array(list(sorted(set(self.ccluster) - set(elements), key=xx.index)))

    def remove_elements(self, elements):
        self.ccluster = np.array([x for x in self.ccluster if x not in elements])
                       
    #the function that is called by other functions in order to display images
    def display_images(self, cluster,input_origin=None): #x is the list with the image names, cluster is the list with ids. 
        
        self.c.unbind("<ButtonRelease-1>")
        self.c.unbind("<B1-Motion>")
        # self.c.unbind("<Shift-Button-1>")
        self.c.unbind("<Shift-B1-Motion>")
        self.c.unbind("<Shift-ButtonRelease-1>")
        # self.c.unbind("<Button-1>")
        
        # try:
        #     self.seenX = np.max(self.totimseen)
        #     self.totimseen = []
        # except AttributeError:
        #     pass
        # except ValueError:
        #     if len(self.ccluster) > self.num_im_row*self.num_im_col:
        #            self.seenX = self.num_im_row*self.num_im_col
        #     else:
        #         self.seenX = len(self.ccluster)
                
        if len(cluster) == 0:
            self.communication_label.configure(text='No images to show')
            return
        self.ccluster = cluster
        # if input_origin == 'overview':
        #     self.oview = 1
        #     # self.bO3["state"] = 'normal'
        # else:
        #     self.oview = 0
        #     # self.bO3["state"] = 'disabled'
       
        self.num_im_row = math.floor(self.screen_width / (self.imsize + self.image_distance)) #the total number of images that fit from left to right


        
        if self.bucketDisp != 1 and (self.nonR.var.get() == 1 or (self.inbucket.var.get() == 1 and self.oview != 1)):
            # Initialize the set of elements to be removed
            t = set()
        
            # If nonR is selected, add non-relevant items to the removal set
            if self.nonR.var.get() == 1:
                t.update(self.theBuckets['Non-RelevantItems'])
        
            # If inbucket is selected and not in overview mode, add all bucketed items to the removal set
            if self.inbucket.var.get() == 1 and self.oview != 1:
                t.update(chain.from_iterable(self.theBuckets.values()))        
            self.remove_elements(t)


                

        num_im =int(self.e1.get())
        if num_im == 0:
            num_im = len(self.im_list)
        if num_im > 4950:
            num_im = 4950
        if num_im > len(cluster):
            num_im = len(cluster)
        else:
            self.ccluster = self.ccluster[0:num_im]
        x = []
        for ij in range(0,len(self.ccluster)):
            x.append(self.im_list[self.ccluster[ij]])
        self.imagex = []
        self.c['scrollregion'] = (0,0,0,math.ceil(len(x)/self.num_im_row)*(self.imsize+self.image_distance))
        self.c.delete("all")
        # self.greentangles = []
        # self.purpletangles = []
        # if self.bucketDisp == 5:
        #     self.bucketDisp == 0

            
        
        try: #### this fixes a memory leak by deleting currently loaded images in the memory.
            for uut in range(0,len(self.my_img)):
                self.my_img[uut].destroy()
        except AttributeError:
            pass
        self.my_img = []
        with h5py.File(self.hdf_path, 'r') as hdf:
            for j in range(0,len(x)):

                #load = Image.open(x[j])
                #test = np.asarray(hdf.get('thumbnails')[1],dtype='uint8')
                load = Image.fromarray(np.array(hdf.get('thumbnail_images')[self.ccluster[j]],dtype='uint8'))
                render = ImageTk.PhotoImage(load)
                self.my_img.append(render)
                row_ = j // self.num_im_row
                column_ = j % self.num_im_row
                x_pos = column_ * (self.imsize + self.image_distance) + (self.imsize / 2)
                y_pos = row_ * (self.imsize + self.image_distance) + (self.imsize / 2)


                # self.my_img.append([])
                # self.my_img[j] = Label(self.c,background='#555555')
                # self.my_img[j].image = render
                # row_ = math.floor(j/self.num_im_row)
                # column_ = j%self.num_im_row
                self.imagex.append(self.c.create_image(x_pos,y_pos, image=render, tags=self.ccluster[j]))
                
                
#                self.my_img[j].destroy()
                # if self.bucketDisp != 1:
                #     if int(self.ccluster[j]) in [xx for vv in self.theBuckets.values() for xx in vv]:
                #             self.greentangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='cyan2',width=self.rectanglewidth,tags = self.ccluster[j]))                    
                # if self.subcyes == 1:
                #     if int(self.ccluster[j]) in self.sel_im:
                #         self.purpletangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='blueviolet',width=self.rectanglewidth,tags = self.ccluster[j]))
                # if self.oview == 1:
                #     txtov = self.c.create_text(column_*(self.imsize + self.image_distance)+ (self.imsize / 2),row_*(self.imsize + self.image_distance)+ self.imsize - (self.image_distance/2) ,anchor='nw',fill="white",font="Calibri 8",
                #                        text=str(len(np.where(self.df[:,j]>-1)[0])))
                #     txtbb = self.c.create_rectangle(self.c.bbox(txtov),fill="#666666",outline="");self.c.tag_lower(txtbb,txtov)
                # self.c.update()
            # self.row_ = row_
            # self.column_ = column_
            
        #     if self.ctrlzused == 1: 
        #         self.ctrlzused = 0
        #     else:
        #         self.czcurrent = -1

        #         if len(self.controlZ) < 10:                
        #             self.controlZ.append(cluster)
        #         else:
        #             del self.controlZ[0]
        #             self.controlZ.append(cluster)                    
        # self.subcyes = 0
        
    def _on_mousewheel(self, event):
        self.c.yview_scroll((int(-1*event.delta/120)), "units")
        self.scrolled_images = self.c.yview()
        
        if self.scrolled_images[1] == 1:
            self.totimseen.append(len(self.ccluster))
        else:
            num_above = int(self.scrolled_images[0]*self.row_)*self.num_im_row
            if len(self.ccluster)- num_above < self.num_im_row*self.num_im_col:
                num_below = len(self.ccluster)- num_above
            else:
                num_below = self.num_im_row*self.num_im_col
            self.totimseen.append(num_above+num_below)
         
        
    def _on_mousewheel_second_window(self, event):
        # Windows only
        event.widget.yview_scroll(int(-1*(event.delta/120)), "units")
        
        
    #function to update the image size if the user changes it
    def get_imsize(self):
        self.imsize = self.imsize_entry.get()
        try:
            self.imsize = int(self.imsize)
            if int(self.imsize) > 9:
                self.communication_label.configure(text='Image size set to ' + str(self.imsize))
                self.communication_label['background'] = '#FFFFFF'
                self.num_im_row = math.floor(self.screen_width / (self.imsize + self.image_distance))
                return True
            else:
                self.communication_label.configure(text='Image size needs to be 10 or higher')
                self.communication_label['background'] = '#FE9696'
                self.imsize = 10
                return True
            
        except ValueError:
            self.communication_label.configure(text='Please enter an integer!')
            self.communication_label['background'] = '#FE9696'
            return False

    #function to delete selected images from a bucket
    def click_del(self):
        if len(self.BucketSel) == 1:
            self.catList = self.categories.get(0,END)
            if self.bucketDisp == 1:
                for g in range(0,len(self.selected_images)):
                    self.c.delete(self.imagex[self.selected_images[g]])
                    self.current_bucket[self.selected_images[g]] = -1
                self.current_bucket = np.array(self.current_bucket)
                self.current_bucket = self.current_bucket[self.current_bucket > -1]
                self.theBuckets[self.catList[int(self.BucketSel[0])]] = self.current_bucket


    #function to focus on a selected image. From here you can draw a square to select a part of an image to compare against all other images                   
    def focus_image(self):
        if len(self.selected_images) == 1:
            self.c['scrollregion'] = (0,0,0,800)
            self.squares = None
            im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
            self.bucketDisp = 5
            self.c.delete("all")
            self.focused_image = []
            d = self.im_list[int(im_tag[0])]
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'focused image')
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(int(im_tag[0])))
            
            self.focused_image.append(d)
            for j in range(0,len(self.focused_image)):
                load = Image.open(self.focused_image[j])
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(load._getexif().items())
                
                    if exif[orientation] == 3:
                        load=load.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        load=load.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        load=load.rotate(90, expand=True)                
                except AttributeError:
                    pass
                except KeyError:
                    pass

                load = load.resize((800,800))
                render = ImageTk.PhotoImage(load)
            # labels can be text or images
                my_img = Label(self,background='#555555')#, image=render)
                my_img.image = render
                #my_img.grid()
                #image_.append(my_img)
                row_ = math.floor(j/self.num_im_row)
                column_ = j%self.num_im_row
                #image_[j].grid(row = row_, column = column_)
                self.imagex.append(self.c.create_image(column_*(800 + self.image_distance)+ (800 / 2),row_*(800 + self.image_distance)+ (800 / 2), image =render,tags=im_tag))
                #if int(self.rank_list[j]) in self.theBuckets.values():
                #    self.greentangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance), outline='cyan2',width=5,tags = self.selected_images[j]))
    #Same function as rank images, but with a button on the GUI
    def rank_image(self):
        if len(self.selected_images) == 1:
            #self.c['scrollregion'] = (0,0,0,600)
            #self.squares = None
            im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
            self.bucketDisp = 2
    
            im_tag = int(float(im_tag[0]))
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'query image')
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(im_tag))
            
            with h5py.File(self.hdf_path, 'r') as hdf:
                self.rank_list = np.array(hdf.get('cm')[int(im_tag),:])            
            #self.rank_list = self.cm[:,int(im_tag)] # get all the correlations between the selected image and all the other images
            if len(self.video_cm)>0:
                self.rank_list = np.hstack((self.rank_list,self.video_cm[int(im_tag),:]))
            temp_list = np.sort(self.rank_list,0)[::-1] #sorts the correlations
            self.rank_list = np.argsort(self.rank_list,0)[::-1] #sorts the id of all the images based on correlation
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.rank_list = np.asarray(self.rank_list)
            temp_list = np.asarray(temp_list)
            temp_list[np.isnan(temp_list)] = -100   #some images have no correlations due to issues with loading and extracting the image features. Ususally means the image file is damaged
            self.rank_list = self.rank_list[temp_list>-50] # this removes all the broken images
            self.rank_list = np.append(im_tag,self.rank_list) # this adds the selected (queried) image to the image list to be displayed
            #self.bucketDisp = 0
            self.c.delete("all")
            self.display_images(self.rank_list) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets

            self.communication_label['background'] = '#FFFFFF'
            self.communication_label.configure(text='Finished calculating. Showing the ranking')
            #plt.close("all")

    def rank_bucket(self):
        if self.bucketDisp == 1:
            self.bucketDisp = 0
            cur_buck = np.sort(self.current_bucket)
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'rank bucket')
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(cur_buck))
            with h5py.File(self.hdf_path, 'r') as hdf:
                cmlen = hdf.get('cm').len()
                cur_buck = cur_buck[cur_buck<cmlen]
                bucket_cm = np.array(hdf.get('cm')[cur_buck])
            
            #bucket_cm = self.cm[self.current_bucket]
            
            bucket_cm2 = bucket_cm.reshape((-1,1))
            super_index = np.arange(0,len(self.im_list))
            super_index = np.resize(super_index,len(super_index)*len(cur_buck))
            #super_index = super_index[np.flip(np.argsort(np.nan_to_num(bucket_cm2),0))]
            super_index = pd.unique(super_index[np.flip(np.argsort(np.nan_to_num(bucket_cm2),0))].squeeze())
            
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.rank_list = super_index
            self.c.delete("all")
            self.display_images(self.rank_list) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets

            self.communication_label['background'] = '#FFFFFF'
            self.communication_label.configure(text='Finished calculating. Showing the ranking')

    def source_cluster(self):
        try:
            if len(self.selected_images) == 1:
                #self.c['scrollregion'] = (0,0,0,600)
                #self.squares = None
                im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
                self.bucketDisp = 2
        
                im_tag = int(float(im_tag[0]))            
                
                #s_cluster = self.df[self.df.isin([im_tag])].stack().keys()[0][1]
                s_cluster = np.where(self.df == im_tag)[1]
                self.bucketDisp = 0
                self.imagex = []
                self.c.delete("all")
                num_c = s_cluster
                #df = pd.read_csv('forAppOID.csv', header=None) 
                #df=df-1 #
                self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'source cluster')                
                self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'cluster number' + str(num_c))

                try:
                    cluster = self.df[:,num_c]
                    cluster = cluster[cluster > -1]
                    self.currentcluster = cluster
                    self.greentangles = []
                    self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters. This cluster contains ' + str(len(cluster)) + ' images.')
                except KeyError:
                    cluster = []         
                self.display_images(cluster)
        except TypeError:
            im_tag = self.selected_images
            #s_cluster = self.df[self.df.isin([im_tag])].stack().keys()[0][1]
            s_cluster = np.where(self.df == im_tag)[1]
            self.bucketDisp = 0
            self.imagex = []
            self.c.delete("all")
            num_c = s_cluster
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'source cluster')                
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'cluster number' + str(num_c))
            #df = pd.read_csv('forAppOID.csv', header=None) 
            #df=df-1 #
            try:
                cluster = self.df[:,num_c]
                cluster = cluster[cluster > -1]
                self.currentcluster = cluster
                self.greentangles = []
                self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters. This cluster contains ' + str(len(cluster)) + ' images.')
            except KeyError:
                cluster = []
            self.display_images(cluster)

    def display_cluster_info(self):
        try:
            if len(self.selected_images) == 1:
                im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
                im_tag = int(float(im_tag[0]))
                s_cluster = np.where(self.df == im_tag)[1]
    
                # Print the cluster number
                cluster_num = s_cluster[0] if s_cluster.size > 0 else None
                if cluster_num is not None:
                    cluster_info = f'Image is in cluster number: {cluster_num}.'
    
                    # Check which buckets the image is in
                    buckets = [bucket for bucket, images in self.theBuckets.items() if im_tag in images]
                    if buckets:
                        bucket_info = f' The image is also in buckets: {", ".join(buckets)}.'
                    else:
                        bucket_info = ' The image is not in any buckets.'
                    
                    self.communication_label.configure(text=cluster_info + bucket_info)
                else:
                    self.communication_label.configure(text='Cluster number not found.')
        except TypeError:
            self.communication_label.configure(text='Error in processing the selected image.')                

    def vcorrcoef(X,y):
        Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
        ym = np.mean(y)
        r_num = np.sum((X-Xm)*(y-ym),axis=1)
        r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
        r = r_num/r_den
        return r


    # This function shows a filtered selection of buckets.                   
    # def filter_buckets(self):

    #     def find_dubs(L):
    #         seen = set()
    #         seen2 = set()
    #         seen_add = seen.add
    #         seen2_add = seen2.add
    #         for item in L:
    #             if item in seen:
    #                 seen2_add(item)
    #             else:
    #                 seen_add(item)
    #         return list(seen2)
        
    #     self.cluster_in = []
    #     self.cluster_out = []
    #     for i in range(0,len(self.filter_in)):
    #         print('filter_in', self.filter_in[i],'catget',self.categories.get(self.filter_in[i]),'bucket',self.theBuckets[self.categories.get(self.filter_in[i])])
    #         self.cluster_in.append(self.theBuckets[self.categories.get(self.filter_in[i])])
    #     self.cluster_in = list(chain.from_iterable(self.cluster_in))
    #     if len(self.filter_in)> 1:
    #         self.cluster_in = find_dubs(self.cluster_in)
    #     try:
    #         len(self.filter_out)
    #     except AttributeError:
    #         self.filter_out = []
    #     for i in range(0,len(self.filter_out)):
    #             self.cluster_out.append(self.theBuckets[self.categories.get(self.filter_out[i])])
    #     self.cluster_out = list(chain.from_iterable(self.cluster_out)) 
    #     self.filtered_bucket = np.setdiff1d(self.cluster_in,self.cluster_out)
    #     self.imagex = []
    #     self.c.delete("all")
    #     self.display_images(self.filtered_bucket)
    #     self.filter_in = []
    #     self.filter_out = []
            #self.cluster = self.cluster[np.nonzero(self.cluster)]
    def filter_buckets(self):
    
        def find_intersection(lists):
            # Start with the first list's set
            result = set(lists[0])
            # Intersect with the subsequent sets
            for lst in lists[1:]:
                result.intersection_update(lst)
            return list(result)
        
        self.cluster_in = []
        self.cluster_out = []
        
        # Gather the lists for filter_in
        filter_in_names = []
        for i in range(len(self.filter_in)):
            bucket_name = self.categories.get(self.filter_in[i])
            filter_in_names.append(bucket_name)
            self.cluster_in.append(self.theBuckets[bucket_name])
        
        # Find the intersection if there are multiple lists
        if len(self.filter_in) > 1:
            self.cluster_in = find_intersection(self.cluster_in)
        else:
            self.cluster_in = list(chain.from_iterable(self.cluster_in))
        
        # Initialize filter_out if not already done
        if not hasattr(self, 'filter_out'):
            self.filter_out = []
    
        # Gather the lists for filter_out
        filter_out_names = []
        for i in range(len(self.filter_out)):
            bucket_name = self.categories.get(self.filter_out[i])
            filter_out_names.append(bucket_name)
            self.cluster_out.append(self.theBuckets[bucket_name])
        
        # Flatten the cluster_out list
        self.cluster_out = list(chain.from_iterable(self.cluster_out)) 
    
        # Get the difference between cluster_in and cluster_out
        self.filtered_bucket = list(set(self.cluster_in) - set(self.cluster_out))
        filter_in_text = ', '.join(filter_in_names)
        filter_out_text = ', '.join(filter_out_names)
        self.communication_label.configure(text=f'Used buckets: {filter_in_text}. Excluded buckets: {filter_out_text}.')
        self.imagex = []
        self.c.delete("all")
        self.display_images(self.filtered_bucket)
        
        # Reset the filters
        self.filter_in = []
        self.filter_out = []

    def clear_filter(self):
        self.categories2.selection_clear(0, END)
        self.categories3.selection_clear(0, END)

        
    # This function compares the drawn square to all other images                   
    def query_selection(self):
        load = Image.open(self.focused_image[0])
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'query selected square')                
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(self.focused_image[0]))
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            exif=dict(load._getexif().items())
        
            if exif[orientation] == 3:
                load=load.rotate(180, expand=True)
            elif exif[orientation] == 6:
                load=load.rotate(270, expand=True)
            elif exif[orientation] == 8:
                load=load.rotate(90, expand=True)                
        except AttributeError:
            pass
        except KeyError:
            pass

        width, height = load.size
        evex1 = min(self.evex1,self.evex2)
        evex2 = max(self.evex1,self.evex2)
        evey1 = min(self.evey1,self.evey2)
        evey2 = max(self.evey1,self.evey2)
        
        evex1 = width / (800 / evex1)
        evex2 = width / (800 / evex2)
        evey1 = height / (800 / evey1)
        evey2 = height / (800 / evey2)
        

        def feature_extraction2(neural_net, im_list,evex1,evex2,evey1,evey2):
            f = im_list
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if neural_net == 'inception_v3':
                f_size = 2048
                model = models.inception_v3(pretrained='imagenet')
                layer = model._modules.get('Mixed_7c')
            elif neural_net == 'resnet152': #2084
                f_size = 2048
                model = models.resnet152(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'resnet18': #512
                f_size =512
                model = models.resnet18(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'vgg16': #4096
                f_size = 4096
                model = models.vgg16(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer   = 'nothing'
            elif neural_net == 'vgg19': #4096
                f_size =    4096
                model = models.vgg19(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'densenet161': #2208
                f_size =2208
                model = models.densenet161(pretrained=True)	
                model = model.features
            elif neural_net == 'squeezenet1_0': #512
                f_size = 1000
                model = models.squeezenet1_0(pretrained=True)
                #model = model.features
                layer = 'nothing'
            elif neural_net == 'alexnet':    
                f_size = 4096
                model = models.alexnet(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            model.eval()
            model = model.to(device)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            if neural_net == 'inception_v3':
                transform = transforms.Compose([
                            transforms.Resize((299,299)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            else:
                transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            def get_vector2(image_name,f_size, layer, transform, model, neural_net,evex1,evex2,evey1,evey2):
                img = Image.open(image_name)
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(img._getexif().items())
                
                    if exif[orientation] == 3:
                        img=img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img=img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img=img.rotate(90, expand=True)                
                except AttributeError:
                    pass
                except KeyError:
                    pass
                img = img.crop((evex1,evey1,evex2,evey2))
                
                if img.mode == 'RGB':
                    try:
                        t_img = transform(img).unsqueeze(0)
                    except OSError:
                        t_img = transform(img).unsqueeze(0)
                    t_img = t_img.to(device)
                    if neural_net == 'alexnet' or neural_net =='vgg19' or neural_net =='vgg16' or neural_net =='alexnet' or neural_net =='squeezenet1_0':
                        torch.cuda.empty_cache()
                        my_embeddingz = model(t_img)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'densenet161':
                        featuresY = model(t_img)
                        my_embeddingz = F.relu(featuresY,inplace= True)
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'inception_v3':
                        my_embeddingz = torch.zeros((1,f_size,8,8))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                        my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    else:
                        my_embeddingz = torch.zeros((1,f_size,1,1))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                else:
                    my_embeddingz = np.zeros((f_size,))
                return my_embeddingz
            my_embed = []
            self.progress["value"] = 0
            self.progress["maximum"] = len(f)-1
            for i in range(0,len(f)):
                my_embed.append(get_vector2(f[i],f_size,layer,transform,model,neural_net,evex1,evex2,evey1,evey2))

            features = np.asarray(my_embed)
            return features


        self.focusfeatures = feature_extraction2(self.neuralnet,self.focused_image,evex1,evex2,evey1,evey2)

        def create_matrix2(focusfeatures,features,distance_metric):
            focusfeatures = np.squeeze(np.expand_dims(focusfeatures,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            focusfeatures_t = np.transpose(focusfeatures)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(focusfeatures_t)
            sumsquareX = sum(features_t**2)
            
#                for i in range(0,features.shape[0]):
            feat0 = focusfeatures
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asarray(cm)
            return cm
        self.focuscm = create_matrix2(self.focusfeatures,self.features,'correlation')
        temp_list = np.sort(self.focuscm.transpose(),0)[::-1]
        
        self.focus_list = np.argsort(self.focuscm.transpose(),0)[::-1]
        #self.rank_list = self.rank_list[0:1000]
        self.c.xview_moveto(self.origX)  #####
        self.c.yview_moveto(self.origY) ######
        self.focus_list = np.asarray(self.focus_list)
        temp_list = np.asarray(temp_list)
        temp_list[np.isnan(temp_list)] = -100
        self.focus_list = self.focus_list[temp_list>-50]
        self.bucketDisp = 0
        self.c.delete("all")
        x = []
        for i in range(0,len(self.focus_list)):
            d = self.im_list[self.focus_list[i]]
            x.append(d)
        self.display_images(self.focus_list)

    #Clusters the currently displayed images into subclusters per the selected images by the user
    def subcluster(self):
        m = len(self.selected_images) #number of images selected (number of subclusters)
        im_tags = [] 
        for ts in range(len(self.imagex)):
            im_tags.append(int(self.c.gettags(self.imagex[ts])[0])) #all images currently displayed
        im_tags = np.asarray(im_tags)
        im_tags = np.sort(im_tags)
        self.sel_im = im_tags[self.selected_images] #id of images currently selected
        #sel_im = sel_im.to_numpy()
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'subclustering')
        with h5py.File(self.hdf_path, 'r') as hdf:
                cmlen = hdf.get('cm').len()
                im_tags = im_tags[im_tags<cmlen]
                cur_cm = np.array(hdf.get('cm')[im_tags])
        
        #cur_cm = self.cm[im_tags] #correlations of images currently selected
        
        cur_cm = cur_cm[:,self.sel_im] #correlations of images currently selected
        cur_cm = np.nan_to_num(cur_cm) #removes nans
        cur_cmax = np.argmax(cur_cm,1) #determine which selected image correlates higher with each image
        nco = np.array([],dtype='int') #empty array for subclustered images
        for kk in range(m): #going through the selected images
            tb = im_tags[cur_cmax==kk]
            tc = cur_cm[cur_cmax==kk][:,kk]
            
            tb = tb[np.flip(np.argsort(tc))]
            tb = np.delete(tb,np.intersect1d(tb,self.sel_im,return_indices=True)[1],0)
            try:
                #nco= np.hstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.hstack([nco,self.sel_im[kk],tb])

            except ValueError:
                #nco= np.vstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.vstack([nco,self.sel_im[kk],tb])
        #for overview:
        try:
            self.xsorted
        except AttributeError:
            self.xsorted = np.argsort(im_tags)
            self.xsortedimtags = im_tags[self.xsorted]
                            
        ypos = np.searchsorted(self.xsortedimtags, nco)
        
        self.ind_for_overview = self.xsorted[ypos]
        ###
        self.subcyes = 1
        self.display_images(nco)
        
    def subclusterkmeans(self):
        m = len(self.selected_images)
        im_tags = [] 
        for ts in range(len(self.imagex)):
            im_tags.append(int(self.c.gettags(self.imagex[ts])[0]))
        im_tags = np.asarray(im_tags)
        self.sel_im = im_tags[self.selected_images]
        
        
        def color_features(an_image):
            an_image = an_image.convert('HSV')    
            imarray = np.asarray(an_image)
            imarray = imarray.reshape(imarray.shape[0]*imarray.shape[1],3)
            reimarray = np.vstack((imarray[:,0]/255*360,imarray[:,1]/255,imarray[:,2]/255)).transpose()
            reimarray[:,0][reimarray[:,0] < 20] = 0
            reimarray[:,0][np.where((reimarray[:,0] >= 20) & (reimarray[:,0] <= 40))] = 1
            reimarray[:,0][np.where((reimarray[:,0] >= 40) & (reimarray[:,0] <= 75))] = 2
            reimarray[:,0][np.where((reimarray[:,0] >= 75) & (reimarray[:,0] <= 155))] = 3
            reimarray[:,0][np.where((reimarray[:,0] >= 155) & (reimarray[:,0] <= 190))] = 4
            reimarray[:,0][np.where((reimarray[:,0] >= 190) & (reimarray[:,0] <= 270))] = 5
            reimarray[:,0][np.where((reimarray[:,0] >= 270) & (reimarray[:,0] <= 295))] = 6
            reimarray[:,0][reimarray[:,0] > 295] = 7
            
            reimarray[:,1][reimarray[:,1] < 0.2] = 0
            reimarray[:,1][np.where((reimarray[:,1] >= 0.2) & (reimarray[:,1] <= 0.7))] = 1
            reimarray[:,1][reimarray[:,1] > 0.7] = 2
            
            reimarray[:,2][reimarray[:,2] < 0.2] = 0
            reimarray[:,2][np.where((reimarray[:,2] >= 0.2) & (reimarray[:,2] <= 0.7))] = 1
            reimarray[:,2][reimarray[:,2] > 0.7] = 2
        
            colorvector = reimarray[:,0] * 9 + reimarray[:,1] * 3 +reimarray[:,2]
            colorvector = np.histogram(colorvector,76)[0]
            return colorvector
        def get_colors(image_name):
            img = Image.open(image_name)
            color_feature = color_features(img)
            return color_feature
#        my_colors = []
#        for i in range(len(im_tags)):
#            my_colors.append(get_colors(self.im_list[im_tags[i]]))
#        my_colors = np.asarray(my_colors)
        features = self.features[im_tags]
#        features = features/np.max(features)
#        features = np.hstack((features,my_colors))
#        features = my_colors
        centers = features[np.isin(im_tags,self.sel_im)]
        #sel_im = sel_im.to_numpy()
        ward = KMeans(n_clusters=len(self.sel_im), init=centers, n_init=1).fit(features)
        test2 = ward.labels_
#        cluster_ind = []
#        for i in range(0,len(sel_im)):
#            cluster_ind.append([])
#        for k in range(0,len(test2)):
#            cluster_ind[test2[k]].append(k)
#        xlength = []
#        for q in range(0,len(cluster_ind)):
#            xlength.append(len(cluster_ind[q]))
#        cluster_indX = np.zeros((max(xlength),len(cluster_ind)))-1        
#        for r in range(0,len(cluster_ind)):
#            for s in range(0,len(cluster_ind[r])):
#                cluster_indX[s,r] = int(cluster_ind[r][s])
#        cluster_indX = cluster_indX.astype(int)
        


#        cur_cm = self.cm[im_tags]
#        cur_cm = cur_cm[:,sel_im]
#        cur_cm = np.nan_to_num(cur_cm)
#        cur_cmax = np.argmax(cur_cm,1)
        nco = np.array([],dtype='int')
        for kk in range(m):
            tb = im_tags[test2==kk]#.to_numpy()
            try:
                #nco= np.hstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.hstack([nco,self.sel_im[kk],tb])

            except ValueError:
                #nco= np.vstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.vstack([nco,self.sel_im[kk],tb])
        self.subcyes = 1
        self.display_images(nco)


    def purtangles(self):
        self.purpletangles = []
        self.sel_im
        self.purpletangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='cyan2',width=self.rectanglewidth,tags = self.ccluster[j]))
        
    def query_external(self):
        self.external_image = str(filedialog.askopenfilename())
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'query external image ' + self.external_image)
        self.externalfeatures = self.feature_extraction3(self.neuralnet,self.external_image)
        
        self.focuscm = self.create_matrix3(self.externalfeatures,self.features,'correlation')
        temp_list = np.sort(self.focuscm.transpose(),0)[::-1]
        
        self.focus_list = np.argsort(self.focuscm.transpose(),0)[::-1]
        #self.rank_list = self.rank_list[0:1000]
        self.c.xview_moveto(self.origX)  #####
        self.c.yview_moveto(self.origY) ######
        self.focus_list = np.asarray(self.focus_list)
        temp_list = np.asarray(temp_list)
        temp_list[np.isnan(temp_list)] = -100
        self.focus_list = self.focus_list[temp_list>-50]
        self.bucketDisp = 0
        self.c.delete("all")
        x = []
        for i in range(0,len(self.focus_list)):
            d = self.im_list[self.focus_list[i]]
            x.append(d)
        self.display_images(self.focus_list)

    
    def query_clipboard_image(self):
        # try:
            # Get image from clipboard
            clipboard_image = ImageGrab.grabclipboard()
            
            if clipboard_image is None:
                self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'No image in clipboard')
                self.communication_label.configure(text='No image in clipboard.')
                return
            
            if clipboard_image.mode == 'RGBA':
                clipboard_image = clipboard_image.convert('RGB')
            # Save the clipboard image to a BytesIO object
            image_bytes = io.BytesIO()
            clipboard_image.save(image_bytes, format='jpeg')
            image_bytes.seek(0)  # Move to the beginning of the BytesIO object
            
            self.external_image = image_bytes
            
            self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'Query clipboard image')
            
            self.externalfeatures = self.feature_extraction3(self.neuralnet, self.external_image)
            
            self.focuscm = self.create_matrix3(self.externalfeatures, self.features, 'correlation')
            temp_list = np.sort(self.focuscm.transpose(), 0)[::-1]
            
            self.focus_list = np.argsort(self.focuscm.transpose(), 0)[::-1]
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.focus_list = np.asarray(self.focus_list)
            temp_list = np.asarray(temp_list)
            temp_list[np.isnan(temp_list)] = -100
            self.focus_list = self.focus_list[temp_list > -50]
            self.bucketDisp = 0
            self.c.delete("all")
            x = []
            for i in range(0, len(self.focus_list)):
                d = self.im_list[self.focus_list[i]]
                x.append(d)
            self.display_images(self.focus_list)
            
        # except Exception as e:
        #     self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'Error querying clipboard image: ' + str(e))
    


    def expand_cluster(self):
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'expand cluster' + str(self.num_clus))
        cluster_e = self.df[:,self.num_clus]
        cluster_e = cluster_e[cluster_e > -1]
        current_features = np.mean(self.features[cluster_e],0)
        try:
            self.new_threshold = self.new_threshold
        except AttributeError:
                self.new_threshold = self.threshold
                
        def create_matrix3(current_features,features,distance_metric):
            current_features = np.squeeze(np.expand_dims(current_features,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            current_features_t = np.transpose(current_features)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(current_features_t)
            sumsquareX = sum(features_t**2)
            
#                for i in range(0,features.shape[0]):
            feat0 = current_features
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asarray(cm)
            return cm
        currentcm = np.nan_to_num(create_matrix3(current_features,self.features,'correlation').transpose())
        self.new_threshold = self.new_threshold - 0.1
        expanded_cluster = np.expand_dims(np.arange(0,len(self.im_list)),1)
        expanded_cluster = expanded_cluster[currentcm > self.new_threshold]
        expanded_cluster = np.setdiff1d(expanded_cluster,cluster_e)
        expanded_cluster = np.concatenate((expanded_cluster,cluster_e))
        self.display_images(expanded_cluster)

    # function to display the contents of the selected bucket 
    def showEdge(self):
#        with open('D:\\PhD\\Visual Analytics\\MockCase\\zForBucket.pickle', 'wb') as handle:
#            pickle.dump(self.theBuckets,handle,protocol=pickle.HIGHEST_PROTOCOL)        
        self.c.delete("all")
        self.selected_edge = self.categories.get(self.categories.curselection()) #acquire the selected category
        if len(self.selected_edge) == 1:
            # self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'show bucket ' + self.categories.get(selected))
            # self.BucketSel = selected
            # self.cluster = self.theBuckets[self.categories.get(selected)]
            # self.cluster = np.asarray(self.cluster)
            # self.cluster = self.cluster[np.nonzero(self.cluster)]
            # self.current_bucket = self.cluster
            
            self.communication_label.configure(text='The hyperedge '+ self.categories.get(self.selected_edge) +' is shown.')


        # if len(self.selected_edge) > 1:
        #     self.cluster = []
        #     bucket_com = ''
        #     for i in range(0,len(selected)):
        #         self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'show the following buckets ' + self.categories.get(selected[i]))
        #         self.cluster.append(self.theBuckets[self.categories.get(selected[i])])
        #         bucket_com = bucket_com + self.categories.get(selected[i]) + '|' 

        #     self.cluster = list(chain.from_iterable(self.cluster)) 
        #     self.cluster = list(set(self.cluster))
        #     #self.cluster = self.cluster[np.nonzero(self.cluster)]
        #     self.current_bucket = self.cluster
        #     num_im = len(self.cluster)
        #     self.communication_label.configure(text='The following buckets are shown: '+ bucket_com +'. These buckets contain a total of ' + str(num_im) + ' images.')
            
        # self.imagex = []
        # self.cluster = np.asarray(self.cluster)
        self.display_hyperedges()
    
        
    


    # function to add the currently displayed cluster to the selected bucket(s)
    def addCluster2bucket(self):
        self.refresh = 0
        selected = self.categories.curselection()
        bucket_com = ''
        
        for p in range(0, len(selected)):
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'add cluster' + str(self.num_clus) + ' to bucket(s) ' + str(self.categories.get(selected[p])))
            temp_c = self.theBuckets[self.categories.get(selected[p])]
            temp_c = np.asarray(temp_c)
            temp_x = np.concatenate((self.currentcluster,temp_c),axis = 0)
            temp_y = numpy_unique_ordered(temp_x)
            temp_y = np.asarray(temp_y).astype(int)
            self.theBuckets[self.categories.get(selected[p])] = temp_y
            bucket_com = bucket_com + str(self.categories.get(selected[p])) + ' '
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(temp_y) + ' added to ' + self.categories.get(selected[p]))
        self.communication_label.configure(text='cluster ' + str(self.num_clus) + ' has been added to ' + bucket_com)
        #self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+self.theBuckets)
        
    # function to add the currently selected image(s) to the selected bucket(s)
    def addSelection2bucket(self):
        self.refresh = 0
        selected = self.categories.curselection()
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'add selection to bucket ' + str(selected))
        bucket_com = ''
        for p in range(0, len(selected)):
            bucket_com = bucket_com + str(self.categories.get(selected[p])) + ' '
            temp_c = np.asarray(self.theBuckets[self.categories.get(selected[p])])
#            if self.bucketDisp==0:
            temp_z = self.ccluster[self.selected_images]
            temp_x = np.concatenate((temp_z,temp_c),axis = 0)
#            elif self.bucketDisp==1:
#                temp_z = self.ccluster[self.selected_images]
#                temp_x = np.concatenate((temp_z,temp_c),axis = 0)
#            elif self.bucketDisp==2:
#                temp_z = self.ccluster[self.selected_images]
#                temp_x = np.concatenate((temp_z,temp_c),axis = 0)
            temp_y = numpy_unique_ordered(temp_x)
            temp_y = np.asarray(temp_y).astype(int)            
            self.theBuckets[self.categories.get(selected[p])] = temp_y
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(temp_y) + ' added to ' + self.categories.get(selected[p]))
        #self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+self.theBuckets)
        if len(temp_z) == 1:
            self.communication_label.configure(text=str(len(temp_z)) + ' image has been added to ' + bucket_com)
        else:
            self.communication_label.configure(text=str(len(temp_z)) + ' images have been added to ' + bucket_com)
            
    # function to add a new user created bucket. Buckets are sorted alphabeteically.
    def addCategory(self):
        if self.e2.get() not in self.catList:
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'added new cat to list: ' + self.e2.get())
            self.categories.insert(END, self.e2.get())
            self.theBuckets[self.e2.get()] = []
            self.catList = self.categories.get(0,END)
            self.catList = sorted(self.catList, key=str.lower)
            boxscrollbar = Scrollbar(width = 10)
            #self.categories = Listbox(width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar.set,exportselection=0)
            #self.categories['selectmode'] = 'extended'               
            #self.categories.place(x=585,y=50)
            self.categories.delete(0,END)
            for k in range(0,len(self.catList)):
                self.categories.insert(END,self.catList[k])
            boxscrollbar.config(command=self.categories.yview)
            boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
            self.categories.select_set(self.catList.index(self.e2.get()))
            self.categories.see(self.catList.index(self.e2.get()))
            self.categories.bind('<Button-1>', self.deselect_list )
    #function to display the next bucket
    def nextCluster(self):
        if self.num_clus < self.df.shape[1]-1:
            self.num_clus += 1
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'clicked next cluster: '+ str(self.num_clus))
            #self.b1['text'] ="next cluster" + ' (' + str(self.num_clus) + ')'
            self.c.xview_moveto(self.origX)
            self.c.yview_moveto(self.origY)
            self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
            self.new_threshold = self.threshold
            
            
            
    
    #function to display the previous bucket
    def prevCluster(self):
        if self.num_clus > 0:
            self.num_clus -= 1
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'clicked previous cluster: '+ str(self.num_clus))
            self.c.xview_moveto(self.origX)
            self.c.yview_moveto(self.origY)
            self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
            self.new_threshold = self.threshold

    def showCluster(self):
        self.num_clus = int(self.ecluster.get())-1
        self.c.xview_moveto(self.origX)
        self.c.yview_moveto(self.origY)
        self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
        self.new_threshold = self.threshold
        

    def change_name(self):
        new_bucket = self.e2.get()
        old_bucket = self.categories.curselection()
        if len(old_bucket) == 1:
            old_bucket = self.catList[old_bucket[0]]
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'changed name of bucket'+ old_bucket + ' into '+ new_bucket)
            self.catList = list(self.catList)
            self.catList[self.catList.index(old_bucket)] = new_bucket
            self.theBuckets[new_bucket] = self.theBuckets.pop(old_bucket)
            self.catList = sorted(self.catList, key=str.lower)
            boxscrollbar = Scrollbar(width = 10)
            self.categories.delete(0,END)
            for k in range(0,len(self.catList)):
                self.categories.insert(END,self.catList[k])
            boxscrollbar.config(command=self.categories.yview)
            boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
            self.categories.select_set(self.catList.index(self.e2.get()))
            self.categories.see(self.catList.index(self.e2.get()))
            self.categories.bind('<Button-1>', self.deselect_list )
            
    #function to display the images of the current cluster, e.g. to switch from the bucket view back to the current cluster.
    def showImg(self):
        self.bucketDisp = 0
        self.imagex = []
        self.c.delete("all")
        # self.num_clus
        # self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'showing cluster ' + str(self.num_clus))        
        
        try:
            cluster = np.asarray(self.df[:,self.num_clus])
            cluster = cluster[cluster > -1]
            self.currentcluster = cluster
            self.greentangles = []
            self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters. This cluster contains ' + str(len(cluster)) + ' images.')
        except KeyError:
            cluster = []
        self.display_images(cluster)
    #function to save the current session as a pickle file. The buckets, the list of images, the clusters, the current cluster number, the array with correlations, and the array with features are saved.
    

    def initialize_hyperedges(self):
        n_images, n_hyperedges = self.df.shape
        self.hyperedges = {}
        self.image_mapping = {}
        for h_idx in range(n_hyperedges):
            hyperedge_name = f'edge_{h_idx}'
            # Get the column corresponding to the hyperedge
            column = self.df[:, h_idx]
            # Find image indices where the entry is 1
            image_indices = np.nonzero(column)[0]
            # Update hyperedges dictionary
            self.hyperedges[hyperedge_name] = set(image_indices)
            
            # Update images dictionary
            for i_idx in image_indices:
                if i_idx not in self.image_mapping:
                    self.image_mapping[i_idx] = set()
                self.image_mapping[i_idx].add(hyperedge_name)



    def save_as(self):
        self.catList = self.categories.get(0, END)
        self.answer = filedialog.asksaveasfilename(defaultextension=".h5")
        
        with h5py.File(self.answer, 'a') as hdf:
            def create_or_replace_dataset(name, savedata):
                if name in hdf:
                    del hdf[name]
                hdf.create_dataset(name, data=savedata)
            
            h5_im_list = [n.encode("utf-8", "ignore") for n in self.im_list]
            create_or_replace_dataset('im_list', h5_im_list)
    
            # h5_log = [n.encode("utf-8", "ignore") for n in self.log]
            # create_or_replace_dataset('log', h5_log)
    
            create_or_replace_dataset('df', self.df)
    
            h5_catlist = [n.encode("utf-8", "ignore") for n in list(self.catList)]
            create_or_replace_dataset('catList', h5_catlist)

            # Save hyperedges dictionary           
            hyperedge_names = list(self.hyperedges.keys())
            hyperedge_image_ids = [np.array(list(image_ids), dtype='int64') for image_ids in self.hyperedges.values()]
            dt = h5py.vlen_dtype(np.dtype('int64'))
            create_or_replace_dataset('hyperedge_names', np.array(hyperedge_names, dtype=h5py.string_dtype()))
            create_or_replace_dataset('hyperedge_image_ids', np.array(hyperedge_image_ids, dtype=dt))
            # Save image_mapping dictionary
            image_ids = list(self.image_mapping.keys())
            image_hyperedge_names = [np.array(list(hyperedge_names), dtype=h5py.string_dtype()) for hyperedge_names in self.image_mapping.values()]
            dt_str = h5py.vlen_dtype(h5py.string_dtype())
            create_or_replace_dataset('image_ids', np.array(image_ids, dtype='int64'))
            create_or_replace_dataset('image_hyperedge_names', np.array(image_hyperedge_names, dtype=dt_str))

            create_or_replace_dataset('num_clus', self.num_clus)
    
            path_list = [self.hdf_path]
            p_list = [n.encode("utf-8", "ignore") for n in path_list]
            create_or_replace_dataset('cm_path', p_list)
    
            create_or_replace_dataset('features', self.features)
            # create_or_replace_dataset('X_embed', self.x_test)
    
            self.communication_label.configure(text='Saved!')

    
    #function to load a previous session. 
    def load_as(self):
        self.answer2 = filedialog.askopenfilename(defaultextension=".h5")  # This will make the file path a string
        with h5py.File(self.answer2, 'r') as hdf:
            # Load image list
            self.im_list = hdf['file_list'][()]
            # If the elements are bytes, decode them to strings
            if isinstance(self.im_list[0], bytes):
                self.im_list = [n.decode("utf-8", "ignore") for n in self.im_list]

            # Load clustering results
            # Assuming 'clustering_results' is a dataset containing numerical data
            self.df = hdf['clustering_results'][()]

            # Check if 'catList' exists in the HDF5 file
            if 'catList' in hdf:
                self.catList = list(hdf['catList'][()])
                # Decode bytes to strings if necessary
                if isinstance(self.catList[0], bytes):
                    self.catList = [n.decode("utf-8", "ignore") for n in self.catList]
            else:
                # Create catList based on the number of columns in self.df
                num_columns = self.df.shape[1]
                self.catList = tuple(f'edge_{i}' for i in range(num_columns))

            if 'hyperedges' in hdf:
                hyperedge_names = hdf['hyperedge_names'][()]
                hyperedge_image_ids = hdf['hyperedge_image_ids'][()]
                self.hyperedges = {
                    name: set(ids) for name, ids in zip(hyperedge_names, hyperedge_image_ids)
                }

                image_ids = hdf['image_ids'][()]
                image_hyperedge_names = hdf['image_hyperedge_names'][()]
                self.image_mapping = {
                    id: set(names) for id, names in zip(image_ids, image_hyperedge_names)
                }
            else:
                self.initialize_hyperedges()

            # Load 'num_clus' if it exists
            if 'num_clus' in hdf:
                self.num_clus = hdf['num_clus'][()]
            else:
                self.num_clus = 0  # Or set a default value

            self.hdf_path = self.answer2
            self.features = hdf['features'][()]

        # Update categories Listbox
        self.categories.delete(0, END)
        for category in self.catList:
            self.categories.insert(END, category)

        # Configure scrollbar for the Listbox
        boxscrollbar = Scrollbar(width=10)
        boxscrollbar.config(command=self.categories.yview)
        boxscrollbar.place(in_=self.categories, relx=1.0, relheight=1)
        self.categories.bind('<Button-1>', self.deselect_list)


    #function to export (make a copy of) all the images in all the buckets to folders with the buckets' names, in a location specified by the user.        
    def export_buckets(self):
        self.catList = self.categories.get(0,END)        
        self.answer = filedialog.askdirectory()   #this will make the file directory a string
        for p in range(0,len(self.catList)):
            bucket = self.theBuckets[self.catList[p]]
            if len(bucket) > 0:
                bucket_dir = self.answer + '/' + self.catList[p]
                if not os.path.exists(bucket_dir):
                        os.makedirs(bucket_dir)
                for q in range(0,len(bucket)):
                    source_ = self.im_list[int(bucket[q])]
                    destination_ = bucket_dir + "/" + os.path.basename(source_)
                    if os.path.isfile(destination_):
                        shutil.copy2(source_, destination_ + id_generator())
                    else:
                        shutil.copy2(source_, destination_)
            
    #function to select the image folder that the user wants to analyze
    def select_foldere(self):
        self.selected_folder = filedialog.askdirectory()   #this will make the file directory a string
        if len(self.selected_folder) > 0:
            self.im_list = glob.glob(self.selected_folder + '/**/*.jpg', recursive=True)
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.JPG', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.gif', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.GIF', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.png', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.PNG', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.tif', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.TIF', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.bmp', recursive=True))
            self.im_list.extend(glob.glob(self.selected_folder + '/**/*.BMP', recursive=True))
            self.im_list = list(set(self.im_list))
            self.communication_label.configure(text='Continue by pressing the Calculate image features button')
            self.sel_folder.set('found ' + str(len(self.im_list)) + ' images in ' + self.selected_folder)

    


    #function to extract features from a neural network as defined below. New networks can be added if needed.
    def feature_extraction(self, neural_net, im_list,new_imgs=False):
            f = im_list
            with h5py.File(self.hdf_path, 'a') as hdf:
                try:
                    hdf.create_dataset('thumbnail_images',((len(im_list),100,100,3)),chunks=(1,100,100,3),maxshape=(None,100,100,3))
                except RuntimeError:
                    pass

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if neural_net == 'inception_v3':
                f_size = 2048
                model = models.inception_v3(pretrained='imagenet')
                layer = model._modules.get('Mixed_7c')
            elif neural_net == 'resnet152': #2084
                f_size = 2048
                model = models.resnet152(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'resnet18': #512
                f_size =512
                model = models.resnet18(pretrained=True)
                layer = model._modules.get('avgpool')            
            elif neural_net == 'swin2': #512
                f_size = 1000
                model = timm.create_model('swinv2_large_window12to16_192to256_22kft1k',pretrained=True)
                layer = 'nothing'
            elif neural_net == 'vgg16': #4096
                f_size = 4096
                model = models.vgg16(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer   = 'nothing'
            elif neural_net == 'vgg19': #4096
                f_size =    4096
                model = models.vgg19(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'densenet161': #2208
                f_size =2208
                model = models.densenet161(pretrained=True)	
                model = model.features
            elif neural_net == 'squeezenet1_0': #512
                f_size = 1000
                model = models.squeezenet1_0(pretrained=True)
                #model = model.features
                layer = 'nothing'
            elif neural_net == 'alexnet':    
                f_size = 4096
                model = models.alexnet(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'clip':
                f_size = 1024
                #model, preprocess = clip.load("ViT-B/32", device=device)
                model, preprocess = clip.load("RN50x64", device=device)
                layer = 'nothing'
                
            model.eval()
            model = model.to(device)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            if neural_net == 'inception_v3':
                transform = transforms.Compose([
                            transforms.Resize((299,299)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            elif neural_net == 'swin2':
                transform = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])
            else:
                transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            def get_vector(image_name,f_size, layer, transform, model, neural_net):
                try:
                    image_name = image_name.strip()
                    img = Image.open(image_name)
                        
                        
                except OSError:
                    #if an image fails to open, a black replacement image is created...
                    print('failopen',image_name)
                    img = Image.new('RGB',(100,100))
                
                if neural_net == 'clip':
                    pass
                else:
                    if img.mode == 'RGB':
                        try:
                            t_img = transform(img).unsqueeze(0)
                        except OSError:
                            t_img = transform(img).unsqueeze(0)
                        
                        
                        t_img = t_img.to(device)
                        if neural_net == 'alexnet' or neural_net =='vgg19' or neural_net =='vgg16' or neural_net=='swin2' or neural_net =='alexnet' or neural_net =='squeezenet1_0':
                            torch.cuda.empty_cache()
                            my_embeddingz = model(t_img)
                            my_embeddingz = my_embeddingz.cpu()
                            my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                        elif neural_net == 'densenet161':
                            featuresY = model(t_img)
                            my_embeddingz = F.relu(featuresY,inplace= True)
                            my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                            my_embeddingz = my_embeddingz.cpu()
                            my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                        elif neural_net == 'inception_v3':
                            my_embeddingz = torch.zeros((1,f_size,8,8))
                            my_embeddingz = my_embeddingz.to(device)
                                # 4. Define a function that will copy the output of a layer
                            def copy_data(m, i, o):
                                my_embeddingz.copy_(o.data)
                                # 5. Attach that function to our selected layer
                            h = layer.register_forward_hook(copy_data)
                                # 6. Run the model on our transformed image
                            model(t_img)
                            #    # 7. Detach our copy function from the layer
                            h.remove()
                            my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                            my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                            my_embeddingz = my_embeddingz.cpu()
                            my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                        elif neural_net == 'clip':
                            t_img = preprocess(img).unsqueeze(0).to(device)
                            with torch.no_grad():
                                my_embeddingz = model.encode_image(t_img)
                                my_embeddingz = my_embeddingz.cpu()
                                my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())                                                            
                        else:
                            my_embeddingz = torch.zeros((1,f_size,1,1))
                            my_embeddingz = my_embeddingz.to(device)
                                # 4. Define a function that will copy the output of a layer
                            def copy_data(m, i, o):
                                my_embeddingz.copy_(o.data)
                                # 5. Attach that function to our selected layer
                            h = layer.register_forward_hook(copy_data)
                                # 6. Run the model on our transformed image
                            model(t_img)
                            #    # 7. Detach our copy function from the layer
                            h.remove()
                            my_embeddingz = my_embeddingz.cpu()
                            my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    else:
                        my_embeddingz = np.zeros((f_size,))
                return my_embeddingz,img
            my_embed = []
#            self.progress["value"] = 0
#            self.progress["maximum"] = len(f)-1
            with h5py.File(self.hdf_path, 'a') as hdf:            
                for i in range(0,len(f)):
                    veccie,img_to_store = get_vector(f[i],f_size,layer,transform,model,neural_net)
                    my_embed.append(veccie)
                    img_to_store = np.array(img_to_store.resize((100,100)))

                    if len(img_to_store.shape) == 2:
                        img_to_store = np.stack((img_to_store,)*3, axis=-1)
                    elif img_to_store.shape[2] == 4:
                        img_to_store = img_to_store[:,:,0:3]
                    elif img_to_store.shape[2] == 2:
                        img_to_storex = np.zeros((100,100,3))+200
                        img_to_storex[:,:,0:2] = img_to_store
                        img_to_store = img_to_storex
                    if new_imgs == True:
                        hdf['thumbnail_images'][i+len(self.features)] = img_to_store
                    else:
                        hdf['thumbnail_images'][i] = img_to_store
                    print ("\r Extracting image {} out of {} images ".format(i+1,len(f)), end="")
    #                if i%10 == 0:
                    self.communication_label.configure(text='Processing image '+ str(i) + ' of ' + str(len(f)))
                    #                    self.communication_label['background'] = '#99CCFF'
                    self.communication_label.update()
    #                self.communication_label.update()
            print("Finished extracting")
            features = np.asarray(my_embed)
            
            return features

    def feature_extraction3(self,neural_net, image_source):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if neural_net == 'inception_v3':
            f_size = 2048
            model = models.inception_v3(pretrained='imagenet')
            layer = model._modules.get('Mixed_7c')
        elif neural_net == 'resnet152': #2084
            f_size = 2048
            model = models.resnet152(pretrained=True)
            layer = model._modules.get('avgpool')
        elif neural_net == 'resnet18': #512
            f_size =512
            model = models.resnet18(pretrained=True)
            layer = model._modules.get('avgpool')
        elif neural_net == 'vgg16': #4096
            f_size = 4096
            model = models.vgg16(pretrained=True)
            model.classifier = model.classifier[:-1]
            layer   = 'nothing'
        elif neural_net == 'vgg19': #4096
            f_size =    4096
            model = models.vgg19(pretrained=True)
            model.classifier = model.classifier[:-1]
            layer = 'nothing'
        elif neural_net == 'densenet161': #2208
            f_size =2208
            model = models.densenet161(pretrained=True)    
            model = model.features
        elif neural_net == 'squeezenet1_0': #512
            f_size = 1000
            model = models.squeezenet1_0(pretrained=True)
            #model = model.features
            layer = 'nothing'
        elif neural_net == 'alexnet':    
            f_size = 4096
            model = models.alexnet(pretrained=True)
            model.classifier = model.classifier[:-1]
            layer = 'nothing'
        model.eval()
        model = model.to(device)
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if neural_net == 'inception_v3':
            transform = transforms.Compose([
                        transforms.Resize((299,299)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
        else:
            transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
        def get_vector3(image_source, f_size, layer, transform, model, neural_net):
            img = Image.open(image_source)
            
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(img._getexif().items())
            
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)
            except AttributeError:
                pass
            except KeyError:
                pass
    
            if img.mode == 'RGB':
                try:
                    t_img = transform(img).unsqueeze(0)
                except OSError:
                    t_img = transform(img).unsqueeze(0)
                t_img = t_img.to(device)
                if neural_net in ['alexnet', 'vgg19', 'vgg16', 'squeezenet1_0']:
                    torch.cuda.empty_cache()
                    my_embeddingz = model(t_img)
                    my_embeddingz = my_embeddingz.cpu()
                    my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                elif neural_net == 'densenet161':
                    featuresY = model(t_img)
                    my_embeddingz = F.relu(featuresY, inplace=True)
                    my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                    my_embeddingz = my_embeddingz.cpu()
                    my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                elif neural_net == 'inception_v3':
                    my_embeddingz = torch.zeros((1, f_size, 8, 8))
                    my_embeddingz = my_embeddingz.to(device)
                    def copy_data(m, i, o):
                        my_embeddingz.copy_(o.data)
                    h = layer.register_forward_hook(copy_data)
                    model(t_img)
                    h.remove()
                    my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                    my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                    my_embeddingz = my_embeddingz.cpu()
                    my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                else:
                    my_embeddingz = torch.zeros((1, f_size, 1, 1))
                    my_embeddingz = my_embeddingz.to(device)
                    def copy_data(m, i, o):
                        my_embeddingz.copy_(o.data)
                    h = layer.register_forward_hook(copy_data)
                    model(t_img)
                    h.remove()
                    my_embeddingz = my_embeddingz.cpu()
                    my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
            else:
                my_embeddingz = np.zeros((f_size,))
            return my_embeddingz
        my_embed = []
        self.progress["value"] = 0
        self.progress["maximum"] = 0  # Update this with the actual number of images if you have a progress bar.
        
        my_embed.append(get_vector3(image_source, f_size, layer, transform, model, neural_net))
    
        features = np.asarray(my_embed)
        return features

    
    def create_matrix(self,features,distance_metric):
            cm = []
            if distance_metric == 'correlation':
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                cm = []
                sumX = sum(features_t)
                sumsquareX = sum(features_t**2)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    sumXY = np.dot(feat0,features_t)
                    r = features.shape[1]*sumXY - sumX[i]*sumX 
                    s = ((features.shape[1] * sumsquareX) - sumX**2)
                    t = 1./((s[i]*s)**0.5)
                    u = r * t
                    cm.append(u)
                cm = np.asarray(cm)
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
            elif distance_metric == 'l2':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                #features = preprocessing.normalize(features, norm='l2')
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    sumIT = sumIT ** 0.5
                    cm.append(sumIT)
                cm = np.asarray(cm)
                cm = 1-cm
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            elif distance_metric == 'l1':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    cm.append(sumIT)
                cm = np.asarray(cm)
                cm = 1-cm            
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
            elif distance_metric == 'euclidean':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    sumIT = sumIT / np.max(sumIT)
                    cm.append(sumIT)
                   
                cm = np.asarray(cm)
                cm = 1-cm            
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            elif distance_metric == 'cosine':
                cm = []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                rootsumsquareX = sum(features_t**2)**0.5
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    sXY = np.dot(feat0,features_t).transpose()
                    t_fea = sXY/(rootsumsquareX*rootsumsquareX[i])
                    cm.append(t_fea)
                cm = np.asarray(cm).astype('float32')
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            return cm


    def create_matrix3(self,focusfeatures,features,distance_metric):
        focusfeatures = np.squeeze(np.expand_dims(focusfeatures,0))
        features = np.squeeze(np.expand_dims(features,0))
        features_t = np.transpose(features)
        focusfeatures_t = np.transpose(focusfeatures)
        cm = []                
        sumX = sum(features_t)
        focussumX = sum(focusfeatures_t)
        sumsquareX = sum(features_t**2)
        
#                for i in range(0,features.shape[0]):
        feat0 = focusfeatures
        sumXY = np.dot(feat0,features_t)
        r = features.shape[1]*sumXY - focussumX*sumX 
        s = ((features.shape[1] * sumsquareX) - sumX**2)
        t = 1./((s[0]*s)**0.5)
        u = r * t
        cm.append(u)
        cm = np.asarray(cm)
        return cm

    
    def calculate_features(self):
        self.hdf_path = filedialog.asksaveasfilename(defaultextension=".h5")

        self.features = self.feature_extraction(self.neuralnet,self.im_list)
        self.communication_label.configure(text = 'Calculated the features. You can now start clustering the images by pressing Cluster images. You may also want to save now that the features are calculated')
        self.communication_label['background'] = '#FFFFFF'

        def create_matrix_fast(features,hdf_path):            
            block_size = 1000
            len_f = len(features)
        
            with h5py.File(hdf_path, 'a') as hdf:
                try:
                    hdf.create_dataset('cm',((len_f,len_f)),chunks=(1,len_f),maxshape=(None,None))#(,len_f)
                except RuntimeError:
                    del hdf['cm']
                    hdf.create_dataset('cm',((len_f,len_f)),chunks=(1,len_f),maxshape=(None,None))#(len_f,len_f)
        
                loops = math.floor(len_f/block_size)
                start_loop1 = -1
                for loop2 in range(loops):        
                    features_p2 = features[loop2*block_size:loop2*block_size + block_size]
                    start_loop1 = start_loop1 + 1
                    for loop1 in range(start_loop1, loops):
                        features_p1 = features[loop1*block_size:loop1*block_size + block_size]
                        cm_p = np.corrcoef(features_p1,features_p2)[block_size::,0:block_size]
                        if loop1 == loop2:
                            for id_1 in range(len(features_p1)):
                                cm_p[id_1,id_1]=0
        
                        hdf['cm'][loop2*block_size:loop2*block_size + block_size, loop1*block_size:loop1*block_size + block_size] = cm_p
        
                        if loop1 != loop2:
                            hdf['cm'][loop1*block_size:loop1*block_size + block_size,loop2*block_size:loop2*block_size + block_size] = cm_p.transpose()
        
                remaining = len_f - block_size * loops
                if remaining > 0:
                    features_p1 = features[block_size*loops::]
                    for loop2 in range(loops):
                        features_p2 = features[loop2*block_size:loop2*block_size + block_size]
                        cm_p = np.corrcoef(features_p2,features_p1)[0:block_size,block_size::]
                        hdf['cm'][loop2*block_size:loop2*block_size + block_size, block_size*loops::] = cm_p
        
                    cm_p = np.corrcoef(features_p1)
                    for id_1 in range(len(cm_p)):
                        cm_p[id_1,id_1]=0
        
                    hdf['cm'][block_size*loops::,block_size*loops::] = cm_p
                    hdf['cm'][block_size*loops::,:] = np.array(hdf.get('cm')[:,block_size*loops::]).transpose()
        create_matrix_fast(self.features,self.hdf_path)
        
        
    def add_new_images(self):
        self.new_image_file = filedialog.askopenfilename(defaultextension=".txt")
        with open(self.new_image_file, 'r') as file:
            new_images = file.readlines()
            
        num_new_images = len(new_images)
        with h5py.File(self.hdf_path, 'a') as hdf:
            hdf['thumbnail_images'].resize((len(hdf['thumbnail_images']) + num_new_images, 100, 100, 3))

        new_features = self.feature_extraction(self.neuralnet,new_images,new_imgs=True)
        block_size = 1000
        len_f = len(self.features)
        len_nf = len(new_features)
    
        with h5py.File(self.hdf_path, 'a') as hdf:
            hdf['cm'].resize((len_f + len_nf, len_f + len_nf))

            loops_new = math.floor(len_nf/block_size)
        
            # calculate new features correlation
            for loop1 in range(loops_new):
                features_p1 = new_features[loop1*block_size:loop1*block_size + block_size]
                for loop2 in range(loop1, loops_new):
                    features_p2 = new_features[loop2*block_size:loop2*block_size + block_size]
                    cm_p = np.corrcoef(features_p1,features_p2)[block_size::,0:block_size]
                    if loop1 == loop2:
                        for id_1 in range(len(features_p1)):
                            cm_p[id_1,id_1]=0
        
                    hdf['cm'][len_f+loop2*block_size:len_f+loop2*block_size + block_size, len_f+loop1*block_size:len_f+loop1*block_size + block_size] = cm_p
        
                    if loop1 != loop2:
                        hdf['cm'][len_f+loop1*block_size:len_f+loop1*block_size + block_size,len_f+loop2*block_size:len_f+loop2*block_size + block_size] = cm_p.transpose()
        
            remaining_new = len_nf - block_size * loops_new
            if remaining_new > 0:
                features_p1 = new_features[block_size*loops_new::]
                for loop2 in range(loops_new):
                    features_p2 = new_features[loop2*block_size:loop2*block_size + block_size]
                    cm_p = np.corrcoef(features_p2,features_p1)[0:block_size,block_size::]
                    hdf['cm'][len_f+loop2*block_size:len_f+loop2*block_size + block_size, len_f+block_size*loops_new::] = cm_p
        
                cm_p = np.corrcoef(features_p1)
                for id_1 in range(len(cm_p)):
                    cm_p[id_1,id_1]=0
        
                hdf['cm'][len_f+block_size*loops_new::,len_f+block_size*loops_new::] = cm_p
                hdf['cm'][len_f+block_size*loops_new::,:] = np.array(hdf.get('cm')[:,len_f+block_size*loops_new::]).transpose()

            loops = math.floor(len_f/block_size)
            # calculate correlation between old and new features
            for loop1 in range(loops):
                features_p1 = self.features[loop1*block_size:loop1*block_size + block_size]
                for loop2 in range(loops_new):
                    features_p2 = new_features[loop2*block_size:loop2*block_size + block_size]
                    cm_p = np.corrcoef(features_p1,features_p2)[block_size::,0:block_size]
            
                    hdf['cm'][loop1*block_size:loop1*block_size + block_size, len_f+loop2*block_size:len_f+loop2*block_size + block_size] = cm_p
                    hdf['cm'][len_f+loop2*block_size:len_f+loop2*block_size + block_size, loop1*block_size:loop1*block_size + block_size] = cm_p.transpose()
            
            remaining_old_new = len_f - block_size * loops
            if remaining_old_new > 0:
                features_p1 = self.features[block_size*loops::]
                for loop2 in range(loops_new):
                    features_p2 = new_features[loop2*block_size:loop2*block_size + block_size]
                    cm_p = np.corrcoef(features_p1,features_p2)[block_size::,0:block_size]
                    hdf['cm'][block_size*loops::,len_f+loop2*block_size:len_f+loop2*block_size + block_size] = cm_p
                    hdf['cm'][len_f+loop2*block_size:len_f+loop2*block_size + block_size,block_size*loops::] = cm_p.transpose()
            
            remaining_new_old = len_nf - block_size * loops_new
            if remaining_new_old > 0:
                features_p2 = new_features[block_size*loops_new::]
                for loop1 in range(loops):
                    features_p1 = self.features[loop1*block_size:loop1*block_size + block_size]
                    cm_p = np.corrcoef(features_p1,features_p2)[0:block_size,block_size::]
                    hdf['cm'][loop1*block_size:loop1*block_size + block_size,len_f+block_size*loops_new::] = cm_p
                    hdf['cm'][len_f+block_size*loops_new::,loop1*block_size:loop1*block_size + block_size] = cm_p.transpose()
        
        self.features = np.vstack((self.features, new_features))
        self.im_list = self.im_list + new_images

        self.communication_label.configure(text = 'Added new images. You can now start clustering the images by pressing Cluster images. You may also want to save now that the features are calculated')
        self.communication_label['background'] = '#FFFFFF'

            

    def remove_images(self):
        remove_image_file = filedialog.askopenfilename(defaultextension=".txt")
        with open(remove_image_file, 'r') as file:
            images_to_remove = file.readlines()

        images_to_remove_normalized = [os.path.normpath(path) for path in images_to_remove]
        im_list_normalized = [os.path.normpath(path) for path in self.im_list]
        
        # Find indices to remove
        indices_to_remove = [i for i, img in enumerate(im_list_normalized) if img in images_to_remove_normalized]
        
        # Remove from self.im_list
        self.im_list = [img for i, img in enumerate(self.im_list) if i not in indices_to_remove]
        self.features = [img for i, img in enumerate(self.features) if i not in indices_to_remove]
        for key in self.theBuckets.keys():
            self.theBuckets[key] = [index for index in self.theBuckets[key] if index not in indices_to_remove]
        with h5py.File(self.hdf_path, 'a') as hdf:
            old_ds = hdf['cm']
    
            # Calculate new shape after removing rows and columns
            new_shape = (old_ds.shape[0] - len(indices_to_remove), old_ds.shape[1] - len(indices_to_remove))
            
            new_ds = hdf.create_dataset('cm_new', new_shape, dtype=old_ds.dtype)
            
            # Indices to keep
            indices_to_keep = [i for i in range(old_ds.shape[0]) if i not in indices_to_remove]
            
            for i, old_i in enumerate(indices_to_keep):
                row_data = old_ds[old_i]
                row_data = np.delete(row_data, indices_to_remove)  # remove columns
                new_ds[i] = row_data
        
            del hdf['cm']  # delete the old dataset
        
            # Once the old dataset is deleted, create a new dataset with the desired name 'cm' and copy data from 'cm_new' to 'cm'
            hdf['cm'] = new_ds[...]
            del hdf['cm_new']  # Now you can delete the 'cm_new' dataset
        self.communication_label.configure(text = 'Images removed. Make sure to save, or the save file will be broken.')
        self.communication_label['background'] = '#FFFFFF'

        
        

    def create_matrix2(self,focusfeatures,features,distance_metric):
            focusfeatures = np.squeeze(np.expand_dims(focusfeatures,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            focusfeatures_t = np.transpose(focusfeatures)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(focusfeatures_t)
            sumsquareX = sum(features_t**2)            
            feat0 = focusfeatures
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asarray(cm)
            return cm
        
    #function to cluster all images based on the correlation score, as specified in []
    def cluster_images(self):
        self.communication_label.configure(text='Calculating the clusters. This may take several minutes, depending on the number of images.')
        self.communication_label['background'] = '#99CCFF'
        self.communication_label.update()
        try:
            self.threshold = float(self.threshold_entry.get())
        except ValueError:
            self.threshold = 0.5
        if self.threshold < 0:
            self.threshold = 0.5
        if self.threshold > 1:
            self.threshold = 0.5
        
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'clustering images with threshold '+ str(self.threshold))
        
        def clustering(cm_path,threshold):

            cluster_ind = []
            clustered_images = []
            with h5py.File(cm_path, 'r') as hdf:
                cm_len = hdf['cm'].shape[1]
                cm_sum = np.zeros((cm_len,1))
                for mm in range(cm_len):
                    #warnings.filterwarnings("ignore")
                    # cm_sum[mm,0] = len(np.where(np.array(hdf.get('cm')[mm,:])>threshold)[0])
                    cm_sum[mm,0] = np.count_nonzero(np.array(hdf.get('cm')[mm,:]) > threshold)
                size = len(clustered_images)
                weight2 = 1
                m = cm_len
                tt = 0
                while size < cm_len:
                    if tt > 0 and threshold > 0.1:
                        threshold = threshold-0.1
                        cm_len = hdf['cm'].shape[1]
                        cm_sum = np.zeros((cm_len,1))
                        for mm in range(cm_len):
                            # cm_sum[mm,0] = len(np.where(np.array(hdf.get('cm')[mm,:])>threshold)[0])
                            cm_sum[mm,0] = np.count_nonzero(np.array(hdf.get('cm')[mm,:]) > threshold)
                    elif tt > 0 and threshold <= 0.1:
                        remains = np.arange(0,m)
                        remains[np.asarray(clustered_images)] = -100            
                        final_column = remains[remains > 0]
                        cluster_ind.append(final_column)
                        # cm[final_column,:] = -100
                        # cm[:,final_column] = -100
                        break
                    
                    tt = 1
                    new_cluster = []
                    #while np.sum(cm_bool) > 0:
                    while np.sum(cm_sum) - len(new_cluster)  > 0:
                        tg = 0
                        for g in range(0,len(cluster_ind)):
                            tg = tg + len(cluster_ind[g])
                        #cm_sum = np.sum(cm_bool,1)
                        cm_most = np.argmax(cm_sum)
                        column1 = np.array(hdf.get('cm')[cm_most,:])
                        try:
                            column1[np.asarray(clustered_images)] = -100
                        except IndexError:
                            pass                                            
                        new_cluster = []
                        column_max = bn.nanargmax(column1)
                        if column1[column_max] > threshold:
                            column2 = np.array(hdf.get('cm')[column_max,:])
                            try:
                                column2[np.asarray(clustered_images)] = -100
                            except IndexError:
                                pass
                                
                            
                            weight1 = 1        
                            new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
                            ## cm_bool[:,cm_most] = 0
                            ## cm_bool[cm_most,:] = 0
                            ## cm_bool[column_max,:] = 0
                            ## cm_bool[:,column_max] = 0
                            new_cluster.append(cm_most)
                            new_cluster.append(column_max)
                            clustered_images.append(cm_most)
                            clustered_images.append(column_max)
                            new_column[new_cluster] = 0
                            while bn.nanmax(new_column) > threshold:
                                weight1 = weight1 + 1
                                column1 = new_column
                                column_max = bn.nanargmax(column1)
                                
                                column2 = np.array(hdf.get('cm')[column_max,:])
                                try:
                                    column2[np.asarray(clustered_images)] = -100
                                except IndexError:
                                    pass
                                new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
                                # cm_bool[column_max,:] = 0
                                # cm_bool[:,column_max] = 0
                                new_cluster.append(column_max)
                                clustered_images.append(column_max)
                                new_column[new_cluster] = 0
                        else:
                            break
                        cluster_ind.append(new_cluster)
                        cm_sum[new_cluster] = 0
                        ## cm[new_cluster,:] = -100
                        ## cm[:,new_cluster] = -100
                        size = len(clustered_images)
            xlength = []
            for q in range(0,len(cluster_ind)):
                xlength.append(len(cluster_ind[q]))
            cluster_indX = np.zeros((max(xlength),len(cluster_ind)))-1        
            for r in range(0,len(cluster_ind)):
                for s in range(0,len(cluster_ind[r])):
                    cluster_indX[s,r] = int(cluster_ind[r][s])
            cluster_indX = cluster_indX.astype(int)                        
                
            return cluster_indX


        self.df = clustering(self.hdf_path,self.threshold)
        #self.df = pd.DataFrame.from_records(self.df)
        self.communication_label.configure(text='Calculated the clusters. You can now start browsing the clusters by Next cluster. You could also preload the images for a smoother browsing experience. Preloading takes a moment initially and uses a bit more system memory.')
        self.communication_label['background'] = '#FFFFFF'
        num_c = 0
        self.num_c = 0
    

    def hyperedge_preparation(self):
        """Based on the currently selected hyperedge, determine which other hyperedges need to be shown."""
        if self.selected_edge is None:
            self.selected_edge = next(iter(self.hyperedges))
        
        # Get the images in the selected hyperedge
        self.edge_images = self.hyperedges[self.selected_edge]
        
        overlapping_hyperedges = set()
        for image_id in self.edge_images:
            hyperedges_of_image = self.image_mapping[image_id]
            overlapping_hyperedges.update(hyperedges_of_image)
        
        # Remove the selected hyperedge from the set of overlapping hyperedges
        overlapping_hyperedges.discard(self.selected_edge)
        
        # Prepare a list to store hyperedge data
        self.overlapping_hyperedges = []
        edge_ids = []
        for hyperedge in overlapping_hyperedges:
            edge_ids.append(hyperedge)
            images_in_hyperedge = self.hyperedges[hyperedge]
            overlapping_images = self.edge_images.intersection(images_in_hyperedge)
            non_overlapping_images = images_in_hyperedge - overlapping_images
            print(overlapping_images,non_overlapping_images)
            # Create an ordered list of images: shared images first
            ordered_images = list(overlapping_images) + list(non_overlapping_images)
            
            # Convert the ordered images list to a NumPy array
            hyperedge_array = np.array(ordered_images, dtype=int)
            
            # Append the NumPy array to the list
            self.overlapping_hyperedges.append(hyperedge_array)
        
        # Optionally, sort the arrays by the number of overlapping images (if desired)
        # self.overlapping_hyperedges = sorted(
        #     self.overlapping_hyperedges,
        #     key=lambda x: len(np.intersect1d(x, list(self.edge_images))),
        #     reverse=True
        # )
        sorted_indices = sorted(
            range(len(self.overlapping_hyperedges)),
            key=lambda i: len(np.intersect1d(self.overlapping_hyperedges[i], list(self.edge_images))),
            reverse=True
        )
    
        self.overlapping_hyperedges = [self.overlapping_hyperedges[i] for i in sorted_indices]
        self.edge_ids = [edge_ids[i] for i in sorted_indices]

        # Store the edge images as a NumPy array
        self.edge_images = np.array(list(self.edge_images), dtype=int)
        
        # Optionally, store the edge images as a NumPy array if needed




    def display_overlapping_hyperedges(self):
        # Destroy existing widgets in the new window
        for widget in self.newWindow.winfo_children():
            widget.destroy()
        
        # Clear the list of hyperedge_canvases
        self.hyperedge_canvases.clear()
        
        # Create a main canvas and scrollbar in the new window
        self.main_canvas = Canvas(self.newWindow, bg='#555555')
        self.main_canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        
        scrollbar = Scrollbar(self.newWindow, orient=VERTICAL, command=self.main_canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create a frame inside the main canvas
        frame = Frame(self.main_canvas, bg='#555555')
        self.main_canvas.create_window((0, 0), window=frame, anchor='nw')
        
        # Update scrollregion when the frame size changes
        frame.bind('<Configure>', lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all')))
        
        # For each overlapping hyperedge, create a canvas and display images
        for he_id, hyperedge in enumerate(self.overlapping_hyperedges):
            # Label for the hyperedge
            label = Label(frame, text=self.edge_ids[he_id], bg='#555555', fg='white', font=('Arial', 14))
            label.pack(fill=X, padx=10, pady=5)
        
            # Frame to hold the canvas and its scrollbar
            canvas_frame = Frame(frame)
            canvas_frame.pack(padx=10, pady=5, fill=BOTH, expand=TRUE)

            # Canvas for the hyperedge images
            hyperedge_canvas = Canvas(canvas_frame, bg='#555555', width=800, height=600)
            hyperedge_canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
            hyperedge_canvas.my_tag = he_id
            # Vertical scrollbar for the hyperedge canvas
            v_scrollbar = Scrollbar(canvas_frame, orient=VERTICAL, command=hyperedge_canvas.yview)
            v_scrollbar.pack(side=RIGHT, fill=Y)
        
            hyperedge_canvas.configure(yscrollcommand=v_scrollbar.set)
            self.list_of_colors
            
            # Bind mouse wheel events to the hyperedge canvas
            hyperedge_canvas.bind('<Enter>', lambda event: event.widget.focus_set())
            hyperedge_canvas.bind('<Enter>', self._bound_to_mousewheel_second_window)
            hyperedge_canvas.bind('<Leave>', self._unbound_to_mousewheel_second_window)
            hyperedge_canvas.bind('<Button-2>', self.open_image3)
            hyperedge_canvas.bind('<Button-1>', self.click_select)
            hyperedge_canvas.bind('<Shift-Button-1>', self.shift_click_select)
            hyperedge_canvas.bind('<Control-Button-1>', self.ctrl_click_select)
            hyperedge_canvas.bind('<Button-3>', self.rank_images)
            hyperedge_canvas.bind('<Double-Button-1>', self.double_click_overview)
        
            # Create an inner frame inside the hyperedge canvas
        
            # Update scrollregion when the inner frame size changes
            
            self.hyperedge_canvases.append(hyperedge_canvas)
            # Get image IDs and display them on the inner frame
            # image_ids = self.hyperedges[self.edge_ids[he_id]]
            image_ids = hyperedge
            print('imgids:',image_ids)
            image_indices = np.array(list(image_ids), dtype=int)
            print('imgindices:',image_indices)
            # inner_frame = Frame(hyperedge_canvas, bg='#555555')
            # inner_frame.bind('<Configure>', lambda e, canvas=hyperedge_canvas: canvas.configure(scrollregion=canvas.bbox('all')))
            # hyperedge_canvas.create_window((0, 0), window=inner_frame, anchor='nw')
            
            self.display_images_on_canvas(self.hyperedge_canvases[-1], image_indices)
            # self.display_images_on_frame(inner_frame, image_indices)




    def display_images_on_canvas(self, canvas, image_indices):
        # Similar setup as in display_images, but use the provided canvas
        self.sharetangles = []
        num_im_row = math.floor(800 / (self.imsize + self.image_distance))  # Assuming canvas width is 800

        # Prepare the canvas scroll region
        total_rows = math.ceil(len(image_indices) / num_im_row)
        canvas.config(scrollregion=(0, 0, 0, total_rows * (self.imsize + self.image_distance)))

        # Clear the canvas
        canvas.delete("all")

        # Initialize a list to store image references
        if not hasattr(canvas, 'image_refs'):
            canvas.image_refs = []
        else:
            # Clear previous references
            canvas.image_refs.clear()

        # Load and display images
        with h5py.File(self.hdf_path, 'r') as hdf:
            for idx, img_idx in enumerate(image_indices):
                load = Image.fromarray(np.array(hdf.get('thumbnail_images')[img_idx], dtype='uint8'))
                render = ImageTk.PhotoImage(load)
                row_ = idx // num_im_row
                column_ = idx % num_im_row
                x_pos = column_ * (self.imsize + self.image_distance) + (self.imsize / 2)
                y_pos = row_ * (self.imsize + self.image_distance) + (self.imsize / 2)
                for_bb = canvas.create_image(x_pos, y_pos, image=render,tags=(img_idx))
                canvas.image_refs.append(render)
                if img_idx in self.edge_images:
                    bbox = canvas.bbox(for_bb)
            
                    rect = canvas.create_rectangle(
                        bbox,
                        outline='yellow',
                        width=2,
                        tags='share_tag'
                    )
                    self.sharetangles.append(rect)




    def display_hyperedges(self):
        self.hyperedge_preparation()
        self.display_images(self.edge_images)
        self.display_overlapping_hyperedges()


    # function to sort clusters based on a bucket. Average feature vector for 
    # each cluster correlated with average feature vector of bucket. 
    def sorting_clusters(self):
        if self.bucketDisp == 1:
            self.num_clus_bu = copy.deepcopy(self.num_clus)
            self.df_bu = copy.deepcopy(self.df)
            
            self.df = np.asarray(self.df)
            catfeat = np.zeros((1,self.features.shape[1]))
            catfeat[0] = np.mean(self.features[np.asarray(self.current_bucket),:],0) #the average of all features in the current cat bucket
            avg_feat = []
            for i in range(0,self.df.shape[1]):
                cluster = self.df[:,i]
                cluster = cluster[np.where(cluster>-1)]
                avg_feat.append(np.mean(self.features[cluster],0))
            avg_feat = np.asarray(avg_feat)
            sortedcluster_ind = self.create_matrix2(catfeat,avg_feat,'')
            sortedcluster_ind = np.flip(np.argsort(sortedcluster_ind))
            sortedcluster_ind = pd.DataFrame(sortedcluster_ind.T)
            self.df  = self.df[:,sortedcluster_ind[0]]
            #self.df = pd.DataFrame.from_records(self.df)
            self.num_clus = 0
            self.showImg()
            self.communication_label.configure(text='Sorted the clusters, now showing the first cluster')
        else:
            self.communication_label.configure(text='Show a bucket first, before sorting.')
        
    def restore_cluster_order(self):
        self.num_clus = self.num_clus_bu
        self.df = self.df_bu
        self.showImg()
        
        
    #function to calculate the average features of a cluster. This way, the most representative image of a cluster can be found. Doing so, an overview of all clusters using the representative image can be generated.
    def calculate_avg_vector(self):
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'calculate overview')
        self.communication_label.configure(text='Calculating the vector. Please wait a moment.')
        self.communication_label['background'] = '#99CCFF'
        self.communication_label.update_idletasks()
        df = self.df + 1
        nonzeros = df.astype(bool).sum(axis=0)
        df = None
        avg_feat_vec = []
        self.cluster_to_vector = []
        self.represent=[]
        for i in range(0,self.df.shape[1]):
            avg_feat = np.zeros([nonzeros[i],np.size(self.features,1)])
            for j in range(0,nonzeros[i]):
                avg_feat[j,0:np.size(self.features,1)] = self.features[self.df[j][i]]
            avg_feat_vec.append(np.mean(avg_feat,axis=0))
            calc_corr = []
            for k in range(0,nonzeros[i]):
                cc = np.corrcoef(avg_feat_vec[i],avg_feat[k])
                calc_corr.append(cc[0][1])
            self.represent.append(self.df[np.argmax(calc_corr),i])
            self.cluster_to_vector.append(calc_corr)
        self.communication_label.configure(text='The vector has been calculated. You can now press Show overview to see a representative image of each cluster. You can then select a cluster and view it by pressing Show selected cluster.')
        self.communication_label['background'] = '#99CCFF'

    #function to show the overview of representative images for each cluster.    
    def show_overview(self):
        if len(self.represent) != self.df.shape[1]:
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'calculate overview')
            self.communication_label.configure(text='Calculating the vector. Please wait a moment.')
            self.communication_label['background'] = '#99CCFF'
            self.communication_label.update_idletasks()
            df = self.df + 1
            nonzeros = df.astype(bool).sum(axis=0)
            df = None
            avg_feat_vec = []
            self.cluster_to_vector = []
            self.represent=[]
            for i in range(0,self.df.shape[1]):
                avg_feat = np.zeros([nonzeros[i],np.size(self.features,1)])
                for j in range(0,nonzeros[i]):
                    avg_feat[j,0:np.size(self.features,1)] = self.features[self.df[j][i]]
                avg_feat_vec.append(np.mean(avg_feat,axis=0))
                calc_corr = []
                for k in range(0,nonzeros[i]):
                    cc = np.corrcoef(avg_feat_vec[i],avg_feat[k])
                    calc_corr.append(cc[0][1])
                self.represent.append(self.df[np.argmax(calc_corr),i])
                self.cluster_to_vector.append(calc_corr)
            
            
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'showing overview')
        try:
            del self.ind_for_overview
            del self.xsorted
            del self.xsortedimtags
        except AttributeError:
            pass
        self.bucketDisp = 0
        self.c.delete("all")
        self.im_numX = []
        self.imagex = []
        self.oview = 1
        self.display_images(self.represent,input_origin='overview')
        self.communication_label.configure(text='Now showing the overview.')
        self.communication_label['background'] = '#99CCFF'

    #Function that displays the cluster selected by the user from the overview of representative images
    def show_selected_cluster(self):
        im_num = self.selected_images
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'show selected clusters from overview: ' + str(im_num))
        if im_num:
            try:            
                im_num = self.ind_for_overview[im_num]
            except AttributeError:
                pass
            self.c.delete("all")
            #self.num_clus = im_num
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            cluster = []
            for tt in range(len(im_num)):
                
                cluster.append(self.df[:,im_num[tt]])
            cluster = np.asarray(cluster)
            cluster = cluster[cluster > -1]
            num_im =int(self.e1.get())
            self.imagex = []
        if num_im > len(cluster):
            num_im = len(cluster)
        self.display_images(cluster)
        self.communication_label.configure(text='Showing cluster ' + str([xa+1 for xa in im_num]) +'.') # because humans start count from 1, unlike python
        self.communication_label['background'] = '#99CCFF'
        self.communication_label.update_idletasks()
        
    
    def recreate_tsne(self):
        modelu = umap.UMAP(
        n_neighbors=200,
        min_dist=0.1,
        n_components=2,
        metric='euclidean'
        )

        self.x_test = modelu.fit_transform(self.features)
        
        self.X_embed = copy.deepcopy(self.x_test)
        self.X_embed[:,0] = self.X_embed[:,0] + abs(np.min(self.X_embed[:,0]))
        self.X_embed[:,1] = self.X_embed[:,1] + abs(np.min(self.X_embed[:,1]))
        self.X_embed[:,0] = self.X_embed[:,0]/np.max(self.X_embed[:,0])
        self.X_embed[:,1] = self.X_embed[:,1]/np.max(self.X_embed[:,1])
    

    def create_tsne(self):
        self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'Showing or creating umap')
        self.tsne_squares = None
    
        def tsne_click(event):
            self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'display image from umap selection')
            self.tsneclick += 1
            if self.tsneclick == 2:
                self.evex_tsne2 = self.canvas_tsne.canvasx(event.x)
                self.evey_tsne2 = self.canvas_tsne.canvasy(event.y)
                self.tsneclick = 0
                if self.tsne_squares is not None:
                    self.canvas_tsne.delete(self.tsne_squares)
                self.tsne_squares = self.canvas_tsne.create_rectangle(self.evex_tsne1, self.evey_tsne1, self.evex_tsne2, self.evey_tsne2)
    
                xmin = self.evex_tsne1 / self.canvas_tsne.winfo_width()
                xmax = self.evex_tsne2 / self.canvas_tsne.winfo_width()
                ymin = 1 - (self.evey_tsne1 / self.canvas_tsne.winfo_height())
                ymax = 1 - (self.evey_tsne2 / self.canvas_tsne.winfo_height())
    
                X_images = np.where((self.X_embed[:, 0] > np.min((xmin, xmax))) &
                                    (self.X_embed[:, 0] < np.max((xmin, xmax))) &
                                    (self.X_embed[:, 1] > np.min((ymin, ymax))) &
                                    (self.X_embed[:, 1] < np.max((ymin, ymax))))
    
                self.display_images(X_images[0])
            else:
                self.evex_tsne1 = self.canvas_tsne.canvasx(event.x)
                self.evey_tsne1 = self.canvas_tsne.canvasy(event.y)
    
        if len(self.x_test) < 1:
            modelu = umap.UMAP()
            self.x_test = modelu.fit_transform(self.features)
    
        self.X_embed = copy.deepcopy(self.x_test)
        self.X_embed[:, 0] = self.X_embed[:, 0] + abs(np.min(self.X_embed[:, 0]))
        self.X_embed[:, 1] = self.X_embed[:, 1] + abs(np.min(self.X_embed[:, 1]))
        self.X_embed[:, 0] = self.X_embed[:, 0] / np.max(self.X_embed[:, 0])
        self.X_embed[:, 1] = self.X_embed[:, 1] / np.max(self.X_embed[:, 1])
    
        # Prepare data for plotting
        data = {
            'x': self.X_embed[:, 0],
            'y': self.X_embed[:, 1],
            'category': ['Not in bucket'] * len(self.X_embed)
        }
    
        t = list(chain.from_iterable(self.theBuckets.values()))
        xx = list(np.arange(0, len(self.im_list)))
        not_in_bucket = list(sorted(set(np.arange(0, len(self.im_list))) - set(t), key=xx.index))
        in_bucket = t
    
        for idx in not_in_bucket:
            data['category'][idx] = 'Not in bucket'
        for idx in in_bucket:
            data['category'][idx] = 'In bucket'
        for key, data_list in self.theBuckets.items():
            for idx in data_list:
                data['category'][idx] = key
    
        df = pd.DataFrame(data)
    
        # Create a color map for buckets
        unique_buckets = list(set(data['category']) - {'Not in bucket'})
        color_map = {bucket: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for bucket in unique_buckets}
        color_map['Not in bucket'] = (0, 0, 0)  # Black for points not in any bucket
    
        # Create an image with PIL
        img_width = int(self.screen_width / 2)
        img_height = self.screen_height
        image = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
    
        # Draw the points with assigned colors
        for i, row in df.iterrows():
            x = int(row['x'] * img_width)
            y = int((1 - row['y']) * img_height)  # Invert the y-coordinate
            color = color_map[row['category']]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    
        # Convert PIL image to Tkinter format
        render_tsne = ImageTk.PhotoImage(image)
    
        # Reuse the canvas if it exists, otherwise create a new one
        if hasattr(self, 'canvas_tsne'):
            self.canvas_tsne.create_image(0, 0, anchor='nw', image=render_tsne)
            self.canvas_tsne.image = render_tsne  # Keep a reference to avoid garbage collection
        else:
            self.canvas_tsne = Canvas(self.newWindow, bg='#555544', bd=0, highlightthickness=0, width=img_width + 1, height=img_height + 1)
            self.canvas_tsne.place(x=self.screen_width / 2, y=100)
            self.canvas_tsne.create_image(0, 0, anchor='nw', image=render_tsne)
            self.canvas_tsne.image = render_tsne  # Keep a reference to avoid garbage collection
            self.canvas_tsne.bind("<Button-1>", tsne_click)

    # function to select video folder that you want to add to your clustering results.
    def select_video_foldere(self):
        self.selected_video_folder = filedialog.askdirectory()   #this will make the file directory a string
        if len(self.selected_video_folder) > 0:
            self.vid_list = glob.glob(self.selected_video_folder + '/**/*.mp4', recursive=True)
            self.vid_list.extend(glob.glob(self.selected_video_folder + '/**/*.MP4', recursive=True))
            self.vid_list.extend(glob.glob(self.selected_video_folder + '/**/*.webm', recursive=True))
            self.vid_list = list(set(self.vid_list))
            self.communication_label.configure(text='Continue by pressing the Add video to clusters. Found ' + str(len(self.vid_list)) + ' videos in ' + self.selected_video_folder)
            #self.sel_folder.set('found ' + str(len(self.vid_list)) + ' videos in ' + self.selected_video_folder)


    def video_extraction(self):
        def video_clustering(vidcm,threshold):
            cluster_ind = []
            clustered_images = []
            
            cm_len = vidcm.shape[1]
            cm_sum = np.zeros((cm_len,1))
            for mm in range(cm_len):
                #warnings.filterwarnings("ignore")
                cm_sum[mm,0] = len(np.where(vidcm[mm,:]>threshold)[0])


            size = len(clustered_images)
            weight2 = 1
            m = cm_len
            tt = 0
            while size < cm_len:
                if tt > 0 and threshold > 0.1:
                    threshold = threshold-0.1
                    cm_len = vidcm.shape[1]
                    cm_sum = np.zeros((cm_len,1))
                    for mm in range(cm_len):
                        cm_sum[mm,0] = len(np.where(vidcm[mm,:]>threshold)[0])
                
                elif tt > 0 and threshold <= 0.1:
                    remains = np.arange(0,m)
                    remains[np.asarray(clustered_images)] = -100            
                    final_column = remains[remains > 0]
                    cluster_ind.append(final_column)
                    # cm[final_column,:] = -100
                    # cm[:,final_column] = -100
                    break
                
                tt = 1
                new_cluster = []
                #while np.sum(cm_bool) > 0:
                while np.sum(cm_sum) - len(new_cluster)  > 0:
                    tg = 0
                    for g in range(0,len(cluster_ind)):
                        tg = tg + len(cluster_ind[g])
                    #cm_sum = np.sum(cm_bool,1)
                    cm_most = np.argmax(cm_sum)
                    column1 = vidcm[:,cm_most]
                    try:
                        column1[np.asarray(clustered_images)] = -100
                    except IndexError:
                        pass                                            
                    new_cluster = []
                    column_max = np.nanargmax(column1)
                    if column1[column_max] > threshold:
                        column2 = vidcm[:,column_max]
                        try:
                            column2[np.asarray(clustered_images)] = -100
                        except IndexError:
                            pass
                            
                        
                        weight1 = 1        
                        new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
                        ## cm_bool[:,cm_most] = 0
                        ## cm_bool[cm_most,:] = 0
                        ## cm_bool[column_max,:] = 0
                        ## cm_bool[:,column_max] = 0
                        new_cluster.append(cm_most)
                        new_cluster.append(column_max)
                        clustered_images.append(cm_most)
                        clustered_images.append(column_max)
                        new_column[new_cluster] = 0
                        while bn.nanmax(new_column) > threshold:
                            weight1 = weight1 + 1
                            column1 = new_column
                            column_max = np.nanargmax(column1)
                            
                            column2 = vidcm[:,column_max]
                            try:
                                column2[np.asarray(clustered_images)] = -100
                            except IndexError:
                                pass
                            new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
                            # cm_bool[column_max,:] = 0
                            # cm_bool[:,column_max] = 0
                            new_cluster.append(column_max)
                            clustered_images.append(column_max)
                            new_column[new_cluster] = 0
                    else:
                        break
                    cluster_ind.append(new_cluster)
                    cm_sum[new_cluster] = 0
                    ## cm[new_cluster,:] = -100
                    ## cm[:,new_cluster] = -100
                    size = len(clustered_images)
            xlength = []
            for q in range(0,len(cluster_ind)):
                xlength.append(len(cluster_ind[q]))
            cluster_indX = np.zeros((max(xlength),len(cluster_ind)))-1        
            for r in range(0,len(cluster_ind)):
                for s in range(0,len(cluster_ind[r])):
                    cluster_indX[s,r] = int(cluster_ind[r][s])
            cluster_indX = cluster_indX.astype(int)
            return cluster_indX
        
        def create_matrix2(focusfeatures,features,distance_metric):
            focusfeatures = np.squeeze(np.expand_dims(focusfeatures,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            focusfeatures_t = np.transpose(focusfeatures)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(focusfeatures_t)
            sumsquareX = sum(features_t**2)            
            feat0 = focusfeatures
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asarray(cm)
            return cm
        
        ### calculate avg or median of existing clustering. 
        if len(self.df)>0:
            npdf = np.asarray(self.df)
            avg_df = []
            median_df = []
            for clstr in range(npdf.shape[1]):
                # misschien overview hergebruiken...?
                clstr_feat = self.features[npdf[:,clstr][npdf[:,clstr]>-1]]
                avg_clstr_feat = np.mean(clstr_feat,0)
                median_clstr_feat = clstr_feat[np.argmax(create_matrix2(avg_clstr_feat,clstr_feat,'correlation'))]
                avg_df.append(avg_clstr_feat)
                median_df.append(median_clstr_feat)
            avg_df = np.asarray(median_df)
            median_df = np.asarray(median_df)
        
        ###create folder for extraction of frames, remove existing folder. 
        self.video_features_list = []
        self.video_features_original_video = [] #keeps list of which cluster belongs to what video.
        self.snapshot_list = []
        video_frame_path = os.getcwd()+'\\frames'
        snapshot_path = os.getcwd()+'\\snapshots'
        try:
                os.mkdir(snapshot_path)
        except FileExistsError:
            pass
        except PermissionError:
            print('not allowed to create folder (permission error)')
        snapshot_counter = 0
        for vidya in self.vid_list:           
            try:
                os.mkdir(video_frame_path)
            except FileExistsError:
                shutil.rmtree(video_frame_path)
                os.mkdir(video_frame_path)
            except PermissionError:
                print('not allowed to create folder (permission error)')
                        
            #### extract frames
            cap = cv2.VideoCapture(vidya)
            vidlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if vidlength > 20000:
                print(vidya + ' video too long and is skipped') ## misschien later video gewoon splitsen in batches van 10000
            else:
                success,image = cap.read()
                count = 0
                while success:
                    cv2.imwrite(video_frame_path + "\\frame%d.png" % count, image)     # save frame as JPEG file      
                    success,image = cap.read()
                    #print('Read a new frame: ', success)
                    count += 1            
                vid_frames = glob.glob(video_frame_path + '/**/*.png', recursive=True)
                vid_features = self.feature_extraction(self.neuralnet,vid_frames)
                cm_vid = self.create_matrix(vid_features,'correlation')
                vid_cluster = video_clustering(cm_vid,0.5)
                for clstr in range(vid_cluster.shape[1]):
                    clstr_feat = vid_features[vid_cluster[:,clstr][vid_cluster[:,clstr]>-1]]
                    avg_clstr_feat = np.mean(clstr_feat,0)
                    median_img = np.argmax(create_matrix2(avg_clstr_feat,clstr_feat,'correlation'))
                    median_clstr_feat = clstr_feat[median_img]
                    video_snapshot = vid_frames[median_img]
                    snpsht = snapshot_path + '\\' + str(snapshot_counter) + '.png'
                    shutil.copy(video_snapshot,snpsht)
                    ## misschien video maken van frames.
                    self.video_features_original_video.append(vidya)
                    self.video_features_list.append(median_clstr_feat)
                    self.snapshot_list.append(snpsht)
                    snapshot_counter += 1
        self.video_features_list = np.squeeze(np.asarray(self.video_features_list))
        #mogelijk moet het volgende nog efficienter gemaakt worden, maar ik denk dat dat wel meevalt. Afhankelijk van hoeveelheid clusters in df.
        combo_features = np.vstack((self.video_features_list,np.squeeze(median_df)))
        len_vidf = len(self.video_features_list)
        if len_vidf == 2048 and len(self.video_features_list.shape) == 1:
            len_vidf = 1
        
        vid_df_cm = np.corrcoef(combo_features)[0:len_vidf,len_vidf::]        
        self.c_id = np.nanargmax(vid_df_cm,1)
        self.c_val = np.argmax(vid_df_cm,1)
        placeholder_array = np.zeros((np.bincount(self.c_id).max(),self.df.shape[1]))-1
        for vc in range(len(self.c_id)):
            empty_spot = np.where(placeholder_array[:,self.c_id[vc]]==-1)[0][0]
            placeholder_array[empty_spot,self.c_id[vc]] = int(vc+len(self.features))
        placeholder_array = np.array(placeholder_array,dtype='int')
        self.features = np.vstack((self.features,self.video_features_list))
        self.im_list = self.im_list + self.snapshot_list
        #placeholder_array = pd.DataFrame.from_records(placeholder_array)
        self.df = np.asarray(self.df,dtype='int')
        self.df = np.vstack((self.df,placeholder_array))
        #self.df = pd.DataFrame.from_records(self.df)
        self.video_cm = np.zeros((len(self.video_features_list),len(self.features)))
        for vdo in range(len(self.video_features_list)):
            self.video_cm[vdo] = create_matrix2(self.video_features_list[vdo],self.features,'correlation')
        
        
                    
    ## SOMETHING IS STILL WRONG WITH WHERE EACH IMAGE IS PLACED.
    
    
    def determinator(self):
        if self.bucketDisp == 1:
            with h5py.File(self.hdf_path, 'r') as hdf:
                self.cur_buck = self.current_bucket
                self.cur_buck = np.sort(self.cur_buck)
                bucket_cm = np.array(hdf.get('cm')[self.cur_buck,:])
                bucket_within = bucket_cm[:,self.cur_buck]
                #np.savetxt("D:\\Stuff\\test\\bucket_within.csv", bucket_within, delimiter="@")
                self.inbetweeners = []
                self.inbetweeners_list = []
                for zz in range(len(self.cur_buck)):
                    #im1 = self.current_bucket[zz]
                    self.inbetweeners.append([])
                    for yy in range(len(self.cur_buck)):
                        if yy != zz:
                            try:
                                #im2 = self.current_bucket[yy]
                                curr_corr = bucket_within[zz,yy]
                                corr_all = np.vstack([bucket_cm[zz],bucket_cm[yy]])
                                corr_all2 = copy.deepcopy(corr_all)
                                corr_all2[corr_all2>curr_corr] = 1
                                corr_all2[corr_all2<=curr_corr] = 0
                                image_of_interest = np.sum(corr_all2,0)
                                image_of_interest[image_of_interest<2] = 0
                                image_of_interest[image_of_interest==2] = 1
                                ioi = np.sum(corr_all,0)
                                ioi[image_of_interest==0] = np.nan
                                self.inbetweeners[zz].append(np.nanargmin(ioi))
                                self.inbetweeners_list.append(np.nanargmin(ioi))
                                
                            except ValueError:
                                self.inbetweeners[zz].append(-1)
                #np.savetxt("D:\\Stuff\\test\\inbetweeners.csv", self.inbetweeners, delimiter="@")
                self.inbetweeners_list = np.unique(np.asarray(self.inbetweeners_list))
                self.c.delete("all")
                self.im_numX = []
                self.imagex = []
                self.display_images(self.inbetweeners_list)
        else:
            pass
    
    def use_determinator(self):
        positive_images = self.ccluster[self.selected_images]
        connected_images = []
        for zi in range(len(self.inbetweeners)):        
            connected_images.append([])
            for yi in range(len(positive_images)):            
                try:
                    connected_images[zi].append(int(np.where(positive_images[yi]==self.inbetweeners[zi])[0]))
                except TypeError:
                    pass
        
        
        the_groups = [[]]
        for uh in range(len(connected_images)):
            added = 0
            for ih in range(len(the_groups)):
                if set(the_groups[ih]).intersection(connected_images[uh]):
                    the_groups[ih] = the_groups[ih] + connected_images[uh]
                    added = 1
                else:
                    pass
            if added == 0:
                the_groups.append(connected_images[uh])
        added = 1
        while added == 1:
            added = 0
            for ih in range(len(the_groups)):
                for uh in range(len(the_groups)):
                    if ih != uh:
                        if set(the_groups[ih]).intersection(the_groups[uh]):
                            the_groups[ih] = the_groups[ih] + the_groups[uh]                            
                            the_groups[uh] = []
                            added = 1
        final_groups = []
        for grp in the_groups:
            if len(grp) > 0:
                grp = self.cur_buck[np.unique(np.asarray(grp))]
                final_groups.append(grp)
        self.final_groups_display = np.asarray(0)
        for gr in final_groups:
            self.final_groups_display = np.hstack((self.final_groups_display, gr, 0))
        self.display_images(self.final_groups_display)
        ## WHY DO I MISS CERTAIN IMAGES?? ALSO MAYBE 1 IMAGE IS ENOUGH TO MAKE IT A NEW CLUSTER.
    
    # redo and undo DISPLAY IMAGES ONLY, not actions such as adding to bucket.
    def ctrlzprevious(self):
        self.ctrlzused = 1
        if self.czcurrent > -10:
            self.czcurrent -= 1
        cluster = self.controlZ[self.czcurrent]
        self.display_images(cluster)
        # if self.czcurrent < -1 and len(self.controlZ) == 10:
        #     self.czcurrent += 1
        print(self.czcurrent)
        
    def ctrlznext(self):
        self.ctrlzused = 1
        if self.czcurrent < -1:
            self.czcurrent += 1
        cluster = self.controlZ[self.czcurrent]
        self.display_images(cluster)
        # if self.czcurrent < -10 and len(self.controlZ) == 10:
        #     self.czcurrent += - 1
        print(self.czcurrent)



#### meta data scripts ####

    def read_metadata(self):
        def get_exif_data(image_path):
            """
            Extracts the EXIF data from an image.
            """
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
                
            exif_data = {}
            for tag, value in tags.items():
                tag_name = tag.split(' ')[-1]
                exif_data[tag_name] = value

            return exif_data
        
        image_data = []
        all_columns = set()

        for imind, file in enumerate(self.im_list):
            
            exif_data = get_exif_data(file)
            
            # Ensure 'Image Path' column is included
            exif_data['Image Path'] = file
            exif_data['Index'] = imind  # Add index column
            
            image_data.append(exif_data)
            all_columns.update(exif_data.keys())
    
        # Convert the set of all columns to a list
        all_columns = sorted(list(all_columns))
        
        # Create DataFrame with all possible columns
        self.meta_data = pd.DataFrame(image_data, columns=all_columns)        
        self.metadata_listbox.delete(0, END)
        for col in all_columns:
            self.metadata_listbox.insert(END, col)
        self.metaList = self.metadata_listbox.get(0,END)

    def display_selected_metadata(self):
        selected_columns = list(self.metadata_selection_listbox.get(0, END))
        if selected_columns:
            self.display_metadata(selected_columns)
        else:
            self.communication_label.configure(text='No metadata items selected.')
            
            
    def display_metadata(self, columns):
        meta_window = Toplevel(root)
        meta_window.title("Metadata")
        
        if 'Index' not in columns:
            columns = ['Index'] + columns
        
        # Create a frame for the Treeview and scrollbar
        frame = ttk.Frame(meta_window)
        frame.pack(expand=True, fill='both')
    
        # Add a vertical scrollbar
        vsb = ttk.Scrollbar(frame, orient="vertical")
        vsb.pack(side='right', fill='y')
    
        # Add a horizontal scrollbar
        hsb = ttk.Scrollbar(frame, orient="horizontal")
        hsb.pack(side='bottom', fill='x')
    
        tree = ttk.Treeview(frame, columns=columns, show='headings', yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        for col in columns:
            tree.heading(col, text=col, command=lambda _col=col: self.sort_column(tree, _col, False))
            tree.column(col, width=100)
        
        for index, row in self.meta_data[columns].iterrows():
            tree.insert("", "end", values=row.tolist())
    
        tree.pack(expand=True, fill='both')
        
        # Configure the scrollbars
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        
        self.tree = tree


    def sort_column(self, tree, col, reverse):
        data = [(tree.set(child, col), child) for child in tree.get_children('')]
        data.sort(reverse=reverse)
    
        for index, (val, child) in enumerate(data):
            tree.move(child, '', index)
    
        tree.heading(col, command=lambda: self.sort_column(tree, col, not reverse))
        
        self.sorted_indices = [tree.item(child)['values'][0] for child in tree.get_children('')]
    
    def get_sorted_indices(self):
        try:
            return self.sorted_indices
        except AttributeError:
            return list(range(len(self.im_list)))
    
    
    def get_selected_and_subsequent_indices(self):
        selected_items = self.tree.selection()
        if not selected_items:
            return []
    
        # Get the index of the first selected item
        first_selected_item = selected_items[0]
        first_selected_index = self.tree.index(first_selected_item)
    
        # Get all items in the tree
        all_items = self.tree.get_children()
    
        # Retrieve indices of the selected item and all subsequent items
        subsequent_indices = [self.tree.item(item)['values'][0] for item in all_items[first_selected_index:]]
    
        return subsequent_indices
    
    
    def get_metadata(self):
        self.communication_label.configure(text='Collecting meta data. This may take a moment.')
        self.communication_label.update()
        self.read_metadata()
        # self.display_metadata()

    def get_selected_column(self):
        pass  

    def show_images_by_metadata(self):
        selected_and_subsequent_indices = np.array(self.get_selected_and_subsequent_indices())
        self.display_images(selected_and_subsequent_indices)
    
    def update_the_metalist(self,*args):
        search_term = self.metasearch_var.get()

        self.metadata_listbox.delete(0, END)
        
        for item in self.metaList:
            if search_term.lower() in item.lower():
                self.metadata_listbox.insert(END, item)
    
    def add_selected_metadata(self):
        selected_indices = self.metadata_listbox.curselection()
        for i in selected_indices:
            item = self.metadata_listbox.get(i)
            if item not in self.metadata_selection_listbox.get(0, END):
                self.metadata_selection_listbox.insert(END, item)
                
                

    def client_exit(self):
        exit()

    
root = Tk()
root.geometry("1900x700")
root.configure(background='#555555')
app = Application(root)

root.mainloop()  