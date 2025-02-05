import tkinter as tk
from tkinter import ttk, Canvas, Scrollbar
from PIL import Image, ImageTk
import numpy as np
import h5py
from tkintertable import TableCanvas, TableModel


class TableCanvasWithCustomSorting(TableCanvas):
    def sortTable(self, columnIndex=0, columnName=None, reverse=0):
        # call the original sort function
        super().sortTable(columnIndex, columnName, reverse)
        # Now call your custom callback to update the images
        #self.event_generate("<<TableSorted>>")
        print('custom worked')
        self.event_generate("<<TableSorted>>")

class LazyImagePanel(tk.Frame):
    def __init__(
        self, parent, images_in_selected, hdf_path, row_height=100, **kwargs
    ):
        super().__init__(parent, **kwargs)
        

        self.header_offset = 0
        self.images_in_selected = list(images_in_selected)
        self.hdf_path = hdf_path
        self.row_height = row_height
        self.row_count = len(self.images_in_selected)
        
        self.canvas = tk.Canvas(self, bg="white", width=120)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)


        self.active_rows = {}       
        self.active_images = {}     

        total_height = self.header_offset + self.row_count * self.row_height
        self.canvas.config(scrollregion=(0, 0, 120, total_height))

        self.canvas.bind("<Configure>", lambda e: self.render_visible_rows())

    def set_yview(self, *args):
        """For sync with your scrollbar code."""
        self.canvas.yview(*args)
        self.render_visible_rows()  

    def yview(self):
        """Return current yview fraction for synchronization."""
        return self.canvas.yview()

    def render_visible_rows(self):
        """
        Figure out which rows are in the visible vertical region, plus a buffer.
        Create or destroy canvas items so only those rows are drawn.
        """
        print('render row tirgered')
        frac_top, frac_bottom = self.canvas.yview()
        total_height = self.header_offset + self.row_count * self.row_height
        pix_top = int(frac_top * total_height)
        pix_bottom = int(frac_bottom * total_height)

        first_row = max(0, ((pix_top - self.header_offset) // self.row_height) - 10)
        last_row  = min(self.row_count, ((pix_bottom - self.header_offset) // self.row_height) + 10)

        rows_to_remove = [r for r in self.active_rows if r < first_row or r > last_row]
        for r in rows_to_remove:
            self.canvas.delete(self.active_rows[r])
            del self.active_rows[r]
            del self.active_images[r]

        for r in range(first_row, last_row+1):
            if r not in self.active_rows:
                self.render_single_row(r)

    def render_single_row(self, r):
        """
        Load the image for row r from the HDF, create a PhotoImage, and draw it.
        """
        y1 = self.header_offset + r * self.row_height
        y_center = y1 + self.row_height // 2

        img_id = self.images_in_selected[r]


        with h5py.File(self.hdf_path, "r") as hdf:
            raw_arr = np.array(hdf["thumbnail_images"][img_id], dtype="uint8")
        pil_img = Image.fromarray(raw_arr)
        pil_img = pil_img.resize((100, 100), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.active_images[r] = tk_img

        item_id = self.canvas.create_image(
            10, y_center, anchor="w", image=tk_img, tags=(img_id)
        )
        self.active_rows[r] = item_id





class SyncScrollExample(tk.Frame):
    def __init__(self, parent, images_in_selected, all_edges, hyperedges, hdf_path, data, model, **kwargs):
        super().__init__(parent, **kwargs)


        self.grid_columnconfigure(0, weight=0)  # Table body (expands)
        self.grid_columnconfigure(1, weight=0)  # Scrollbar (fixed width)
        self.grid_columnconfigure(2, weight=0)  # Image panel (expands)
        self.grid_rowconfigure(0, weight=0)     # Header (fixed height)
        self.grid_rowconfigure(1, weight=1)     # Body (expands)

      

        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.on_scrollbar)
        self.vscroll.grid(row=1, column=2, sticky="ns")  # Scrollbar in column 1

        
        self.tree = TableCanvasWithCustomSorting(
            parent       = self,
            model        = model,
            showkeynames = True,   
            rowheaderwidth=10,
            cellwidth    = 60,
            thefont      = ('Arial', 12),
            rowheight    = 100,
            bgcolor      = '#555555',
            editable     = False,
        )
        
        self.tree.bgcolor = '#555555'

        self.tree.grid(row=1, column=1, sticky="nsew")  
        self.tree.show()
        
        for image_key, row_dict in data.items():
            row_index = self.tree.model.getRecordIndex(image_key)

            for col_name, val in row_dict.items():
                if col_name == 'label':
                    col_index = self.tree.model.columnNames.index(col_name)
                    self.tree.model.setColorAt(row_index, col_index, color='#555555',key='bg')  
                if val == '1':
                    col_index = self.tree.model.columnNames.index(col_name)
                    self.tree.model.setColorAt(row_index, col_index, color='seagreen',key='bg')
                else:
                    col_index = self.tree.model.columnNames.index(col_name)
                    self.tree.model.setColorAt(row_index, col_index, color='#555555',key='bg')
        

        self.tree.redraw()
        print('one')
        self.tree.configure(yscrollcommand=self.on_treeview_scroll)
        
        
        print('two')
        self.image_panel = LazyImagePanel(
            self,
            images_in_selected=images_in_selected,            
            row_height=100,
            hdf_path=hdf_path
        )
        self.image_panel.grid(row=1, column=0, sticky="nsew")  
        self.image_panel.canvas.config(yscrollcommand=self.on_canvas_scroll)
        self.on_treeview_scroll(*self.tree.yview())
        self.tree.bind("<<TableSorted>>", self.on_table_sorted)


    def on_scrollbar(self, *args):
        """ Scroll both widgets. """
        self.image_panel.set_yview(*args)
        self.tree.yview(*args)  
        self.vscroll.set(*self.tree.yview())

    def on_treeview_scroll(self, first, last):
        """ Called by the Treeview's yscrollcommand """
        self.vscroll.set(first, last)
        self.image_panel.set_yview("moveto", first)

    def on_canvas_scroll(self, first, last):
        """ Called by the Canvas's yscrollcommand """
        self.vscroll.set(first, last)
        self.tree.yview("moveto", first)
    
    def on_table_sorted(self, event):
        print('on table triggered')
        new_order = self.tree.model.reclist
        self.image_panel.images_in_selected = new_order
        self.image_panel.row_count = len(new_order)
        total_height = self.image_panel.header_offset + self.image_panel.row_count * self.image_panel.row_height
        self.image_panel.canvas.config(scrollregion=(0, 0, 120, total_height))
        self.image_panel.active_rows.clear()
        self.image_panel.active_images.clear()
        self.image_panel.render_visible_rows()