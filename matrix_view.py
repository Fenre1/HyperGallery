import tkinter as tk
from tkinter import ttk, Canvas, Scrollbar
from PIL import Image, ImageTk
import numpy as np
import h5py
from tkintertable import TableCanvas, TableModel


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
        # self.canvas.pack(fill="both", expand=True)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)


        # Keep track of which rows are currently drawn
        self.active_rows = {}       # row_index -> canvas item ID
        self.active_images = {}     # row_index -> PhotoImage (to avoid garbage collection)

        # Set total scroll region
        total_height = self.header_offset + self.row_count * self.row_height
        self.canvas.config(scrollregion=(0, 0, 120, total_height))

        # Whenever the canvas is scrolled or resized, re-render visible rows
        self.canvas.bind("<Configure>", lambda e: self.render_visible_rows())
        # If you want to catch Canvas scrolling precisely, you could also bind
        # self.canvas.config(yscrollcommand=self.on_canvas_scroll)
        # then call self.render_visible_rows() in on_canvas_scroll().
        # But in your code, you're already calling on_canvas_scroll -> set_yview -> etc.

    def set_yview(self, *args):
        """For sync with your scrollbar code."""
        self.canvas.yview(*args)
        self.render_visible_rows()  # re-check which rows should be visible

    def yview(self):
        """Return current yview fraction for synchronization."""
        return self.canvas.yview()

    def render_visible_rows(self):
        """
        Figure out which rows are in the visible vertical region, plus a buffer.
        Create or destroy canvas items so only those rows are drawn.
        """
        # 1) Figure out which part of the canvas is visible (in pixels).
        frac_top, frac_bottom = self.canvas.yview()
        total_height = self.header_offset + self.row_count * self.row_height
        # Visible pixel range
        pix_top = int(frac_top * total_height)
        pix_bottom = int(frac_bottom * total_height)

        # 2) Convert pixel range to row indices, with a buffer of ~10 rows
        # first_row = max(0, (pix_top - self.header_offset // self.row_height) - 10)
        # last_row = min(self.row_count, (pix_bottom - self.header_offset// self.row_height) + 10)
        first_row = max(0, ((pix_top - self.header_offset) // self.row_height) - 10)
        last_row  = min(self.row_count, ((pix_bottom - self.header_offset) // self.row_height) + 10)



        # 3) Remove rows that are no longer in the range
        rows_to_remove = [r for r in self.active_rows if r < first_row or r > last_row]
        for r in rows_to_remove:
            # Remove the canvas item
            self.canvas.delete(self.active_rows[r])
            del self.active_rows[r]
            del self.active_images[r]

        # 4) Add rows that are in the range but not yet rendered
        for r in range(first_row, last_row+1):
            if r not in self.active_rows:
                self.render_single_row(r)

    def render_single_row(self, r):
        """
        Load the image for row r from the HDF, create a PhotoImage, and draw it.
        """
        # Y-coords of this row
        y1 = self.header_offset + r * self.row_height
        y_center = y1 + self.row_height // 2

        # The image ID
        img_id = self.images_in_selected[r]

        # 1) Load from HDF (pseudo-code; adapt to your actual data reading)
        #    This might be the slowest step, so you only do it for visible rows.

        with h5py.File(self.hdf_path, "r") as hdf:
            raw_arr = np.array(hdf["thumbnail_images"][img_id], dtype="uint8")
        pil_img = Image.fromarray(raw_arr)
        pil_img = pil_img.resize((100, 100), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        # 2) Keep a reference to avoid garbage collection
        self.active_images[r] = tk_img

        # 3) Draw it in the canvas
        item_id = self.canvas.create_image(
            10, y_center, anchor="w", image=tk_img, tags=(img_id)
        )
        self.active_rows[r] = item_id



# class MembershipTree(ttk.Treeview):
#     def __init__(self, parent, images_in_selected, all_edges, hyperedges, **kwargs):
#         """
#         images_in_selected: the same list of image IDs used by LazyImagePanel
#         all_edges: list/tuple of edge names
#         hyperedges: dict mapping edge -> set of image IDs
#         """
#         # The columns will be the edges, plus one for the 'Image' label:
#         columns = ["Image"] + list(all_edges)
        
#         super().__init__(parent, columns=columns, show="headings", **kwargs)

#         # Make headings
#         for col in columns:
#             self.heading(col, text=col)
#             self.column(col, width=80, stretch=True, anchor="center")

#         # Insert data
#         for i, img_id in enumerate(images_in_selected):
#             # For each edge, "1" if in hyperedges[edge], else "0"
#             row_values = [img_id] + [
#                 "1" if img_id in hyperedges[edge] else "0" 
#                 for edge in all_edges
#             ]
#             # Row striping
#             if i % 2 == 0:
#                 self.insert("", "end", values=row_values, tags=("oddrow",))
#             else:
#                 self.insert("", "end", values=row_values, tags=("evenrow",))

#         style = ttk.Style(self)
#         style.theme_use("clam")
#         style.configure(
#             "Dark.Treeview",
#             background="black",
#             fieldbackground="black",
#             foreground="white",
#             rowheight=100
#         )
#         style.configure(
#             "Dark.Treeview.Heading",
#             background="gray25",
#             foreground="white",
#         )
#         self.configure(style="Dark.Treeview")

#         # -- Configure tag colors
#         self.tag_configure("oddrow", background="#101010")
#         self.tag_configure("evenrow", background="#202020")

#         self.configure(style="Dark.Treeview")
#         # Tag config, etc.

#     def set_yview(self, *args):
#         """Matches your sync approach in SyncScrollExample."""
#         self.yview(*args)

#     def yview(self, *args):
#         return super().yview(*args)





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

        # Left lazy image panel
        
        # Right membership tree 
        self.tree = TableCanvas(
            parent       = self,
            model        = model,
            showkeynames = True,   # Show row label from 'label' field
            rowheaderwidth=10,
            cellwidth    = 60,
            thefont      = ('Arial', 12),
            rowheight    = 100,
            bgcolor      = '#555555',
            editable     = False,
        )
        
        self.tree.bgcolor = '#555555'
        #table.setbgcolor()
        # Render the table first so rows and columns are set up
        # self.header_frame = tk.Frame(self)
        # self.header_frame.grid(row=0, column=0, sticky="ew")
        # for col_idx, col_name in enumerate(self.tree.model.columnNames):
        #     label = tk.Label(self.header_frame, text=col_name, bg="gray25", fg="white")
        #     label.grid(row=0, column=col_idx, sticky="ew", padx=1, pady=1)

        self.tree.grid(row=1, column=1, sticky="nsew")  # Use sticky="nsew" to fill cell
        self.tree.show()
        
        # Now color cells with value '1' in 'seagreen'
        for image_key, row_dict in data.items():
            # Get the row index for this image
            row_index = self.tree.model.getRecordIndex(image_key)

            for col_name, val in row_dict.items():
                if col_name == 'label':
                    col_index = self.tree.model.columnNames.index(col_name)
                    self.tree.model.setColorAt(row_index, col_index, color='#555555',key='bg')  
                if val == '1':
                    # Find column index
                    col_index = self.tree.model.columnNames.index(col_name)
                    # Set background color for this cell
                    self.tree.model.setColorAt(row_index, col_index, color='seagreen',key='bg')
                else:
                    col_index = self.tree.model.columnNames.index(col_name)
                    # Set background color for this cell
                    self.tree.model.setColorAt(row_index, col_index, color='#555555',key='bg')
        # Redraw the table to show the new colors
        

        self.tree.redraw()
        print('one')
        # self.tree.pack(side="right", fill="both", expand=True)
        # Sync the scrollbar:
        self.tree.configure(yscrollcommand=self.on_treeview_scroll)
        
        
        print('two')
        self.image_panel = LazyImagePanel(
            self,
            images_in_selected=images_in_selected,            
            row_height=100,
            hdf_path=hdf_path
        )
        # self.image_panel.pack(side="right", fill="both", expand=False)
        self.image_panel.grid(row=1, column=0, sticky="nsew")  # Use sticky="nsew"
        self.image_panel.canvas.config(yscrollcommand=self.on_canvas_scroll)
        self.on_treeview_scroll(*self.tree.yview())
        



    def on_scrollbar(self, *args):
        """ Scroll both widgets. """
        self.image_panel.set_yview(*args)
        # self.tree.set_yviews(*args)
        self.tree.yview(*args)  # Now targets only the body
        # Update the scrollbar handle from the tree
        self.vscroll.set(*self.tree.yview())

    def on_treeview_scroll(self, first, last):
        """ Called by the Treeview's yscrollcommand """
        self.vscroll.set(first, last)
        self.image_panel.set_yview("moveto", first)

    def on_canvas_scroll(self, first, last):
        """ Called by the Canvas's yscrollcommand """
        self.vscroll.set(first, last)
        # self.tree.set_yviews("moveto", first)
        self.tree.yview("moveto", first)