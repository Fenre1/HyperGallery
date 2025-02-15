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
    def __init__(self, parent, images_in_selected, hdf_path, data, model, **kwargs):
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
        self.tree.configure(yscrollcommand=self.on_treeview_scroll)
        
        
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
        new_order = self.tree.model.reclist
        self.image_panel.images_in_selected = new_order
        self.image_panel.row_count = len(new_order)
        total_height = self.image_panel.header_offset + self.image_panel.row_count * self.image_panel.row_height
        self.image_panel.canvas.config(scrollregion=(0, 0, 120, total_height))
        self.image_panel.active_rows.clear()
        self.image_panel.active_images.clear()
        self.image_panel.render_visible_rows()

class HyperedgeConfusionMatrix(tk.Frame):
    def __init__(self, parent, hyperedges, **kwargs):
        """
        :param parent: The parent widget (e.g. self.tab2)
        :param hyperedges: A dict mapping hyperedge names to a list of image IDs.
                           For example:
                           {
                               "edge_1": ["img1", "img3", "img5"],
                               "edge_2": ["img2", "img4"]
                           }
        """
        super().__init__(parent, **kwargs)
        self.hyperedges = hyperedges
        self.build_table()

    def build_table(self):
        # Build the raw data (overlap counts) for the confusion matrix.
        
        self.grid_columnconfigure(0, weight=0)  # Table body (expands)
        self.grid_columnconfigure(1, weight=0)  # Scrollbar (fixed width)
        self.grid_columnconfigure(2, weight=0)  # Image panel (expands)
        self.grid_rowconfigure(0, weight=0)     # Header (fixed height)
        self.grid_rowconfigure(1, weight=1)     # Body (expands)

        
        data = self.build_confusion_matrix_data()
        
        # Create the TableModel using the data dictionary.
        self.model = TableModel()
        self.model.importDict(data)
        
        hyperedge_keys = sorted(self.hyperedges.keys())
        self.model.rkeys = hyperedge_keys  # This tells the table to display these as row headers
        print(hyperedge_keys)
        # Create the TableCanvas. We use showkeynames=True so that the row headers (keys)
        # are displayed, and we give extra room for longer hyperedge names.
        self.table = TableCanvas(
            self,
            model=self.model,
            editable=False,
            showkeynamesinheader=True,
            rowheaderwidth=100,
            cellwidth=60,
            thefont=('Arial', 12),
            bgcolor      = '#555555',
        )
        self.table.bgcolor = '#555555'
        self.table.show()
        # Place the table so it fills the available space.
        self.table.grid(row=1, column=1, sticky="nsew")
        

        
        # Apply a heatmap-style coloring based on overlap percentage.
        self.apply_heatmap_coloring()

    def build_confusion_matrix_data(self):
        """
        Build a dictionary for tkintertable where:
          - Each key is a hyperedge name (row header).
          - The value is a dict mapping each hyperedge (column) to the number of shared images.
        """
        # Get the sorted hyperedge names (adjust ordering as desired)
        hyperedge_keys = sorted(self.hyperedges.keys())
        data = {}
        for row in hyperedge_keys:
            row_data = {}
            set_row = set(self.hyperedges[row])
            for col in hyperedge_keys:
                set_col = set(self.hyperedges[col])
                # Compute the overlap (number of shared images)
                count = len(set_row & set_col)
                row_data[col] = count
            data[row] = row_data
        return data

    def apply_heatmap_coloring(self):
        """
        Apply heatmap coloring using the harmonic mean of the two overlap ratios.
        For each pair of hyperedges HE1 and HE2, compute:
            p1 = overlap / len(HE1)
            p2 = overlap / len(HE2)
        and then the harmonic mean is:
            score = 2 * (p1 * p2) / (p1 + p2)   (if p1+p2 > 0, else 0)
        We assume the maximum possible score is 1.
        """
        hyperedge_keys = sorted(self.hyperedges.keys())

        # Optionally, you can compute the maximum score from the data.
        # Since p1 and p2 are between 0 and 1, the harmonic mean is at most 1.
        max_score = 0.0
        for row_key in hyperedge_keys:
            for col_key in hyperedge_keys:
                overlap = self.model.data[row_key][col_key]
                p1 = overlap / len(self.hyperedges[row_key])
                p2 = overlap / len(self.hyperedges[col_key])
                score = 2 * (p1 * p2) / (p1 + p2) if (p1 + p2) > 0 else 0
                if score > max_score:
                    max_score = score
        # If max_score is 0 (all overlaps are 0), avoid division by zero.
        if max_score == 0:
            max_score = 1

        # Now update the colors for each cell using the harmonic mean.
        for row_index, row_key in enumerate(hyperedge_keys):
            for col_index, col_key in enumerate(hyperedge_keys):
                overlap = self.model.data[row_key][col_key]  # recalc for current cell
                p1 = overlap / len(self.hyperedges[row_key])
                p2 = overlap / len(self.hyperedges[col_key])
                score = 2 * (p1 * p2) / (p1 + p2) if (p1 + p2) > 0 else 0
                # Here we use max_score (which will be 1 or less) to scale the color intensity.
                color = self.get_heatmap_color(score, max_score)
                self.model.setColorAt(row_index, col_index, color=color, key='bg')
        self.table.redraw()

    def get_heatmap_color(self, score, max_score):
        """
        Map a score (from 0 to max_score) to a color.
        When score == max_score, return full red (#ff0000).
        When score is 0, return white (#ffffff).
        """
        if max_score == 0:
            return '#ffffff'
        # Compute an intensity where a higher score gives a lower intensity for the green and blue channels.
        intensity = int(255 - (score / max_score) * 255)
        intensity = max(0, min(255, intensity))
        return f'#ff{intensity:02x}{intensity:02x}'




    # def apply_heatmap_coloring(self):
    #     """
    #     Apply heatmap coloring based on the percentage overlap.
    #     The percentage is computed for each cell as:
    #         percent = max( overlap/size(row), overlap/size(col) )
    #     This means that if one hyperedge is small and completely overlaps with another,
    #     that cell will have the maximum color intensity.
    #     """
    #     # First, compute the maximum percentage in the matrix.
    #     max_percent = 0.0
    #     hyperedge_keys = sorted(self.hyperedges.keys())
    #     for row in hyperedge_keys:
    #         for col in hyperedge_keys:
    #             overlap = self.model.data[row][col]
    #             size_row = len(self.hyperedges[row])
    #             size_col = len(self.hyperedges[col])
    #             # Compute percentage overlap for this cell.
    #             perc = max(overlap / size_row, overlap / size_col)
    #             if perc > max_percent:
    #                 max_percent = perc

    #     # Now iterate over each cell and set its background color based on the computed percentage.
    #     for row_index, row_key in enumerate(hyperedge_keys):
    #         for col_index, col_key in enumerate(hyperedge_keys):
    #             overlap = self.model.data[row_key][col_key]
    #             size_row = len(self.hyperedges[row_key])
    #             size_col = len(self.hyperedges[col_key])
    #             cell_percentage = max(overlap / size_row, overlap / size_col)
    #             color = self.get_heatmap_color(cell_percentage, max_percent)
    #             self.model.setColorAt(row_index, col_index, color=color, key='bg')
    #     self.table.redraw()

    # def get_heatmap_color(self, cell_percentage, max_percentage):
    #     """
    #     Returns a hex color string based on cell_percentage relative to max_percentage.
    #     - cell_percentage == 0  => white (#ffffff)
    #     - cell_percentage == max_percentage => full red (#ff0000)
    #     Intermediate values create a gradient from white to red.
    #     """
    #     if max_percentage == 0:
    #         return '#ffffff'
    #     # Scale intensity so that cell_percentage/max_percentage == 1 gives 0 intensity for green and blue.
    #     # Intensity goes from 255 (white) down to 0 (red).
    #     intensity = int(255 - (cell_percentage / max_percentage) * 255)
    #     # Clamp intensity between 0 and 255 just in case.
    #     intensity = max(0, min(255, intensity))
    #     return f'#ff{intensity:02x}{intensity:02x}'