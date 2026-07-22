import tkinter as tk
from PIL import Image, ImageTk


class AppView(tk.Frame):
    """
    The main application view that contains all other views.
    """
    
    def __init__(self, master : tk.Tk):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)
        self.create_widgets()
        self.create_menu()

    def create_widgets(self):
        # Create a left frame for the board views
        self.left_frame = tk.Frame(self, bg="green")
        self.left_frame.pack(side="left", fill="both", expand=True)

        # Create a right frame for other content (e.g., settings, logs)
        self.right_frame = tk.Frame(self, bg="red")
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Image display area
        self.image_label = tk.Label(self.right_frame)
        self.image_label.pack(fill="both", expand=True)
        self._photo_image = None
        
    def create_menu(self):
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        self.source_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Source", menu=self.source_menu)

    def display_image(self, pil_image: Image.Image) -> None:
        """Display a PIL image in the right-frame image label.

        The PhotoImage is stored on the instance to avoid garbage collection.
        """
        if pil_image is None:
            self.image_label.config(image="")
            self._photo_image = None
            return

        # Resize to fit the label while preserving aspect ratio
        w = self.image_label.winfo_width() or pil_image.width
        h = self.image_label.winfo_height() or pil_image.height
        img = pil_image.copy()
        img.thumbnail((w, h), Image.LANCZOS)

        self._photo_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self._photo_image)