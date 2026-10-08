from __future__ import annotations

import tkinter as tk

import ttkbootstrap as ttk


class AnnotationView(ttk.Frame):

    def __init__(self, master: tk.Misc):
        super().__init__(master, style="Card.TFrame", padding=4)