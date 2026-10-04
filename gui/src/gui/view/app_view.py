import tkinter as tk

import ttkbootstrap as ttk

from gui.events.event_channel import EventChannel
from gui.model.layout import Layout, Panel, load_layout
from gui.view.dartboard_view import DartboardView
from gui.view.dartgame_view import DartGameView
from gui.view.panel_manager import PanelManager
from gui.view.source_view import SourceView


class AppView(ttk.Frame):
    """
    The main application view that contains all other views.

    The window is a menu bar followed by a :class:`PanelManager`, which decides
    which of the dartboard, camera feed and control panel are visible and where
    each of them is docked.
    """

    def __init__(self, master: tk.Tk):
        super().__init__(master, style="Card.TFrame", padding=0)
        self.master = master

        self._event_channel = EventChannel()
        self._layout = load_layout()

        self.create_widgets()

    @property
    def event_channel(self) -> EventChannel:
        """Bus shared with the controllers for panel lifecycle notifications."""
        return self._event_channel

    def create_widgets(self) -> None:
        """Build the menu bar and the panel area."""
        self.menu_frame = ttk.Frame(self.master)
        self.menu_frame.pack(fill="x", side="top")

        self.menu_separator = ttk.Separator(self.master, orient="horizontal")
        self.menu_separator.pack(fill="x", side="top")

        self.pack(fill="both", expand=True)

        self.panel_manager = PanelManager(self, self._event_channel)
        self.panel_manager.pack(fill="both", expand=True)
        self.panel_manager.register(Panel.DARTBOARD, DartboardView)
        self.panel_manager.register(Panel.SOURCE, SourceView)
        self.panel_manager.register(Panel.CONTROL, DartGameView)
        self.panel_manager.build(self._layout)

    @property
    def layout(self) -> Layout:
        """Return the layout currently on screen."""
        return self._layout

    def apply_layout(self, layout: Layout) -> None:
        """Dock the panels according to *layout*."""
        self._layout = layout
        self.panel_manager.apply(layout)

    @property
    def dartboard_view(self) -> DartboardView:
        """The live dartboard panel, which moves when the layout changes."""
        widget = self.panel_manager.widget(Panel.DARTBOARD)
        assert isinstance(widget, DartboardView)
        return widget

    @property
    def source_view(self) -> SourceView:
        """The live camera feed panel, which moves when the layout changes."""
        widget = self.panel_manager.widget(Panel.SOURCE)
        assert isinstance(widget, SourceView)
        return widget

    @property
    def game_view(self) -> DartGameView:
        """The live control panel, which moves when the layout changes."""
        widget = self.panel_manager.widget(Panel.CONTROL)
        assert isinstance(widget, DartGameView)
        return widget
