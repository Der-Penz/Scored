from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np
import ttkbootstrap as ttk

import gui.events.event_types as events
from gui.controller.annotation_editor import AnnotationEditor, key_targets_annotation
from gui.events.event_channel import EventChannel
from gui.model.args import AppMode
from gui.model.label_session import LabelSession
from gui.model.layout import Panel
from gui.model.model import AppModel
from gui.protocols.controller import BaseController
from gui.view.annotation.annotation_view import AnnotationView
from gui.view.app_view import AppView


class LabelController(BaseController):
    """Label the darts on a folder of images, one image at a time.

    The panel shows a single sample per screen: an existing ``annotation.json``
    next to the image is loaded, an image without one starts unlabeled. Clicks
    work like in the game panel, except that a frame may hold any number of
    darts, so a new dart is added once every dart on screen is labeled.

    The labels are written next to the image when the user leaves it, presses
    Save, opens another folder or switches back to the game. In a read-only
    session nothing is written at all; the panel then only shows what is
    already on disk.
    """

    def __init__(self, view: AppView, model: AppModel, event_channel: EventChannel):
        super().__init__(view, model, event_channel)

        self.session = LabelSession(read_only=model.config.read_only)
        self.editor = AnnotationEditor(allow_new_darts=True)
        self._current_image: np.ndarray | None = None
        self._keys_bound = False

    @property
    def annotation_view(self) -> AnnotationView:
        return self._view.annotation_view

    @property
    def _active(self) -> bool:
        """Whether the label mode owns the annotation panel right now."""
        return self._model.mode is AppMode.LABEL

    def bind_menu(self, menu: ttk.Menu) -> None:
        menu.add_command(label="Open Folder…", command=self.open_folder, accelerator="Ctrl+O")
        menu.add_command(label="Reload Folder", command=self.reload, accelerator="F5")
        menu.add_separator()
        menu.add_command(label="Previous Image", command=lambda: self._step(-1), accelerator="PgUp")
        menu.add_command(label="Next Image", command=lambda: self._step(1), accelerator="PgDn")
        menu.add_command(label="Save Labels", command=self._save, accelerator="Ctrl+S")

    def bind_components(self) -> None:
        view = self.annotation_view
        view.on_previous_clicked = lambda: self._step(-1)
        view.on_next_clicked = lambda: self._step(1)
        view.on_save_clicked = self._save
        view.on_delete_clicked = self._on_delete
        view.on_point_placed = self._on_point_placed
        view.on_point_selected = self._on_point_selected
        view.on_point_moved = self._on_point_moved
        if not self._keys_bound:
            self._keys_bound = True
            self._view.master.bind_all("<Key>", self._on_key)
            self._view.master.bind_all("<Control-o>", lambda _: self.open_folder())
            self._view.master.bind_all("<F5>", lambda _: self.reload())

    def start(self) -> None:
        self._event_channel.subscribe(events.PanelRebound, self._on_panel_rebound)
        self._event_channel.subscribe(events.ModeChanged, self._on_mode_changed)

        if not self._active:
            return
        self.bind_components()
        directory = self._model.config.label_dir
        if directory is not None:
            self.open(directory)
            return
        self._refresh()

    def _on_key(self, event: tk.Event) -> str | None:
        if not self._active:
            return None
        key = (getattr(event, "keysym", "") or "").lower()
        if not key_targets_annotation(event.widget, self.annotation_view):
            return None
        if int(event.state) & 0x0004:
            if key == "s":
                self._save()
                return "break"
            return None
        if key in ("prior", "kp_prior"):
            self._step(-1)
        elif key in ("next", "kp_next"):
            self._step(1)
        elif key in ("a", "s", "d"):
            self._select_dart({"a": 0, "s": 1, "d": 2}[key])
        elif key in ("delete", "kp_delete"):
            self._on_delete()
        else:
            return None
        return "break"

    def open_folder(self) -> None:
        """Ask for a folder and label the images in it."""
        if not self._active:
            return
        directory = filedialog.askdirectory(title="Select a folder of images", parent=self._view)
        if not directory:
            return
        self.open(Path(directory))

    def open(self, directory: Path) -> None:
        """Label the images below *directory*, replacing the current folder."""
        self.save_if_dirty()
        count = self.session.open(directory)
        self._load_current()
        self._refresh()
        if count:
            self.annotation_view.set_status(f"{count} images")
        else:
            self.annotation_view.set_status("No images found in this folder.")

    def reload(self) -> None:
        """Scan the current folder again, keeping the image that is on screen."""
        if not self._active:
            return
        directory = self.session.directory
        if directory is None:
            self.annotation_view.set_status("No folder open.")
            return
        position = self.session.index
        self.save_if_dirty()
        self.session.open(directory)
        self.session.index = min(position, max(len(self.session.samples) - 1, 0))
        self._load_current()
        self._refresh()
        self.annotation_view.set_status(f"{len(self.session.samples)} images")

    def _step(self, offset: int) -> None:
        """Move to the previous or next image, saving the current one first."""
        if not self._active:
            return
        self.save_if_dirty()
        if not self.session.step(offset):
            self.annotation_view.set_status("Last image." if offset > 0 else "First image.")
            return
        self.annotation_view.set_status("")
        self._load_current()
        self._refresh()

    def _load_current(self) -> None:
        """Point the editor at the sample on screen and load its frame."""
        self.editor.set_annotation(self.session.annotation)
        self._current_image = self.session.image()
        path = self.session.image_path
        if self._current_image is None and path is not None:
            self.annotation_view.set_status(f"Could not read {path.name}.")

    def _refresh(self) -> None:
        """Repaint the panel from the session."""
        view = self.annotation_view
        view.set_toolbar_mode("label")
        annotation = self.session.annotation
        if annotation is None:
            view.clear()
            view.set_session(self.session.position, self.session.dirty, self.session.read_only)
            self._apply_banner()
            return

        view.set_session(self.session.position, self.session.dirty, self.session.read_only)
        view.show_image(
            self._current_image,
            self.editor.points(),
            self.editor.rows(),
            self.editor.selected_entry,
            self._header(),
        )
        self._apply_banner()

    def _header(self) -> str:
        """Describe the image on screen and how much of it is labeled."""
        annotation = self.session.annotation
        path = self.session.image_path
        name = path.name if path is not None else ""
        recorded = ""
        if annotation is not None and annotation.round is not None:
            recorded = f"Round {annotation.round} · Throw {annotation.throw} · "
        return f"{recorded}{name} · {self.editor.summary()}"

    def _apply_banner(self) -> None:
        """Show which folder is being labeled and whether it is written to."""
        view = self.annotation_view
        directory = self.session.directory
        if directory is None:
            view.set_banner("No folder open — Labels ▸ Open Folder…", "secondary")
        elif self.session.read_only:
            view.set_banner(f"Read only: {directory} — labels are not saved.", "warning")
        else:
            count = len(self.session.samples)
            view.set_banner(f"Labeling {count} images in {directory}", "success")

    def save_if_dirty(self) -> None:
        """Write the labels of the image on screen when they changed."""
        if self.session.dirty:
            self.session.save()

    def _save(self) -> None:
        """Write the labels of the image on screen."""
        if self.session.read_only:
            self.annotation_view.set_status("Read-only session, nothing is saved.")
            return
        if not self.session.dirty:
            self.annotation_view.set_status("Nothing to save.")
            return
        target = self.session.save()
        if target is None:
            self.annotation_view.set_status("Could not write the labels, see the log.")
            return
        self.annotation_view.set_status(f"Saved {target.name}.")
        self._refresh()

    def _on_point_placed(self, x: float, y: float) -> None:
        if not self._active:
            return
        message = self.editor.place(x, y)
        if message is None:
            self.session.dirty = True
        self._refresh()
        if message is not None:
            self.annotation_view.set_status(message)

    def _on_point_selected(self, index: int) -> None:
        if not self._active:
            return
        self.editor.select(index)
        self._refresh()

    def _on_point_moved(self, index: int, kind: str, x: float, y: float) -> None:
        if not self._active:
            return
        self.editor.move(index, kind, x, y)
        self.session.dirty = True
        self._refresh()

    def _on_delete(self) -> None:
        if not self._active:
            return
        changed = self.editor.draft_active or self.editor.selected_entry is not None
        self.editor.delete()
        if changed:
            self.session.dirty = True
        self._refresh()

    def _select_dart(self, index: int) -> None:
        """Select the dart at *index* (0, 1 or 2) in the current image."""
        self._on_point_selected(index, kind=None)

    def _on_panel_rebound(self, event: events.PanelRebound) -> None:
        if event.panel is Panel.ANNOTATION and self._active:
            self.bind_components()
            self._refresh()

    def _on_mode_changed(self, event: events.ModeChanged) -> None:
        if event.mode is AppMode.LABEL:
            self.bind_components()
            directory = self._model.config.label_dir
            if not self.session.samples and directory is not None:
                self.open(directory)
                return
            self._refresh()
            return
        self.save_if_dirty()
        self.session.dirty = False
        logging.debug(f"Left label mode with folder {self.session.directory}")
