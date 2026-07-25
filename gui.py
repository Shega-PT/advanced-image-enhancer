from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from image_enhancer import EnhancementConfig, EnhancementMode
from image_enhancer.enhancers import (
    adaptive_sharpening,
    apply_non_local_means_denoising,
    apply_super_resolution_effect,
    auto_white_balance,
    enhance_contrast_local,
    enhance_low_light,
    enhance_saturation,
    resize_to_target,
    save_image,
)
from image_enhancer.utils import get_target_size

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def _cv2_to_tk(cv_img: np.ndarray, max_size: tuple = (420, 420)) -> Optional[ImageTk.PhotoImage]:
    if cv_img is None or cv_img.size == 0:
        return None
    h, w = cv_img.shape[:2]
    scale = min(max_size[0] / w, max_size[1] / h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        display = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        display = cv_img.copy()
    return ImageTk.PhotoImage(Image.fromarray(display))


class ImageEnhancerGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Advanced Image Enhancer")
        self.root.geometry("1140x780")
        self.root.minsize(900, 650)

        self.original_image: Optional[np.ndarray] = None
        self.preview_image: Optional[np.ndarray] = None
        self.current_input_path: Optional[Path] = None
        self.current_output_path: Optional[Path] = None
        self._lock = threading.Lock()

        self._build_menu()
        self._build_display()
        self._build_controls()
        self._build_buttons()
        self._build_status()

        self.update_status("Ready. Open an image to begin.")

    # ── Menu ──────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        bar = tk.Menu(self.root)
        self.root.config(menu=bar)
        file_menu = tk.Menu(bar, tearoff=0)
        file_menu.add_command(label="Open Image...", command=self._open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self._save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self._save_image_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        bar.add_cascade(label="File", menu=file_menu)
        self.root.bind("<Control-o>", lambda e: self._open_image())
        self.root.bind("<Control-s>", lambda e: self._save_image())

    # ── Display ───────────────────────────────────────────────────────────

    def _build_display(self) -> None:
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        orig = ttk.LabelFrame(frame, text=" Original ", padding=4)
        orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.orig_label = ttk.Label(orig, text="No image loaded", anchor=tk.CENTER)
        self.orig_label.pack(fill=tk.BOTH, expand=True)

        prev = ttk.LabelFrame(frame, text=" Preview ", padding=4)
        prev.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.preview_label = ttk.Label(prev, text="No preview", anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    # ── Controls ──────────────────────────────────────────────────────────

    def _build_controls(self) -> None:
        ctrl = ttk.LabelFrame(self.root, text=" Controls ", padding=8)
        ctrl.pack(fill=tk.X, padx=8, pady=8)

        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Mode:").pack(side=tk.LEFT, padx=(0, 4))
        self.mode_var = tk.StringVar(value="natural")
        w = ttk.Combobox(row1, textvariable=self.mode_var, state="readonly", width=12)
        w["values"] = [m.value for m in EnhancementMode]
        w.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(row1, text="Size:").pack(side=tk.LEFT, padx=(0, 4))
        self.size_var = tk.StringVar(value="1080p")
        w = ttk.Combobox(row1, textvariable=self.size_var, state="readonly", width=10)
        w["values"] = ["720p", "1080p", "1440p", "4k", "original"]
        w.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(row1, text="Fit:").pack(side=tk.LEFT, padx=(0, 4))
        self.fit_var = tk.StringVar(value="stretch")
        w = ttk.Combobox(row1, textvariable=self.fit_var, state="readonly", width=10)
        w["values"] = ["stretch", "crop", "pad"]
        w.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(row1, text="Format:").pack(side=tk.LEFT, padx=(0, 4))
        self.fmt_var = tk.StringVar(value="png")
        w = ttk.Combobox(row1, textvariable=self.fmt_var, state="readonly", width=8)
        w["values"] = ["png", "jpeg", "webp"]
        w.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(row1, text="Workers:").pack(side=tk.LEFT, padx=(0, 4))
        self.workers_var = tk.StringVar(value="1")
        w = ttk.Spinbox(row1, from_=1, to=8, textvariable=self.workers_var, width=4)
        w.pack(side=tk.LEFT)

        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=4)

        self.denoise_var = tk.DoubleVar(value=3.0)
        self._slider(row2, "Denoise:", self.denoise_var, 0, 10, 0.5)
        self.sharpen_var = tk.DoubleVar(value=1.2)
        self._slider(row2, "Sharpen:", self.sharpen_var, 1.0, 2.0, 0.1)
        self.contrast_var = tk.DoubleVar(value=1.1)
        self._slider(row2, "Contrast:", self.contrast_var, 1.0, 2.0, 0.05)
        self.saturation_var = tk.DoubleVar(value=1.1)
        self._slider(row2, "Saturation:", self.saturation_var, 1.0, 2.0, 0.05)

        row3 = ttk.Frame(ctrl)
        row3.pack(fill=tk.X, pady=2)

        self.wb_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text="Auto White Balance", variable=self.wb_var).pack(side=tk.LEFT, padx=(0, 16))
        self.light_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text="Low Light Correction", variable=self.light_var).pack(side=tk.LEFT, padx=(0, 16))
        self.preserve_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="Preserve Original", variable=self.preserve_var).pack(side=tk.LEFT)

    def _slider(self, parent: ttk.Frame, label: str, var: tk.DoubleVar,
                min_val: float, max_val: float, step: float) -> None:
        f = ttk.Frame(parent)
        f.pack(side=tk.LEFT, padx=(0, 12), fill=tk.X, expand=True)
        ttk.Label(f, text=label).pack(anchor=tk.W)
        sf = ttk.Frame(f)
        sf.pack(fill=tk.X)
        ttk.Scale(sf, from_=min_val, to=max_val, variable=var,
                  orient=tk.HORIZONTAL,
                  command=lambda v, vv=var: self._update_label(vv)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        lbl = ttk.Label(sf, text=f"{var.get():.1f}", width=5)
        lbl.pack(side=tk.LEFT, padx=(4, 0))
        var._label = lbl

    def _update_label(self, var: tk.DoubleVar) -> None:
        if hasattr(var, '_label'):
            var._label.config(text=f"{var.get():.1f}")

    # ── Action Buttons ────────────────────────────────────────────────────

    def _build_buttons(self) -> None:
        bf = ttk.Frame(self.root)
        bf.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.magic_btn = ttk.Button(bf, text="Magic Enhance",
                                    command=self._magic_enhance)
        self.magic_btn.pack(side=tk.LEFT, padx=(0, 8), ipadx=10, ipady=4)

        iframe = ttk.Frame(bf)
        iframe.pack(side=tk.LEFT, fill=tk.X, expand=True)

        for text, cmd, tip in [
            ("Resize", self._op_resize, "Resize only"),
            ("Denoise", self._op_denoise, "Denoise only"),
            ("Sharpen", self._op_sharpen, "Sharpen only"),
            ("Contrast", self._op_contrast, "Contrast only"),
            ("Saturation", self._op_saturation, "Saturation only"),
            ("White Bal.", self._op_wb, "White balance only"),
            ("Low Light", self._op_low_light, "Low light correction only"),
        ]:
            btn = ttk.Button(iframe, text=text, command=cmd, width=11)
            btn.pack(side=tk.LEFT, padx=1)

    # ── Status ────────────────────────────────────────────────────────────

    def _build_status(self) -> None:
        self.status_var = tk.StringVar(value="Ready")
        bar = ttk.Label(self.root, textvariable=self.status_var,
                        relief=tk.SUNKEN, anchor=tk.W)
        bar.pack(fill=tk.X, side=tk.BOTTOM, padx=8, pady=(0, 8))

    def update_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ── File Ops ──────────────────────────────────────────────────────────

    def _open_image(self, event=None) -> None:
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")])
        if not path:
            return
        self.current_input_path = Path(path)
        img = cv2.imread(str(self.current_input_path))
        if img is None:
            messagebox.showerror("Error", f"Could not load: {path}")
            return
        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.preview_image = self.original_image.copy()
        self._update_displays()
        h, w = self.original_image.shape[:2]
        self.update_status(f"Loaded: {self.current_input_path.name} ({w}x{h})")

    def _save_image(self, event=None) -> None:
        if self.preview_image is None:
            return
        if self.current_output_path is None:
            self._save_image_as()
            return
        save_image(self.preview_image, self.current_output_path)
        self.update_status(f"Saved: {self.current_output_path.name}")

    def _save_image_as(self) -> None:
        if self.preview_image is None:
            messagebox.showinfo("Info", "No image to save.")
            return
        ext = self.fmt_var.get()
        default = f"enhanced_{self.mode_var.get()}.{ext}"
        path = filedialog.asksaveasfilename(
            title="Save Enhanced Image",
            defaultextension=f".{ext}",
            filetypes=[(f"{ext.upper()}", f"*.{ext}"), ("All", "*.*")],
            initialfile=default)
        if not path:
            return
        self.current_output_path = Path(path)
        self._save_image()

    def _update_displays(self) -> None:
        if self.original_image is not None:
            tk_img = _cv2_to_tk(self.original_image)
            if tk_img:
                self.orig_label.config(image=tk_img)
                self.orig_label.image = tk_img
        if self.preview_image is not None:
            tk_img = _cv2_to_tk(self.preview_image)
            if tk_img:
                self.preview_label.config(image=tk_img)
                self.preview_label.image = tk_img

    def _build_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            target_size=get_target_size(self.size_var.get()),
            mode=EnhancementMode(self.mode_var.get()),
            denoise_strength=self.denoise_var.get(),
            sharpening_strength=self.sharpen_var.get(),
            contrast_boost=self.contrast_var.get(),
            saturation_boost=self.saturation_var.get(),
            preserve_original=self.preserve_var.get(),
            output_format=self.fmt_var.get(),
            fit_mode=self.fit_var.get(),
            auto_wb=self.wb_var.get(),
            low_light_correction=self.light_var.get(),
            workers=int(self.workers_var.get()),
        )

    def _mode_params(self) -> dict:
        return self._build_config().get_mode_params()

    # ── Background Processing ─────────────────────────────────────────────

    def _process(self, func: Callable[[np.ndarray], np.ndarray], desc: str) -> None:
        if self.original_image is None:
            messagebox.showinfo("Info", "Open an image first.")
            return

        def task():
            if not self._lock.acquire(blocking=False):
                self.update_status("Already processing...")
                return
            try:
                self.update_status(f"Processing: {desc}...")
                self.preview_image = func(self.original_image.copy())
                self.root.after(0, self._update_displays)
                self.update_status(f"Done: {desc}")
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.update_status(f"Error: {desc}")
            finally:
                self._lock.release()

        threading.Thread(target=task, daemon=True).start()

    # ── Magic Enhance ─────────────────────────────────────────────────────

    def _magic_enhance(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Info", "Open an image first.")
            return

        def task():
            if not self._lock.acquire(blocking=False):
                self.update_status("Already processing...")
                return
            try:
                config = self._build_config()
                img = self.original_image.copy()
                params = config.get_mode_params()

                self.update_status("Resizing...")
                img = resize_to_target(img, config.target_size, config.fit_mode)
                if config.low_light_correction:
                    self.update_status("Low light correction...")
                    img = enhance_low_light(img)
                if config.auto_wb:
                    self.update_status("White balance...")
                    img = auto_white_balance(img)
                self.update_status("Denoising...")
                img = apply_non_local_means_denoising(img, params)
                if config.mode in (EnhancementMode.SHARP, EnhancementMode.LANDSCAPE):
                    self.update_status("Super resolution...")
                    img = apply_super_resolution_effect(img, config)
                self.update_status("Contrast enhancement...")
                img = enhance_contrast_local(img, params)
                self.update_status("Saturation...")
                img = enhance_saturation(img, params)
                self.update_status("Sharpening...")
                img = adaptive_sharpening(img, params)

                self.preview_image = img
                self.root.after(0, self._update_displays)
                self.update_status("Magic Enhance complete!")
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.update_status("Failed.")
            finally:
                self._lock.release()

        threading.Thread(target=task, daemon=True).start()

    # ── Individual Ops ────────────────────────────────────────────────────

    def _op_resize(self) -> None:
        cfg = self._build_config()
        self._process(lambda img: resize_to_target(img, cfg.target_size, cfg.fit_mode), "Resize")

    def _op_denoise(self) -> None:
        p = self._mode_params()
        self._process(lambda img: apply_non_local_means_denoising(img, p), "Denoise")

    def _op_sharpen(self) -> None:
        p = self._mode_params()
        self._process(lambda img: adaptive_sharpening(img, p), "Sharpen")

    def _op_contrast(self) -> None:
        p = self._mode_params()
        self._process(lambda img: enhance_contrast_local(img, p), "Contrast")

    def _op_saturation(self) -> None:
        p = self._mode_params()
        self._process(lambda img: enhance_saturation(img, p), "Saturation")

    def _op_wb(self) -> None:
        self._process(auto_white_balance, "White Balance")

    def _op_low_light(self) -> None:
        self._process(enhance_low_light, "Low Light")


def main() -> None:
    if not HAS_PIL:
        print("Pillow is required. Install: pip install Pillow")
        return
    root = tk.Tk()
    ImageEnhancerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
