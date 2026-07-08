import os
import sys
import json
import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch, resample, hilbert
from scipy.ndimage import label
from scipy.fft import rfft, rfftfreq
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tdt

# ==============================================================================
# IMPORT CORE PROCESSING MODULES
# ==============================================================================
from eeg_processing import (
    read_tdt,
    polyphase_resample,
    apply_notch_filter,
    calculate_eeg_features,
    apply_asymmetric_heuristics,
    load_central_manual_thresholds
)

# ==============================================================================
# GUI APP IMPLEMENTATION
# ==============================================================================

class EEGThresholdApp:
    def __init__(self, root, folders):
        self.root = root
        self.folders = [Path(f) for f in folders if Path(f).exists()]
        
        # State variables
        self.current_folder = None
        self.t_disp = None
        self.sig_disp = None
        self.env_disp = None
        self.log_feat = None
        self.fs_disp = 1000.0
        
        # Keep full resolution arrays for switching filter modes
        self.sig_raw = None
        self.sig_sw = None
        self.sig_bs = None
        self.sig_filt = None
        
        self.supp_epochs = []
        self.burst_epochs = []
        self.addition_order = []
        self.thresh_val = None
        self.current_win_size = 60.0  # Keep track of active window size in seconds
        
        self.shading_patches = []
        self.epoch_patches = []
        self.updating_from_event = False
        
        # Configure root window style
        self.root.geometry("1400x950")
        self.root.minsize(1000, 700)
        
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Small.TButton", font=("Segoe UI", 9), padding=2)
        
        # Bind keyboard shortcuts for scrolling
        self.root.bind("<Left>", self.scroll_left)
        self.root.bind("<Right>", self.scroll_right)
        self.root.bind("<Shift-Left>", lambda e: self.scroll_left(factor=0.8))
        self.root.bind("<Shift-Right>", lambda e: self.scroll_right(factor=0.8))
        
        # Layout Frames
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left Panel (Recordings & Finish)
        self.left_frame = ttk.Frame(self.main_paned, width=300, padding=10)
        self.main_paned.add(self.left_frame, weight=1)
        
        # Right Panel (Plots & Controls)
        self.right_frame = ttk.Frame(self.main_paned, padding=10)
        self.main_paned.add(self.right_frame, weight=5)
        
        self.build_left_panel()
        self.build_right_panel()
        
        # Initialize Listbox status
        self.update_listbox_status()
        
        # Load the first recording if available
        if self.folders:
            self.listbox.selection_set(0)
            self.load_recording(self.folders[0])
            
        # Set default sash position for the resizable right panel
        self.root.update()
        try:
            self.right_paned.sashpos(0, 200)
        except Exception:
            pass
            
    def build_left_panel(self):
        lbl = ttk.Label(self.left_frame, text="Recordings in Batch", font=("Segoe UI", 12, "bold"))
        lbl.pack(anchor=tk.W, pady=(0, 5))
        
        # Folder addition controls
        import_btn_frame = ttk.Frame(self.left_frame)
        import_btn_frame.pack(fill=tk.X, pady=(0, 5))
        
        add_btn = ttk.Button(import_btn_frame, text="Add Folder 📁", command=self.gui_add_folder)
        add_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        scan_btn = ttk.Button(import_btn_frame, text="Scan Parent 🔍", command=self.gui_scan_parent)
        scan_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))
        
        clear_btn = ttk.Button(import_btn_frame, text="Clear List ❌", command=self.gui_clear_folders)
        clear_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Listbox with Scrollbar
        list_container = ttk.Frame(self.left_frame)
        list_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(
            list_container, 
            font=("Segoe UI", 10), 
            selectmode=tk.SINGLE, 
            yscrollcommand=scrollbar.set,
            activestyle='none',
            highlightthickness=0
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)
        
        # Status Label
        self.status_label = ttk.Label(
            self.left_frame, 
            text="Select a recording to begin.", 
            font=("Segoe UI", 10, "italic"),
            wraplength=250
        )
        self.status_label.pack(fill=tk.X, pady=10)
        
        # Finish Button
        self.run_btn = ttk.Button(
            self.left_frame, 
            text="Finish Annotation ✓", 
            command=self.start_pipeline_run
        )
        self.run_btn.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
    def build_right_panel(self):
        # Right PanedWindow (Vertical)
        self.right_paned = ttk.PanedWindow(self.right_frame, orient=tk.VERTICAL)
        self.right_paned.pack(fill=tk.BOTH, expand=True)
        
        # Control Panel (Top)
        self.control_frame = ttk.LabelFrame(self.right_paned, text="Annotation Controls", padding=15)
        self.right_paned.add(self.control_frame, weight=1)
        
        # Grid layout for controls (5 columns to fit Zoom Controls)
        self.control_frame.columnconfigure(0, weight=2)
        self.control_frame.columnconfigure(1, weight=2)
        self.control_frame.columnconfigure(2, weight=2)
        self.control_frame.columnconfigure(3, weight=2)
        self.control_frame.columnconfigure(4, weight=1)
        
        # Active recording name
        self.rec_title = ttk.Label(self.control_frame, text="No Recording Loaded", font=("Segoe UI", 10, "bold"))
        self.rec_title.grid(row=0, column=0, columnspan=5, sticky=tk.W, pady=(0, 2))
        
        # Epoch Selection Container
        epoch_select_frame = ttk.LabelFrame(self.control_frame, text="Mark Epochs (Uses Current Zoom/X-limits)", padding=2)
        epoch_select_frame.grid(row=1, column=0, columnspan=4, sticky=tk.EW, padx=5, pady=2)
        
        self.add_supp_btn = ttk.Button(epoch_select_frame, text="+ Add Suppression", command=self.add_suppression_epoch, style="Small.TButton")
        self.add_supp_btn.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.supp_label = ttk.Label(epoch_select_frame, text="Suppression: None", font=("Segoe UI", 9))
        self.supp_label.pack(side=tk.LEFT, padx=(0, 20))
        
        self.add_burst_btn = ttk.Button(epoch_select_frame, text="+ Add Burst", command=self.add_burst_epoch, style="Small.TButton")
        self.add_burst_btn.pack(side=tk.LEFT, padx=5, pady=1)
        
        self.burst_label = ttk.Label(epoch_select_frame, text="Burst: None", font=("Segoe UI", 9))
        self.burst_label.pack(side=tk.LEFT, padx=10)
        
        # Threshold Tuning Row
        thresh_container = ttk.LabelFrame(self.control_frame, text="Threshold Adjustment (Log10 Power)", padding=2)
        thresh_container.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        self.thresh_slider = ttk.Scale(
            thresh_container, 
            orient=tk.HORIZONTAL, 
            from_=-8, 
            to=-1, 
            command=self.on_slider_change,
            state='disabled'
        )
        self.thresh_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.thresh_entry = ttk.Entry(thresh_container, width=10, state='disabled')
        self.thresh_entry.pack(side=tk.RIGHT, padx=5)
        self.thresh_entry.bind("<Return>", self.on_entry_change)
        self.thresh_entry.bind("<FocusOut>", self.on_entry_change)
        
        # Epoch Management Buttons
        epoch_mgmt_container = ttk.Frame(self.control_frame)
        epoch_mgmt_container.grid(row=2, column=2, columnspan=2, sticky=tk.NSEW, padx=5, pady=2)
        
        self.save_btn = ttk.Button(epoch_mgmt_container, text="Save Threshold 💾", command=self.save_threshold_to_json, state='disabled', style="Small.TButton")
        self.save_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        
        self.undo_btn = ttk.Button(epoch_mgmt_container, text="Undo Last ↶", command=self.undo_last_epoch, style="Small.TButton")
        self.undo_btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 2))
        
        self.delete_epochs_btn = ttk.Button(epoch_mgmt_container, text="Delete All ❌", command=self.delete_epochs, style="Small.TButton")
        self.delete_epochs_btn.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))
        
        # General Zoom & Filter Control Panel (X, Y and Filters)
        zoom_container = ttk.LabelFrame(self.control_frame, text="Signal View / Zoom", padding=2)
        zoom_container.grid(row=1, column=4, rowspan=2, sticky=tk.NSEW, padx=5, pady=2)
        
        # Row 1: Filter Mode (Compact Label + Combobox)
        filter_frame = ttk.Frame(zoom_container)
        filter_frame.pack(side=tk.TOP, fill=tk.X, expand=True, pady=1)
        ttk.Label(filter_frame, text="Filter:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=2)
        self.filter_mode_var = tk.StringVar(value="0.5-100 Hz (Filtered)")
        self.filter_mode_menu = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_mode_var,
            values=["0.5-100 Hz (Filtered)", "Raw EEG"],
            width=18,
            state="readonly"
        )
        self.filter_mode_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.filter_mode_menu.bind("<<ComboboxSelected>>", self.on_filter_mode_change)
        
        # Row 2: Zoom X and Zoom Y side-by-side
        zoom_btn_frame = ttk.Frame(zoom_container)
        zoom_btn_frame.pack(side=tk.TOP, fill=tk.X, expand=True, pady=1)
        
        # X Zoom Buttons
        x_frame = ttk.Frame(zoom_btn_frame)
        x_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Label(x_frame, text="X:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=1)
        self.x_zoom_in_btn = ttk.Button(x_frame, text="In", command=self.zoom_x_in, width=3, style="Small.TButton")
        self.x_zoom_in_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.x_zoom_out_btn = ttk.Button(x_frame, text="Out", command=self.zoom_x_out, width=3, style="Small.TButton")
        self.x_zoom_out_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        # Y Zoom Buttons
        y_frame = ttk.Frame(zoom_btn_frame)
        y_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Label(y_frame, text="Y:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=1)
        self.y_zoom_in_btn = ttk.Button(y_frame, text="In", command=self.zoom_y_in, width=3, style="Small.TButton")
        self.y_zoom_in_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.y_zoom_out_btn = ttk.Button(y_frame, text="Out", command=self.zoom_y_out, width=3, style="Small.TButton")
        self.y_zoom_out_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        
        # Row 3: Action Buttons side-by-side
        action_btn_frame = ttk.Frame(zoom_container)
        action_btn_frame.pack(side=tk.TOP, fill=tk.X, expand=True, pady=1)
        
        # Auto Fit Y button
        self.y_auto_btn = ttk.Button(action_btn_frame, text="Auto Fit Y", command=self.auto_y_scale, style="Small.TButton")
        self.y_auto_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Save Plot Image button
        self.save_img_btn = ttk.Button(action_btn_frame, text="Save Plot Image 📷", command=self.save_plot_image, style="Small.TButton")
        self.save_img_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # Plot Frame (Middle/Bottom)
        self.plot_frame = ttk.Frame(self.right_paned)
        self.right_paned.add(self.plot_frame, weight=4)
        
        # Create Matplotlib Figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2, sharex=self.ax1)
        self.ax3 = self.fig.add_subplot(3, 1, 3)
        self.fig.subplots_adjust(hspace=0.55, top=0.95, bottom=0.1)
        
        # 1. PACK BOTTOM CONTROLS FIRST (fixes Tkinter clipping bugs)
        self.nav_frame = ttk.Frame(self.plot_frame)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Time Scroll Slider
        ttk.Label(self.nav_frame, text="Scroll Time:").pack(side=tk.LEFT, padx=5)
        self.time_slider = ttk.Scale(
            self.nav_frame, 
            orient=tk.HORIZONTAL, 
            from_=0, 
            to=100, 
            command=self.on_time_slider_change
        )
        self.time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Window Size Select (added 1s, 2s, and 5s selections)
        ttk.Label(self.nav_frame, text="Window:").pack(side=tk.LEFT, padx=5)
        self.win_size_var = tk.StringVar(value="60s")
        self.win_size_menu = ttk.Combobox(
            self.nav_frame, 
            textvariable=self.win_size_var, 
            values=["1s", "2s", "5s", "10s", "30s", "60s", "120s", "300s", "600s", "Full"], 
            width=8,
            state="readonly"
        )
        self.win_size_menu.pack(side=tk.LEFT, padx=5)
        self.win_size_menu.bind("<<ComboboxSelected>>", self.on_win_size_change)
        
        # 2. CREATE CANVAS
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        
        # 3. CREATE MATPLOTLIB TOOLBAR (automatically packs at the bottom of master)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        # 4. PACK CANVAS LAST WITH EXPAND
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Connect xlim changed callback
        self.ax1.callbacks.connect('xlim_changed', self.on_xlim_changed)
        
    def gui_add_folder(self):
        folder = filedialog.askdirectory(title="Select TDT block folder")
        if folder:
            path = Path(folder)
            if path.exists() and path not in self.folders:
                self.folders.append(path)
                self.update_listbox_status()
                if len(self.folders) == 1:
                    self.listbox.selection_set(0)
                    self.load_recording(self.folders[0])
                    
    def gui_scan_parent(self):
        parent_dir = filedialog.askdirectory(title="Select parent directory to scan")
        if parent_dir:
            parent_path = Path(parent_dir)
            found_folders = []
            for p in parent_path.rglob('*'):
                if p.is_dir():
                    # Look for folder signature of TDT block: contains .tbk or .tsq or .tev
                    tdt_files = list(p.glob('*.tbk')) + list(p.glob('*.tsq')) + list(p.glob('*.tev'))
                    if tdt_files:
                        found_folders.append(p)
            
            added_count = 0
            for f in sorted(found_folders):
                if f not in self.folders:
                    self.folders.append(f)
                    added_count += 1
                    
            if added_count > 0:
                self.update_listbox_status()
                if len(self.folders) == added_count:
                    self.listbox.selection_set(0)
                    self.load_recording(self.folders[0])
                messagebox.showinfo("Scan Completed", f"Successfully found and added {added_count} TDT block folders.")
            else:
                messagebox.showinfo("Scan Completed", "No new TDT block folders found.")
                
    def gui_clear_folders(self):
        if messagebox.askyesno("Clear Folders", "Are you sure you want to clear the list of recordings?"):
            self.folders = []
            self.current_folder = None
            self.t_disp = None
            self.sig_disp = None
            self.env_disp = None
            self.log_feat = None
            
            self.listbox.delete(0, tk.END)
            self.rec_title.config(text="No Recording Loaded")
            self.status_label.config(text="Select a recording to begin.", foreground="black")
            
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            self.ax3.text(0.5, 0.5, "Select Suppression and Burst epochs\nto generate power distributions", 
                          ha='center', va='center', transform=self.ax3.transAxes, color='gray')
            self.ax3.set_title("Epoch Power Distributions")
            self.ax3.set_axis_off()
            self.canvas.draw_idle()

    def update_listbox_status(self):
        self.updating_listbox = True
        try:
            central_thresholds = load_central_manual_thresholds()
            
            # Save active selection index
            current_idx = None
            if self.current_folder in self.folders:
                current_idx = self.folders.index(self.current_folder)
                
            self.listbox.delete(0, tk.END)
            for i, folder in enumerate(self.folders):
                rec_name = folder.name
                parent_name = folder.parent.name if folder.parent.name else "root"
                full_name = parent_name + "_" + rec_name
                
                # Check local sidecar JSON or central fallback JSON
                local_json = folder / "eeg_manual_threshold.json"
                is_annotated = local_json.exists() or (rec_name in central_thresholds) or (full_name in central_thresholds)
                
                prefix = "✓ " if is_annotated else "  "
                self.listbox.insert(tk.END, f"{prefix}{rec_name}")
                
                # Color code items
                if is_annotated:
                    self.listbox.itemconfig(i, fg="green")
                else:
                    self.listbox.itemconfig(i, fg="black")
                    
            # Restore selection
            if current_idx is not None:
                self.listbox.selection_set(current_idx)
                self.listbox.activate(current_idx)
        finally:
            self.updating_listbox = False
                
    def on_listbox_select(self, event):
        if getattr(self, 'updating_listbox', False):
            return
        selection = self.listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        self.load_recording(self.folders[idx])
        
    def load_recording(self, folder):
        self.current_folder = folder
        rec_name = folder.name
        parent_name = folder.parent.name if folder.parent.name else "root"
        full_name = parent_name + "_" + rec_name
        
        self.rec_title.config(text=f"Recording: {full_name}")
        self.status_label.config(text="Loading TDT data...", foreground="orange")
        self.root.update()
        
        try:
            # 1. Read TDT block
            data = read_tdt(str(folder))
            raw_sig_orig = data['rEEG']['data']
            fs_orig = data['rEEG']['fs']
            
            # 2. Resample to 1000Hz Master clock
            fs_target = 1000.0
            raw_sig = polyphase_resample(raw_sig_orig, fs_orig, fs_target)
            fs = fs_target
            
            # Cache the raw resampled signal
            self.sig_raw = raw_sig
            
            # 3. Bandpass filter (0.5 - 100.0 Hz)
            sos_filt = butter(4, (0.5, 100.0), 'bandpass', fs=fs, output='sos')
            self.sig_filt = sosfiltfilt(sos_filt, raw_sig)
            self.sig_filt = apply_notch_filter(self.sig_filt, fs, freq=50.0)
            self.sig_filt = apply_notch_filter(self.sig_filt, fs, freq=100.0)
            
            # Keep backup names to minimize other code changes
            self.sig_sw = self.sig_filt
            self.sig_bs = self.sig_filt
            
            # 5. Features & Envelope
            eeg_feats = calculate_eeg_features(self.sig_bs, fs)
            envelope = eeg_feats['power_env']
            self.log_feat = np.log10(envelope + 1e-12)
            
            # Decimate data for interactive plotting (~100k display points max)
            decimate_factor = max(1, len(self.sig_bs) // 100000)
            self.t_disp = np.arange(0, len(self.sig_bs), decimate_factor) / fs
            self.fs_disp = 1000.0 / decimate_factor
            self.env_disp = self.log_feat[::decimate_factor]
            
            # Initialize display signal based on active dropdown value
            self.filter_mode_var.set("0.5-100 Hz (Filtered)")
            self.sig_disp = self.sig_bs[::decimate_factor]
            
            # Reset Epochs & Thresholds
            self.supp_epochs = []
            self.burst_epochs = []
            self.addition_order = []
            self.thresh_val = None
            self.shading_patches = []
            self.epoch_patches = []
            self.current_win_size = 60.0
            
            self.supp_label.config(text="Suppression: None")
            self.burst_label.config(text="Burst: None")
            
            self.thresh_slider.config(state='disabled')
            self.thresh_entry.config(state='disabled')
            self.save_btn.config(state='disabled')
            
            # Setup plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # Re-connect xlim changed callback because clear() disconnects it!
            self.ax1.callbacks.connect('xlim_changed', self.on_xlim_changed)
            
            # Plot 1: Decimated EEG Filtered Signal
            self.ax1.plot(self.t_disp, self.sig_disp, color='#2c3e50', linewidth=0.5)
            self.ax1.set_ylabel("EEG Filtered\n(V)", rotation=90)
            self.ax1.set_title(f"EEG: {rec_name}", fontsize=11, fontweight='bold')
            self.ax1.grid(True, linestyle=':', alpha=0.5)
            
            # Plot 2: Log Power Envelope
            self.ax2.plot(self.t_disp, self.env_disp, color='#7f8c8d', linewidth=0.5)
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Log10 Power\nEnvelope", rotation=90)
            self.ax2.grid(True, linestyle=':', alpha=0.5)
            
            # Plot 3: Initial empty histogram
            self.ax3.text(0.5, 0.5, "Select Suppression and Burst epochs\nto generate power distributions", 
                          ha='center', va='center', transform=self.ax3.transAxes, color='gray')
            self.ax3.set_title("Epoch Power Distributions")
            self.ax3.set_axis_off()
            
            # Set scrollbar range to the duration of the recording
            t_max = float(self.t_disp[-1])
            self.time_slider.configure(from_=0.0, to=t_max)
            self.time_slider.set(t_max / 2.0)
            
            # Trigger window update
            self.win_size_menu.set("60s")
            self.on_win_size_change()
            
            # Perform initial Auto-fit Y-scale for the 60s view
            self.auto_y_scale()
            
            # Load sidecar manual threshold JSON from TDT folder if exists
            local_json = folder / "eeg_manual_threshold.json"
            saved_info = None
            if local_json.exists():
                try:
                    with open(local_json, 'r') as f:
                        saved_info = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not read local sidecar JSON: {e}")
            
            # Fall back to centralized JSON if not found locally
            if saved_info is None:
                central_thresholds = load_central_manual_thresholds()
                saved_info = central_thresholds.get(rec_name) or central_thresholds.get(full_name)
            
            if saved_info:
                if "supp_epochs" in saved_info:
                    self.supp_epochs = saved_info["supp_epochs"]
                elif "supp_epoch" in saved_info and saved_info["supp_epoch"] is not None:
                    self.supp_epochs = [saved_info["supp_epoch"]]
                else:
                    self.supp_epochs = []
                    
                if "burst_epochs" in saved_info:
                    self.burst_epochs = saved_info["burst_epochs"]
                elif "burst_epoch" in saved_info and saved_info["burst_epoch"] is not None:
                    self.burst_epochs = [saved_info["burst_epoch"]]
                else:
                    self.burst_epochs = []
                
                self.update_epoch_labels()
                self.update_epoch_spans()
                
                # Setup threshold and distributions
                self.check_epochs_and_calc_threshold()
                if saved_info.get("threshold_log") is not None:
                    self.set_threshold(saved_info["threshold_log"])
                
            self.status_label.config(text=f"Loaded: {rec_name}", foreground="green")
            
        except Exception as e:
            self.status_label.config(text=f"Error loading {rec_name}!", foreground="red")
            messagebox.showerror("Loading Error", f"Failed to load TDT data:\n{e}")
            import traceback
            traceback.print_exc()
            
    def on_filter_mode_change(self, event=None):
        if self.t_disp is None:
            return
            
        self.updating_from_event = True
        try:
            # Save X & Y limits before redraw
            xlim = self.ax1.get_xlim()
            
            mode = self.filter_mode_var.get()
            if "0.5-100" in mode:
                sig = self.sig_filt
                ylabel = "EEG Filtered 0.5-100Hz\n(V)"
            else:
                sig = self.sig_raw
                ylabel = "EEG Raw\n(V)"
                
            decimate_factor = max(1, len(sig) // 100000)
            self.sig_disp = sig[::decimate_factor]
            
            # Clear and replot Plot 1
            self.ax1.clear()
            # Re-connect xlim changed callback because clear() disconnects it!
            self.ax1.callbacks.connect('xlim_changed', self.on_xlim_changed)
            self.ax1.plot(self.t_disp, self.sig_disp, color='#2c3e50', linewidth=0.5)
            self.ax1.set_ylabel(ylabel, rotation=90)
            self.ax1.set_title(f"EEG: {self.current_folder.name}", fontsize=11, fontweight='bold')
            self.ax1.grid(True, linestyle=':', alpha=0.5)
            self.ax1.set_xlim(xlim)
            
            # Re-apply epoch shading overlays
            self.update_epoch_spans()
            # Auto-fit Y-scale to the new signal mode in this view
            self.auto_y_scale()
        finally:
            self.updating_from_event = False
        
    def get_window_size(self):
        val_str = self.win_size_var.get()
        if val_str == "Full":
            return float(self.t_disp[-1]) if self.t_disp is not None else 60.0
        return float(val_str[:-1])
        
    def on_win_size_change(self, *args):
        if self.t_disp is None:
            return
        self.current_win_size = self.get_window_size()
        center = self.time_slider.get()
        self.update_view_limits(center)
        
    def on_time_slider_change(self, val):
        if self.updating_from_event or self.t_disp is None:
            return
        self.update_view_limits(float(val))
        
    def update_view_limits(self, center):
        win_size = self.current_win_size
        half_win = win_size / 2.0
        
        xmin = center - half_win
        xmax = center + half_win
        
        t_max = self.t_disp[-1]
        
        if xmin < 0:
            xmin = 0.0
            xmax = min(t_max, win_size)
        elif xmax > t_max:
            xmax = t_max
            xmin = max(0.0, t_max - win_size)
            
        self.updating_from_event = True
        try:
            self.ax1.set_xlim(xmin, xmax)
            self.ax2.set_xlim(xmin, xmax)
            
            self.update_plots_for_threshold()
        finally:
            self.updating_from_event = False
            
    def on_xlim_changed(self, ax):
        if self.updating_from_event or self.t_disp is None:
            return
        self.updating_from_event = True
        try:
            xmin, xmax = ax.get_xlim()
            # Capture the new window size dynamically!
            self.current_win_size = xmax - xmin
            center = (xmin + xmax) / 2.0
            
            # Disable slider callback while setting value to avoid recursion/jitter
            self.time_slider.configure(command='')
            self.time_slider.set(center)
            self.time_slider.configure(command=self.on_time_slider_change)
            
            self.update_plots_for_threshold()
        finally:
            self.updating_from_event = False
            
    def scroll_left(self, event=None, factor=0.2):
        if self.t_disp is None:
            return
        xmin, xmax = self.ax1.get_xlim()
        win_size = xmax - xmin
        shift = win_size * factor
        new_center = ((xmin + xmax) / 2.0) - shift
        # Bound
        new_center = max(self.current_win_size / 2.0, new_center)
        self.time_slider.set(new_center)
        self.update_view_limits(new_center)
        
    def scroll_right(self, event=None, factor=0.2):
        if self.t_disp is None:
            return
        xmin, xmax = self.ax1.get_xlim()
        win_size = xmax - xmin
        shift = win_size * factor
        new_center = ((xmin + xmax) / 2.0) + shift
        # Bound
        t_max = self.t_disp[-1]
        new_center = min(t_max - self.current_win_size / 2.0, new_center)
        self.time_slider.set(new_center)
        self.update_view_limits(new_center)
        
    def zoom_x_in(self):
        if self.t_disp is None:
            return
        # Decrease visible window size by 30%
        self.current_win_size *= 0.7
        self.current_win_size = max(1.0, self.current_win_size)
        center = self.time_slider.get()
        self.update_view_limits(center)
        
    def zoom_x_out(self):
        if self.t_disp is None:
            return
        t_max = self.t_disp[-1]
        # Increase visible window size by 40%
        self.current_win_size *= 1.4
        self.current_win_size = min(float(t_max), self.current_win_size)
        center = self.time_slider.get()
        self.update_view_limits(center)
        
    def zoom_y_in(self):
        ymin, ymax = self.ax1.get_ylim()
        center = (ymin + ymax) / 2.0
        half_span = (ymax - ymin) / 2.0 * 0.75  # Zoom in 25%
        self.ax1.set_ylim(center - half_span, center + half_span)
        self.canvas.draw_idle()
        
    def zoom_y_out(self):
        ymin, ymax = self.ax1.get_ylim()
        center = (ymin + ymax) / 2.0
        half_span = (ymax - ymin) / 2.0 * 1.33  # Zoom out 33%
        self.ax1.set_ylim(center - half_span, center + half_span)
        self.canvas.draw_idle()
        
    def auto_y_scale(self):
        if self.t_disp is None:
            return
        xmin, xmax = self.ax1.get_xlim()
        visible_mask = (self.t_disp >= xmin) & (self.t_disp <= xmax)
        sig_visible = self.sig_disp[visible_mask]
        
        if len(sig_visible) > 0:
            ymin = np.min(sig_visible)
            ymax = np.max(sig_visible)
            # Add 10% vertical padding
            pad = (ymax - ymin) * 0.1
            if pad == 0:
                pad = 1e-6
            self.ax1.set_ylim(ymin - pad, ymax + pad)
            self.canvas.draw_idle()
            
    def delete_epochs(self):
        """Clears epoch selections, resets plots, and disables threshold settings."""
        self.supp_epochs = []
        self.burst_epochs = []
        self.addition_order = []
        self.thresh_val = None
        
        self.update_epoch_labels()
        
        # Disable threshold and save controls
        self.thresh_slider.config(state='disabled')
        self.thresh_entry.config(state='disabled')
        self.save_btn.config(state='disabled')
        
        # Clear overlays
        self.update_epoch_spans()
        
        # Reset histogram subplot (ax3)
        self.ax3.clear()
        self.ax3.text(0.5, 0.5, "Select Suppression and Burst epochs\nto generate power distributions", 
                      ha='center', va='center', transform=self.ax3.transAxes, color='gray')
        self.ax3.set_title("Epoch Power Distributions")
        self.ax3.set_axis_off()
        
        # Clear shading from Plot 1
        for patch in self.shading_patches:
            try:
                patch.remove()
            except Exception:
                pass
        self.shading_patches.clear()
        
        # Clear threshold line from Plot 2
        if hasattr(self, 'env_line') and self.env_line in self.ax2.lines:
            try:
                self.env_line.remove()
            except Exception:
                pass
            del self.env_line
            
        self.canvas.draw_idle()
        
        # Delete local sidecar file if exists
        if self.current_folder:
            local_json = self.current_folder / "eeg_manual_threshold.json"
            if local_json.exists():
                try:
                    os.remove(local_json)
                    print(f"Deleted local sidecar file: {local_json}")
                except Exception as e:
                    print(f"Error removing local sidecar file: {e}")
            
            # Remove from central database if exists
            rec_name = self.current_folder.name
            parent_name = self.current_folder.parent.name if self.current_folder.parent.name else "root"
            full_name = parent_name + "_" + rec_name
            central_json_path = Path(__file__).parent / "manual_thresholds.json"
            if central_json_path.exists():
                try:
                    with open(central_json_path, 'r') as f:
                        central_db = json.load(f)
                    
                    changed = False
                    if rec_name in central_db:
                        del central_db[rec_name]
                        changed = True
                    if full_name in central_db:
                        del central_db[full_name]
                        changed = True
                        
                    if changed:
                        with open(central_json_path, 'w') as f:
                            json.dump(central_db, f, indent=4)
                        print("Deleted entry from central database.")
                except Exception as e:
                    print(f"Warning: Could not update central database: {e}")
            
            # Always update listbox status
            self.update_listbox_status()
                    
    def add_suppression_epoch(self):
        if self.t_disp is None:
            return
        xmin, xmax = self.ax1.get_xlim()
        xmin = max(0.0, xmin)
        xmax = min(float(self.t_disp[-1]), xmax)
        
        if [xmin, xmax] not in self.supp_epochs:
            self.supp_epochs.append([xmin, xmax])
            self.addition_order.append('supp')
            self.update_epoch_spans()
            self.update_epoch_labels()
            self.check_epochs_and_calc_threshold()
            
    def add_burst_epoch(self):
        if self.t_disp is None:
            return
        xmin, xmax = self.ax1.get_xlim()
        xmin = max(0.0, xmin)
        xmax = min(float(self.t_disp[-1]), xmax)
        
        if [xmin, xmax] not in self.burst_epochs:
            self.burst_epochs.append([xmin, xmax])
            self.addition_order.append('burst')
            self.update_epoch_spans()
            self.update_epoch_labels()
            self.check_epochs_and_calc_threshold()
            
    def undo_last_epoch(self):
        if not self.addition_order:
            return
        last_type = self.addition_order.pop()
        if last_type == 'supp' and self.supp_epochs:
            self.supp_epochs.pop()
        elif last_type == 'burst' and self.burst_epochs:
            self.burst_epochs.pop()
            
        self.update_epoch_spans()
        self.update_epoch_labels()
        
        if self.supp_epochs and self.burst_epochs:
            self.check_epochs_and_calc_threshold()
        else:
            self.thresh_val = None
            self.thresh_slider.config(state='disabled')
            self.thresh_entry.config(state='disabled')
            self.save_btn.config(state='disabled')
            
            # Reset hist
            self.ax3.clear()
            self.ax3.text(0.5, 0.5, "Select Suppression and Burst epochs\nto generate power distributions", 
                          ha='center', va='center', transform=self.ax3.transAxes, color='gray')
            self.ax3.set_title("Epoch Power Distributions")
            self.ax3.set_axis_off()
            
            # Clear threshold lines
            if hasattr(self, 'env_line') and self.env_line in self.ax2.lines:
                try:
                    self.env_line.remove()
                except Exception:
                    pass
                del self.env_line
                
            self.canvas.draw_idle()
            
    def update_epoch_labels(self):
        # Update Suppression label
        if not self.supp_epochs:
            self.supp_label.config(text="Suppression: None")
        else:
            num = len(self.supp_epochs)
            text_items = [f"{e[0]:.1f}-{e[1]:.1f}s" for e in self.supp_epochs[-2:]]
            lbl_text = f"Suppression ({num}): " + ", ".join(text_items)
            if num > 2:
                lbl_text = "Suppression (" + str(num) + "): ... " + ", ".join(text_items)
            self.supp_label.config(text=lbl_text)
            
        # Update Burst label
        if not self.burst_epochs:
            self.burst_label.config(text="Burst: None")
        else:
            num = len(self.burst_epochs)
            text_items = [f"{e[0]:.1f}-{e[1]:.1f}s" for e in self.burst_epochs[-2:]]
            lbl_text = f"Burst ({num}): " + ", ".join(text_items)
            if num > 2:
                lbl_text = "Burst (" + str(num) + "): ... " + ", ".join(text_items)
            self.burst_label.config(text=lbl_text)
            
    def update_epoch_spans(self):
        old_updating = self.updating_from_event
        self.updating_from_event = True
        try:
            # SAVE CURRENT LIMITS to prevent any autoscale jumping!
            xlim = self.ax1.get_xlim()
            ylim1 = self.ax1.get_ylim()
            ylim2 = self.ax2.get_ylim()
            
            # Clear existing span markers
            for patch in self.epoch_patches:
                try:
                    patch.remove()
                except Exception:
                    pass
            self.epoch_patches.clear()
            
            # Redraw all suppression epochs
            for epoch in self.supp_epochs:
                p1 = self.ax1.axvspan(epoch[0], epoch[1], color='#e74c3c', alpha=0.25)
                p2 = self.ax2.axvspan(epoch[0], epoch[1], color='#e74c3c', alpha=0.25)
                self.epoch_patches.extend([p1, p2])
                
            # Redraw all burst epochs
            for epoch in self.burst_epochs:
                p3 = self.ax1.axvspan(epoch[0], epoch[1], color='#3498db', alpha=0.25)
                p4 = self.ax2.axvspan(epoch[0], epoch[1], color='#3498db', alpha=0.25)
                self.epoch_patches.extend([p3, p4])
                
            # FORCE MATPLOTLIB TO RESTORE ORIGINAL SCALES (Blocks Matplotlib autoscale overrides)
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim1)
            self.ax2.set_xlim(xlim)
            self.ax2.set_ylim(ylim2)
            
            self.canvas.draw_idle()
        finally:
            self.updating_from_event = old_updating
        
    def check_epochs_and_calc_threshold(self):
        if not self.supp_epochs or not self.burst_epochs:
            return
            
        fs = 1000.0
        vals_supp_list = []
        for epoch in self.supp_epochs:
            idx_start = int(epoch[0] * fs)
            idx_end = int(epoch[1] * fs)
            idx_start = max(0, min(idx_start, len(self.log_feat)-1))
            idx_end = max(0, min(idx_end, len(self.log_feat)))
            vals = self.log_feat[idx_start:idx_end]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                vals_supp_list.append(vals)
                
        vals_burst_list = []
        for epoch in self.burst_epochs:
            idx_start = int(epoch[0] * fs)
            idx_end = int(epoch[1] * fs)
            idx_start = max(0, min(idx_start, len(self.log_feat)-1))
            idx_end = max(0, min(idx_end, len(self.log_feat)))
            vals = self.log_feat[idx_start:idx_end]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                vals_burst_list.append(vals)
                
        if not vals_supp_list or not vals_burst_list:
            return
            
        vals_supp = np.concatenate(vals_supp_list)
        vals_burst = np.concatenate(vals_burst_list)
        
        mean_supp = np.mean(vals_supp)
        mean_burst = np.mean(vals_burst)
        
        # Default suggested threshold
        suggested_thresh = mean_supp + 0.55 * (mean_burst - mean_supp)
        
        # Configure slider and enable
        s_min = float(min(mean_supp - 0.5, mean_burst - 2.0))
        s_max = float(max(mean_burst + 0.5, mean_supp + 2.0))
        
        self.thresh_slider.config(state='normal', from_=s_min, to=s_max)
        self.thresh_entry.config(state='normal')
        self.save_btn.config(state='normal')
        
        # Update Histogram Subplot
        self.ax3.clear()
        self.ax3.set_axis_on()
        self.ax3.hist(vals_supp, bins=30, color='#e74c3c', alpha=0.6, label='Suppression')
        self.ax3.hist(vals_burst, bins=30, color='#3498db', alpha=0.6, label='Burst')
        self.ax3.set_xlabel("Log10(Power)")
        self.ax3.set_ylabel("Count", rotation=90)
        self.ax3.set_title("Power Distribution in Selected Epochs", fontsize=10, fontweight='bold')
        self.ax3.legend()
        self.ax3.grid(True, linestyle=':', alpha=0.5)
        
        self.hist_line = self.ax3.axvline(suggested_thresh, color='black', linestyle='--', linewidth=2)
        
        # Set the threshold value
        self.set_threshold(suggested_thresh)
        
    def set_threshold(self, val):
        self.thresh_val = float(val)
        old_updating = self.updating_from_event
        self.updating_from_event = True
        try:
            self.thresh_slider.set(val)
        finally:
            self.updating_from_event = old_updating
            
        self.thresh_entry.delete(0, tk.END)
        self.thresh_entry.insert(0, f"{val:.4f}")
        self.update_plots_for_threshold()
        
    def on_slider_change(self, val):
        if self.updating_from_event:
            return
        self.thresh_val = float(val)
        self.thresh_entry.delete(0, tk.END)
        self.thresh_entry.insert(0, f"{self.thresh_val:.4f}")
        self.update_plots_for_threshold()
        
    def on_entry_change(self, event=None):
        try:
            val = float(self.thresh_entry.get())
            s_min = float(self.thresh_slider.cget('from'))
            s_max = float(self.thresh_slider.cget('to'))
            if s_min <= val <= s_max:
                self.thresh_slider.set(val)
                self.thresh_val = val
                self.update_plots_for_threshold()
            else:
                messagebox.showwarning("Out of Range", f"Threshold must be between {s_min:.2f} and {s_max:.2f}")
        except ValueError:
            pass
            
    def update_plots_for_threshold(self):
        if self.thresh_val is None or self.t_disp is None:
            if self.t_disp is not None:
                self.canvas.draw_idle()
            return
            
        old_updating = self.updating_from_event
        self.updating_from_event = True
        try:
            # SAVE CURRENT LIMITS to prevent any autoscale jumping!
            xlim = self.ax1.get_xlim()
            ylim1 = self.ax1.get_ylim()
            ylim2 = self.ax2.get_ylim()
            
            # 1. Update horizontal line in Plot 2
            if hasattr(self, 'env_line') and self.env_line in self.ax2.lines:
                self.env_line.set_ydata([self.thresh_val, self.thresh_val])
            else:
                self.env_line = self.ax2.axhline(self.thresh_val, color='black', linestyle='--', linewidth=2)
                
            # 2. Update vertical line in Plot 3 (Histogram)
            if hasattr(self, 'hist_line') and self.hist_line in self.ax3.lines:
                self.hist_line.set_xdata([self.thresh_val, self.thresh_val])
                
            # 3. Update suppression shading in Plot 1
            for patch in self.shading_patches:
                try:
                    patch.remove()
                except Exception:
                    pass
            self.shading_patches.clear()
            
            xmin, xmax = xlim
            visible_mask = (self.t_disp >= xmin) & (self.t_disp <= xmax)
            t_visible = self.t_disp[visible_mask]
            env_visible = self.env_disp[visible_mask]
            
            if len(t_visible) > 0:
                # 1. Binary suppression based on threshold
                supp_visible = env_visible <= self.thresh_val
                
                # 2. Apply Neurophysiological heuristics (bridge too short supp, remove too short burst)
                # 1 = burst, 0 = suppression
                mask_visible = (~supp_visible).astype(float)
                try:
                    # Apply the heuristics using the display sampling frequency (fs_disp)
                    clean_mask_visible = apply_asymmetric_heuristics(
                        mask_visible, 
                        self.fs_disp, 
                        min_supp_sec=0.5, 
                        min_burst_sec=0.1
                    )
                    supp_visible = clean_mask_visible == 0.0
                except Exception as e:
                    print(f"Heuristics fallback: {e}")
                    pass
                
                patch = self.ax1.fill_between(t_visible, ylim1[0], ylim1[1], where=supp_visible, color='#e74c3c', alpha=0.15, step='mid')
                self.shading_patches.append(patch)
                
            # FORCE MATPLOTLIB TO RESTORE ORIGINAL SCALES (Blocks Matplotlib autoscale overrides)
            self.ax1.set_xlim(xlim)
            self.ax1.set_ylim(ylim1)
            self.ax2.set_xlim(xlim)
            self.ax2.set_ylim(ylim2)
            
            self.canvas.draw_idle()
        finally:
            self.updating_from_event = old_updating
        
    def save_threshold_to_json(self):
        if self.current_folder is None or self.thresh_val is None:
            return
            
        rec_name = self.current_folder.name
        
        # Save threshold and selection metadata directly into the TDT folder!
        local_json = self.current_folder / "eeg_manual_threshold.json"
        
        # For backward compatibility with any external tool reading a single range:
        s_epoch = self.supp_epochs[0] if self.supp_epochs else None
        b_epoch = self.burst_epochs[0] if self.burst_epochs else None
        
        saved_data = {
            "threshold_log": self.thresh_val,
            "supp_epochs": self.supp_epochs,
            "burst_epochs": self.burst_epochs,
            "supp_epoch": s_epoch,
            "burst_epoch": b_epoch,
            "saved_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(local_json, 'w') as f:
                json.dump(saved_data, f, indent=4)
            print(f"Saved local sidecar JSON: {local_json}")
            self.update_listbox_status()
            messagebox.showinfo("Threshold Saved", f"Threshold for '{rec_name}' saved directly inside its TDT folder!")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not write sidecar JSON:\n{e}")
            
    def start_pipeline_run(self):
        # Count remaining unannotated
        central_thresholds = load_central_manual_thresholds()
        pending = []
        for f in self.folders:
            rec_name = f.name
            parent_name = f.parent.name if f.parent.name else "root"
            full_name = parent_name + "_" + rec_name
            local_json = f / "eeg_manual_threshold.json"
            if not local_json.exists() and rec_name not in central_thresholds and full_name not in central_thresholds:
                pending.append(rec_name)
                
        if pending:
            msg = f"There are {len(pending)} recordings with no manual threshold selected.\n\nDo you want to close the annotator anyway?"
            if not messagebox.askyesno("Unannotated Recordings", msg):
                return
                
        self.root.destroy()
        
    def save_plot_image(self):
        if self.t_disp is None:
            return
        from tkinter import filedialog
        
        rec_name = self.current_folder.name if self.current_folder else "eeg_plot"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"), ("All Files", "*.*")],
            initialfile=f"{rec_name}_annotation.png",
            title="Save Plot Image"
        )
        if not file_path:
            return
            
        try:
            # Create a dedicated figure with a more square aspect ratio
            export_fig = Figure(figsize=(10, 8), dpi=300)
            
            # Re-create the subplots with generous vertical spacing
            ax1_exp = export_fig.add_subplot(3, 1, 1)
            ax2_exp = export_fig.add_subplot(3, 1, 2, sharex=ax1_exp)
            ax3_exp = export_fig.add_subplot(3, 1, 3)
            
            # Plot 1: EEG Signal
            ax1_exp.plot(self.t_disp, self.sig_disp, color='#2c3e50', linewidth=0.5)
            ax1_exp.set_ylabel(self.ax1.get_ylabel(), rotation=90)
            ax1_exp.set_title(self.ax1.get_title(), fontsize=12, fontweight='bold')
            ax1_exp.grid(True, linestyle=':', alpha=0.5)
            ax1_exp.set_xlim(self.ax1.get_xlim())
            ax1_exp.set_ylim(self.ax1.get_ylim())
            
            # Re-apply epoch spans to ax1_exp
            for epoch in self.supp_epochs:
                ax1_exp.axvspan(epoch[0], epoch[1], color='#e74c3c', alpha=0.25)
            for epoch in self.burst_epochs:
                ax1_exp.axvspan(epoch[0], epoch[1], color='#3498db', alpha=0.25)
                
            # Re-apply suppression shading to ax1_exp if threshold is set
            if self.thresh_val is not None:
                xmin, xmax = self.ax1.get_xlim()
                visible_mask = (self.t_disp >= xmin) & (self.t_disp <= xmax)
                t_visible = self.t_disp[visible_mask]
                env_visible = self.env_disp[visible_mask]
                if len(t_visible) > 0:
                    supp_visible = env_visible <= self.thresh_val
                    # Apply heuristics
                    mask_visible = (~supp_visible).astype(float)
                    try:
                        clean_mask_visible = apply_asymmetric_heuristics(
                            mask_visible, 
                            self.fs_disp, 
                            min_supp_sec=0.5, 
                            min_burst_sec=0.1
                        )
                        supp_visible = clean_mask_visible == 0.0
                    except Exception:
                        pass
                    ylim1 = self.ax1.get_ylim()
                    ax1_exp.fill_between(t_visible, ylim1[0], ylim1[1], where=supp_visible, color='#e74c3c', alpha=0.15, step='mid')
            
            # Plot 2: Log Power Envelope
            ax2_exp.plot(self.t_disp, self.env_disp, color='#7f8c8d', linewidth=0.5)
            ax2_exp.set_xlabel("Time (s)")
            ax2_exp.set_ylabel("Log10 Power\nEnvelope", rotation=90)
            ax2_exp.grid(True, linestyle=':', alpha=0.5)
            ax2_exp.set_xlim(self.ax2.get_xlim())
            ax2_exp.set_ylim(self.ax2.get_ylim())
            
            # Re-apply epoch spans to ax2_exp
            for epoch in self.supp_epochs:
                ax2_exp.axvspan(epoch[0], epoch[1], color='#e74c3c', alpha=0.25)
            for epoch in self.burst_epochs:
                ax2_exp.axvspan(epoch[0], epoch[1], color='#3498db', alpha=0.25)
                
            # Re-apply threshold horizontal line
            if self.thresh_val is not None:
                ax2_exp.axhline(self.thresh_val, color='black', linestyle='--', linewidth=2)
                
            # Plot 3: Histogram
            if self.supp_epochs and self.burst_epochs:
                fs = 1000.0
                vals_supp_list = []
                for epoch in self.supp_epochs:
                    idx_start = int(epoch[0] * fs)
                    idx_end = int(epoch[1] * fs)
                    idx_start = max(0, min(idx_start, len(self.log_feat)-1))
                    idx_end = max(0, min(idx_end, len(self.log_feat)))
                    vals = self.log_feat[idx_start:idx_end]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        vals_supp_list.append(vals)
                        
                vals_burst_list = []
                for epoch in self.burst_epochs:
                    idx_start = int(epoch[0] * fs)
                    idx_end = int(epoch[1] * fs)
                    idx_start = max(0, min(idx_start, len(self.log_feat)-1))
                    idx_end = max(0, min(idx_end, len(self.log_feat)))
                    vals = self.log_feat[idx_start:idx_end]
                    vals = vals[~np.isnan(vals)]
                    if len(vals) > 0:
                        vals_burst_list.append(vals)
                        
                if vals_supp_list and vals_burst_list:
                    vals_supp = np.concatenate(vals_supp_list)
                    vals_burst = np.concatenate(vals_burst_list)
                    ax3_exp.hist(vals_supp, bins=30, color='#e74c3c', alpha=0.6, label='Suppression')
                    ax3_exp.hist(vals_burst, bins=30, color='#3498db', alpha=0.6, label='Burst')
                    ax3_exp.set_xlabel("Log10(Power)")
                    ax3_exp.set_ylabel("Count", rotation=90)
                    ax3_exp.set_title("Power Distribution in Selected Epochs", fontsize=11, fontweight='bold')
                    ax3_exp.legend()
                    ax3_exp.grid(True, linestyle=':', alpha=0.5)
                    if self.thresh_val is not None:
                        ax3_exp.axvline(self.thresh_val, color='black', linestyle='--', linewidth=2)
            else:
                ax3_exp.text(0.5, 0.5, "Select Suppression and Burst epochs\nto generate power distributions", 
                              ha='center', va='center', transform=ax3_exp.transAxes, color='gray')
                ax3_exp.set_title("Epoch Power Distributions")
                ax3_exp.set_axis_off()
                
            export_fig.tight_layout()
            export_fig.subplots_adjust(hspace=0.55, top=0.93, bottom=0.1)
            
            export_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Export Successful", f"Plot image saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save plot image:\n{e}")
            import traceback
            traceback.print_exc()


def run_annotation_gui(valid_folders):
    """Launches the Tkinter App for the list of folders."""
    root = tk.Tk()
    root.title("EEG Standalone Threshold Annotation GUI")
    app = EEGThresholdApp(root, valid_folders)
    root.mainloop()


# Main entry point for standalone running
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone EEG Threshold Annotation GUI")
    parser.add_argument('folders', nargs='*', help='TDT block folders to load')
    args = parser.parse_args()
    
    # Filter valid folders from arguments if provided
    valid_folders = []
    for f in args.folders:
        p = Path(f)
        if p.exists() and p.is_dir():
            valid_folders.append(p)
        else:
            print(f"Warning: folder does not exist or is not a directory: {f}")
            
    run_annotation_gui(valid_folders)
