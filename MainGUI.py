## Update WSL to latest version to handle GUI interface natively
## Ref: https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps
import os
import subprocess
import threading
import signal
import tkinter as tk
from tkinter import font, ttk
from tkinter import Frame, Button, filedialog
from tkinter.messagebox import showerror

# Global variable
current_process_pid = None

# File types
data_filetypes = (("data files", "*.txt"), ("All files", "*.*"))

classifier_filetypes = (("h5 files", "*.h5"), ("All files", "*.*"))


def run_process(command, task_type):
    work_process = subprocess.Popen(command.split())

    # Save current process id to kill by button
    global current_process_pid
    current_process_pid = work_process.pid

    # Disable buttons while process is running
    if task_type == "correction":
        pred_start_button["state"] = "disabled"
        correction_start_button["state"] = "disabled"
        correction_stop_button["state"] = "normal"
    elif task_type == "predict":
        pred_start_button["state"] = "disabled"
        correction_start_button["state"] = "disabled"
        pred_stop_button["state"] = "normal"

    # Wait process to be ended
    work_process.wait()

    # Enable buttons
    pred_start_button["state"] = "normal"
    pred_stop_button["state"] = "disabled"
    correction_start_button["state"] = "normal"
    correction_stop_button["state"] = "disabled"


def run_task_thread(task_type):
    if task_type == "predict":
        pred_target_data_folder = pred_textbox_target_data_folder.get("1.0", "end-1c")
        pred_binary_model = pred_textbox_binary_classifier_model.get("1.0", "end-1c")
        pred_step_model = pred_textbox_setp_classifier_model.get("1.0", "end-1c")

        if len(pred_target_data_folder) == 0:
            showerror(title="error", message="Please set target data folder")
            return

        if len(pred_binary_model) == 0:
            showerror(title="error", message="Please set binary classifier")
            return

        if len(pred_binary_model) == 0:
            showerror(title="error", message="Please set step classifier")
            return

        command = (
            "python 1D_CNN_2L_Prediction.py -f "
            + pred_target_data_folder
            + " -mb "
            + pred_binary_model
            + " -ms "
            + pred_step_model
            + " -p -a"
        )

    elif task_type == "correction":
        correction_target_data_folder = correction_textbox_target_data_folder.get(
            "1.0", "end-1c"
        )

        if len(correction_target_data_folder) == 0:
            showerror(title="error", message="Please set input data folder")
            return

        command = "python GUI.py -i " + correction_target_data_folder

    elif task_type == "merge":
        merge_target_data_folder = merge_textbox_target_data_folder.get("1.0", "end-1c")

        if len(merge_target_data_folder) == 0:
            showerror(title="error", message="Please set input data folder")
            return

        command = "python ResultMerger.py -i " + merge_target_data_folder

    t1 = threading.Thread(target=lambda: run_process(command, task_type))
    t1.start()


# Kill process by pid
def kill_process(pid):
    print("Stopping process pid:", pid)
    os.kill(pid, signal.SIGTERM)


# Select file
def select_file(target_textbox, filetypes):
    filename = filedialog.askopenfilename(
        title="Open a file", initialdir=os.getcwd(), filetypes=filetypes
    )

    target_textbox.delete(1.0, "end")
    target_textbox.insert(tk.END, filename)


# Select folder
def select_folder(target_textbox):
    foldername = filedialog.askdirectory(
        title="Select data folder", initialdir=os.getcwd()
    )

    target_textbox.delete(1.0, "end")
    target_textbox.insert(tk.END, foldername)
    # target_textbox.insert(tk.END, foldername)


tk_root = tk.Tk()
main_frame = Frame(tk_root).grid(sticky="we")
tk_root.grid_columnconfigure(0, weight=1)

tk_root.geometry("800x800")
tk_root.title("ML Classification Toolkit")

def_font = font.nametofont("TkDefaultFont")
def_font.configure(size=24)

header_font_size = 20
label_font_size = 16

# Prediction stage
pred_label = tk.Label(main_frame, text="Prediction Stage", height=1, pady=5)
pred_label.config(font=("TkDefaultFont", header_font_size, "bold", "underline"))
pred_label.grid(row=0, column=0, sticky="we", columnspan=3)

pred_label_target_data_folder = tk.Label(
    main_frame, justify="right", text="Target data folder:"
)
pred_label_target_data_folder.config(font=("TkDefaultFont", label_font_size))
pred_label_target_data_folder.grid(row=1, column=0, pady=5, sticky="we")
pred_textbox_target_data_folder = tk.Text(main_frame, height=2, width=50)
pred_textbox_target_data_folder.grid(row=1, column=1, pady=5, sticky="we")
pred_button_target_data_folder = tk.Button(
    main_frame,
    text="Select folder",
    command=lambda: select_folder(pred_textbox_target_data_folder),
)
pred_button_target_data_folder.grid(row=1, column=2, pady=5)

pred_label_binary_classifier_model = tk.Label(
    main_frame, justify="right", text="Select binary classifier:"
)
pred_label_binary_classifier_model.config(font=("TkDefaultFont", label_font_size))
pred_label_binary_classifier_model.grid(row=2, column=0, pady=5)
pred_textbox_binary_classifier_model = tk.Text(main_frame, height=2, width=50)
pred_textbox_binary_classifier_model.grid(row=2, column=1, pady=5)
pred_button_binary_classifier_model = tk.Button(
    main_frame,
    text="Select file",
    command=lambda: select_file(
        pred_textbox_binary_classifier_model, classifier_filetypes
    ),
)
pred_button_binary_classifier_model.grid(row=2, column=2, pady=5)

pred_label_step_classifier_model = tk.Label(
    main_frame, justify="right", text="Select step classifier:"
)
pred_label_step_classifier_model.config(font=("TkDefaultFont", label_font_size))
pred_label_step_classifier_model.grid(row=3, column=0, pady=5)
pred_textbox_setp_classifier_model = tk.Text(main_frame, height=2, width=50)
pred_textbox_setp_classifier_model.grid(row=3, column=1, pady=5)
pred_button_setp_classifier_model = tk.Button(
    main_frame,
    text="Select file",
    command=lambda: select_file(
        pred_textbox_setp_classifier_model, classifier_filetypes
    ),
)
pred_button_setp_classifier_model.grid(row=3, column=2, pady=5)

pred_start_button = tk.Button(
    main_frame,
    text="Start prediction by ML",
    command=lambda: run_task_thread("predict"),
)
pred_start_button.grid(row=4, column=0, columnspan=3, sticky="WE")

pred_stop_button = tk.Button(
    main_frame,
    text="Stop prediction by ML",
    command=lambda: kill_process(current_process_pid),
)
pred_stop_button["state"] = "disabled"
pred_stop_button.grid(row=5, column=0, columnspan=3, sticky="WE")

ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=6, columnspan=3, sticky="ew")

# Correction stage
correction_label = tk.Label(main_frame, text="Correction Stage", height=1, pady=5)
correction_label.config(font=("TkDefaultFont", header_font_size, "bold", "underline"))
correction_label.grid(row=7, column=0, sticky="we", columnspan=3)

correction_label_target_data_folder = tk.Label(
    main_frame, justify="right", text="Target data folder:"
)
correction_label_target_data_folder.config(font=("TkDefaultFont", label_font_size))
correction_label_target_data_folder.grid(row=8, column=0, pady=5, sticky="we")
correction_textbox_target_data_folder = tk.Text(main_frame, height=2, width=50)
correction_textbox_target_data_folder.grid(row=8, column=1, pady=5, sticky="we")
correction_button_target_data_folder = tk.Button(
    main_frame,
    text="Select folder",
    command=lambda: select_folder(correction_textbox_target_data_folder),
)
correction_button_target_data_folder.grid(row=8, column=2, pady=5)

correction_start_button = tk.Button(
    main_frame, text="Run correction GUI", command=lambda: run_task_thread("correction")
)
correction_start_button.grid(row=9, column=0, columnspan=3, sticky="WE")

correction_stop_button = tk.Button(
    main_frame,
    text="Stop correction GUI",
    command=lambda: kill_process(current_process_pid),
)
correction_stop_button["state"] = "disabled"
correction_stop_button.grid(row=10, column=0, columnspan=3, sticky="WE")

ttk.Separator(main_frame, orient=tk.HORIZONTAL).grid(row=11, columnspan=3, sticky="ew")

# Merging stage
merge_label = tk.Label(main_frame, text="Mering Stage", height=1, pady=5)
merge_label.config(font=("TkDefaultFont", header_font_size, "bold", "underline"))
merge_label.grid(row=12, column=0, sticky="we", columnspan=3)

merge_label_target_data_folder = tk.Label(
    main_frame, justify="right", text="Target data folder:"
)
merge_label_target_data_folder.config(font=("TkDefaultFont", label_font_size))
merge_label_target_data_folder.grid(row=13, column=0, pady=5, sticky="we")
merge_textbox_target_data_folder = tk.Text(main_frame, height=2, width=50)
merge_textbox_target_data_folder.grid(row=13, column=1, pady=5, sticky="we")
merge_button_target_data_folder = tk.Button(
    main_frame,
    text="Select folder",
    command=lambda: select_folder(merge_textbox_target_data_folder),
)
merge_button_target_data_folder.grid(row=13, column=2, pady=5)

merge_start_button = tk.Button(
    main_frame, text="Merge results", command=lambda: run_task_thread("merge")
)
merge_start_button.grid(row=14, column=0, columnspan=3, sticky="WE")

merge_stop_button = tk.Button(
    main_frame,
    text="Stop merging process",
    command=lambda: kill_process(current_process_pid),
)
merge_stop_button["state"] = "disabled"
merge_stop_button.grid(row=15, column=0, columnspan=3, sticky="WE")

tk_root.mainloop()
