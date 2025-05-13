import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from utils.AESCryptoUtils import AESCryptoUtilsClass
from utils.file_utils import get_resource_path
import threading
from UI.LoadingWindow import LoadingWindow

class LogWindow:
    def __init__(self, parent_root):
        self.root = tk.Toplevel(parent_root)
        self.root.title("Encrypted Log Viewer")
        
        # Make window stay on top and grab focus
        self.root.attributes('-topmost', True)
        self.root.focus_force()
        self.root.grab_set()

        self.text_widget = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=100, height=40)
        self.text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 10))

        tk.Button(button_frame, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save to Disk", command=self.save_to_disk).pack(side=tk.LEFT, padx=5)

        self.load_log()

    def load_log(self):
        loading_window = LoadingWindow(
            parent=self.root,
            title="Loading Log",
            initial_text="Decrypting log file...",
            note_text="This may take a moment depending on the log size."
        )
        
        # Ensure loading window stays on top and grabs focus
        loading_window.popup.attributes('-topmost', True)
        loading_window.popup.focus_force()
        loading_window.popup.grab_set()

        def load_thread():
            try:
                file_path = get_resource_path("freescribe.log")
                with open(file_path, "r") as f:
                    encrypted_lines = f.readlines()

                decrypted_lines = []
                for enc_line in encrypted_lines:
                    enc_line = enc_line.strip()
                    if not enc_line:
                        continue
                    try:
                        decrypted_text = AESCryptoUtilsClass.decrypt(enc_line)
                        decrypted_lines.append(decrypted_text + "\n")
                    except Exception as line_err:
                        decrypted_lines.append(f"[Failed to decrypt line]: {line_err}\n")

                # Update UI in the main thread
                self.root.after(0, lambda: self._update_text_widget(decrypted_lines))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load or decrypt log:\n{str(e)}"))
            finally:
                self.root.after(0, loading_window.destroy)

        # Start the loading thread
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()

    def _update_text_widget(self, lines):
        # check if it exists
        if self.text_widget.winfo_exists():
            # Clear the text widget and insert the decrypted lines
            self.text_widget.delete(1.0, tk.END)
            for line in lines:
                self.text_widget.insert(tk.END, line)

    def copy_to_clipboard(self):
        content = self.text_widget.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(content)

    def save_to_disk(self):
        confirm = messagebox.askyesno(
            "Warning: Potential PHI Leak",
            "Saving decrypted logs to disk may leak Protected Health Information (PHI).\n\nAre you sure you want to proceed?"
        )
        if not confirm:
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                content = self.text_widget.get(1.0, tk.END).strip()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                messagebox.showinfo("Saved", f"File saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{str(e)}")