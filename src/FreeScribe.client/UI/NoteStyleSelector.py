import tkinter as tk
import UI.Helpers
from tkinter import ttk, simpledialog, messagebox
from dataclasses import dataclass
import json
import os


@dataclass
class StylePromptInfo:
    style_name: str
    pre_prompt: str
    post_prompt: str

class NoteStyleSelector(tk.Frame):
    # Static class variables
    current_style = "SOAP Note - Default"
    style_data = {}  # Store style data with pre/post prompts
    style_options = ["Add Prompt Template...", "SOAP Note - Default"]
    _styles_file = "note_styles.json"
    
    @classmethod
    def _get_default_styles(cls):
        """Return the default styles configuration"""
        return {
            "SOAP Note - Default": {
                "pre_prompt": "AI, please transform the following conversation into a concise SOAP note. Do not assume any medical data, vital signs, or lab values. Base the note strictly on the information provided in the conversation. Ensure that the SOAP note is structured appropriately with Subjective, Objective, Assessment, and Plan sections. Strictly extract facts from the conversation. Here's the conversation:",
                "post_prompt": "Remember, the Subjective section should reflect the patient's perspective and complaints as mentioned in the conversation. The Objective section should only include observable or measurable data from the conversation. The Assessment should be a summary of your understanding and potential diagnoses, considering the conversation's content. The Plan should outline the proposed management, strictly based on the dialogue provided. Do not add any information that did not occur and do not make assumptions. Strictly extract facts from the conversation."
            }
        }
    
    @classmethod
    def load_styles(cls):
        """Load styles from disk, create defaults if file doesn't exist"""
        try:
            if os.path.exists(cls._styles_file):
                with open(cls._styles_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cls.style_data = data.get('styles', {})
                    cls.current_style = data.get('current_style', "SOAP Note - Default")
            else:
                # Create default styles
                cls.style_data = cls._get_default_styles()
                cls.current_style = "SOAP Note - Default"
                cls.save_styles()
            
            # Update style_options list - only add custom styles that aren't already in the list
            custom_styles = [style for style in cls.style_data.keys() if style not in cls.style_options]
            cls.style_options = ["Add Prompt Template...", "SOAP Note - Default"] + custom_styles

        except Exception as e:
            print(f"Error loading styles: {e}")
            # Fallback to defaults
            cls.style_data = cls._get_default_styles()
            cls.current_style = "SOAP Note - Default"
            # Only add custom styles that aren't already in the list
            custom_styles = [style for style in cls.style_data.keys() if style not in cls.style_options]
            cls.style_options = ["Add Prompt Template...", "SOAP Note - Default"] + custom_styles

    @classmethod
    def save_styles(cls):
        """Save styles to disk"""
        try:
            data = {
                'styles': cls.style_data,
                'current_style': cls.current_style
            }
            with open(cls._styles_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving styles: {e}")
    
    def __init__(self, root=None, parent_frame=None):
        # Load styles before initializing
        if not hasattr(NoteStyleSelector, '_styles_loaded'):
            NoteStyleSelector.load_styles()
            NoteStyleSelector._styles_loaded = True
            
        super().__init__(parent_frame, bg=parent_frame.cget("bg"))
        self.root = root
        self.parent_frame = parent_frame
        self.create_widgets()

    def create_widgets(self):
        # Frame to hold dropdown and buttons horizontally
        self.style_var = tk.StringVar(value=NoteStyleSelector.current_style)

        self.style_combo = ttk.Combobox(self, textvariable=self.style_var, 
                                       values=NoteStyleSelector.style_options, state="readonly", width=35)
        self.style_combo.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        self.style_combo.bind("<<ComboboxSelected>>", self.on_style_change)

        self.edit_button = tk.Button(self, text="Edit", command=self.edit_style)
        self.edit_button.pack(side=tk.LEFT, padx=2)

        self.delete_button = tk.Button(self, text="Delete", command=self.delete_style)
        self.delete_button.pack(side=tk.LEFT, padx=2)

    @staticmethod
    def get_current_prompt_info() -> StylePromptInfo:
        """Static method to get current style's prompt information"""
        current_style = NoteStyleSelector.current_style
        if current_style in NoteStyleSelector.style_data:
            return StylePromptInfo(
                style_name=current_style,
                pre_prompt=NoteStyleSelector.style_data[current_style]['pre_prompt'],
                post_prompt=NoteStyleSelector.style_data[current_style]['post_prompt']
            )
        else:
            return StylePromptInfo(
                style_name=current_style,
                pre_prompt='',
                post_prompt=''
            )

    def on_style_change(self, event=None):
        selected_value = self.style_var.get()
        if selected_value == "Add Prompt Template...":
            self.add_style()
            # Reset to previous valid selection
            if len(NoteStyleSelector.style_options) > 1:
                self.style_var.set(NoteStyleSelector.style_options[1])  # Set to first non-"Add Prompt Template" option
        else:
            NoteStyleSelector.current_style = selected_value
            NoteStyleSelector.save_styles()

    def add_style(self):
        # Create a custom dialog for style creation
        dialog = StyleDialog(self.root, "Add Prompt Template")
        if dialog.result:
            style_name, pre_prompt, post_prompt = dialog.result
            if style_name and style_name not in NoteStyleSelector.style_options:
                NoteStyleSelector.style_options.append(style_name)
                NoteStyleSelector.style_data[style_name] = {
                    'pre_prompt': pre_prompt,
                    'post_prompt': post_prompt
                }
                self.update_dropdown()
                self.style_var.set(style_name)
                NoteStyleSelector.current_style = style_name
                NoteStyleSelector.save_styles()

    def edit_style(self):
        current_style = self.style_var.get()
        if current_style == "Add Prompt Template...":
            messagebox.showwarning("Edit Style", "Cannot edit 'Add Prompt Template...' option.")
            return
        
        # Get existing data if available
        existing_data = NoteStyleSelector.style_data.get(current_style, {'pre_prompt': '', 'post_prompt': ''})
        
        # Check if it's the default style - allow viewing but not editing
        is_default = (current_style == "SOAP Note - Default")
        
        dialog = StyleDialog(self.root, "View Style" if is_default else "Edit Style", 
                           initial_name=current_style,
                           initial_pre=existing_data['pre_prompt'],
                           initial_post=existing_data['post_prompt'],
                           read_only=is_default)
        
        if dialog.result and not is_default:
            new_name, pre_prompt, post_prompt = dialog.result
            if new_name and new_name != current_style and new_name not in NoteStyleSelector.style_options:
                # Replace the old style with the new one
                index = NoteStyleSelector.style_options.index(current_style)
                NoteStyleSelector.style_options[index] = new_name
                
                # Remove old data and add new
                if current_style in NoteStyleSelector.style_data:
                    del NoteStyleSelector.style_data[current_style]
                NoteStyleSelector.style_data[new_name] = {
                    'pre_prompt': pre_prompt,
                    'post_prompt': post_prompt
                }
                
                self.update_dropdown()
                self.style_var.set(new_name)
                NoteStyleSelector.current_style = new_name
                NoteStyleSelector.save_styles()
            elif new_name == current_style:
                # Just update the prompts
                NoteStyleSelector.style_data[current_style] = {
                    'pre_prompt': pre_prompt,
                    'post_prompt': post_prompt
                }
                NoteStyleSelector.save_styles()

    def delete_style(self):
        current_style = self.style_var.get()
        if current_style in ["Add Prompt Template...", "SOAP Note - Default"]:
            messagebox.showwarning("Delete Style", "Cannot delete 'Add Prompt Template...' or 'SOAP Note - Default' style.")
            return
        
        if messagebox.askyesno("Delete Style", f"Are you sure you want to delete '{current_style}'?"):
            NoteStyleSelector.style_options.remove(current_style)
            if current_style in NoteStyleSelector.style_data:
                del NoteStyleSelector.style_data[current_style]
            self.update_dropdown()
            # Set to default after deletion
            self.style_var.set("SOAP Note - Default")
            NoteStyleSelector.current_style = "SOAP Note - Default"
            NoteStyleSelector.save_styles()

    def update_dropdown(self):
        # Update the Combobox values
        self.style_combo['values'] = NoteStyleSelector.style_options

    def apply_style(self):
        selected_style = self.style_var.get()
        if selected_style != "Add Prompt Template...":
            prompt_info = NoteStyleSelector.get_current_prompt_info()
            print(f"Selected Note Style: {prompt_info.style_name}")
            print(f"Pre Prompt: {prompt_info.pre_prompt}")
            print(f"Post Prompt: {prompt_info.post_prompt}")

class StyleDialog:
    def __init__(self, parent, title, initial_name="", initial_pre="", initial_post="", read_only=False):
        self.result = None
        self.read_only = read_only
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("900x650")  # Slightly taller to accommodate warning
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.focus_force()
        self.dialog.lift()
        
        # Make it stay on top
        self.dialog.attributes('-topmost', True)
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        # Add warning for read-only mode
        if self.read_only:
            warning_frame = tk.Frame(self.dialog, bg='#ffeeee', relief='raised', bd=2)
            warning_frame.pack(fill=tk.X, padx=10, pady=5)
            warning_label = tk.Label(warning_frame, 
                                   text="⚠️ Cannot edit default note style - View only mode",
                                   bg='#ffeeee', fg='#cc0000', font=('Arial', 10, 'bold'))
            warning_label.pack(pady=5)
        
        # Style name
        tk.Label(self.dialog, text="Template Name:").pack(pady=5)
        self.name_entry = tk.Entry(self.dialog, width=50, font=('Arial', 10))
        self.name_entry.pack(pady=5)
        self.name_entry.insert(0, initial_name)
        if self.read_only:
            self.name_entry.configure(state='disabled')
        
        # Pre prompt section with explanation
        tk.Label(self.dialog, text="Pre Prompt:").pack(pady=(15, 5))
        
        pre_frame = tk.Frame(self.dialog)
        pre_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        # Pre prompt text box
        self.pre_text = tk.Text(pre_frame, height=8, width=50, font=('Arial', 9))
        self.pre_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.pre_text.insert(tk.END, initial_pre)
        if self.read_only:
            self.pre_text.configure(state='disabled')
        
        # Pre prompt explanation
        pre_explanation = (
            "This is the FIRST part of the AI prompt structure:\n\n"
            "• Acts as the opening instruction to the AI\n"
            "• Sets up how to interpret the conversation\n"
            "• Defines SOAP note format requirements\n"
            "• Conversation will be inserted after this\n\n"
            "⚠️ Modify with caution as it affects AI output quality"
        )
        
        pre_explanation_label = tk.Label(pre_frame, text=pre_explanation, 
                                       justify=tk.LEFT, anchor='nw', 
                                       font=('Arial', 8), fg='gray',
                                       wraplength=215, width=35)
        pre_explanation_label.pack(side=tk.RIGHT, padx=(10, 0), fill=tk.Y)
        
        # Post prompt section with explanation
        tk.Label(self.dialog, text="Post Prompt:").pack(pady=(15, 5))
        
        post_frame = tk.Frame(self.dialog)
        post_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        # Post prompt text box
        self.post_text = tk.Text(post_frame, height=8, width=50, font=('Arial', 9))
        self.post_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.post_text.insert(tk.END, initial_post)
        if self.read_only:
            self.post_text.configure(state='disabled')
        
        # Post prompt explanation
        post_explanation = (
            "This is the LAST part of the AI prompt structure:\n\n"
            "• Added after the conversation text\n"
            "• Provides final formatting instructions\n"
            "• Ensures SOAP note completeness\n"
            "• Helps maintain consistency\n\n"
            "⚠️ Modify with caution as it affects AI output quality"
        )
        
        post_explanation_label = tk.Label(post_frame, text=post_explanation, 
                                        justify=tk.LEFT, anchor='nw', 
                                        font=('Arial', 8), fg='gray',
                                        wraplength=215, width=35)
        post_explanation_label.pack(side=tk.RIGHT, padx=(10, 0), fill=tk.Y)
        
        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20, fill=tk.X)
        
        # Center the buttons
        inner_frame = tk.Frame(button_frame)
        inner_frame.pack()
        
        # Only show OK button if not read-only, or disable it if read-only
        if self.read_only:
            ok_button = tk.Button(inner_frame, text="OK", command=self.ok_clicked, 
                                width=10, font=('Arial', 10), state='disabled')
            ok_button.pack(side=tk.LEFT, padx=10)
            tk.Button(inner_frame, text="Close", command=self.cancel_clicked, 
                     width=10, font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        else:
            tk.Button(inner_frame, text="OK", command=self.ok_clicked, 
                     width=10, font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
            tk.Button(inner_frame, text="Cancel", command=self.cancel_clicked, 
                     width=10, font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        
        # Focus on name entry (if not disabled)
        if not self.read_only:
            self.name_entry.focus_set()
            self.name_entry.select_range(0, tk.END)
        
        UI.Helpers.center_window_to_parent(self.dialog, parent)
        UI.Helpers.set_window_icon(self.dialog)

        # Wait for dialog to close
        self.dialog.wait_window()
    
    def ok_clicked(self):
        if self.read_only:
            return  # Shouldn't be called in read-only mode, but just in case
            
        name = self.name_entry.get().strip()
        pre_prompt = self.pre_text.get("1.0", tk.END).strip()
        post_prompt = self.post_text.get("1.0", tk.END).strip()
        
        if name:
            self.result = (name, pre_prompt, post_prompt)
        
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.dialog.destroy()

if __name__ == "__main__":
    # build a sample tkinter window to test the NoteStyleSelector

    root = tk.Tk()
    root.title("Note Style Selector Test")

    note_style_selector = NoteStyleSelector(root)
    note_style_selector.pack()

    root.mainloop()
