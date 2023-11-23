import tkinter as tk
from tkinter import messagebox 
from urllib.parse import quote
from PIL import Image, ImageTk
import webbrowser
import time
import requests
from io import BytesIO

def on_entry_click(event):
	query = entry.get()
	if query == "Query keywords...":
		entry.delete(0, tk.END)
		entry.config(fg='black')  # Change text color to black

def on_entry_leave(event):
	query = entry.get()
	if query == "":
		entry.insert(0, "Query keywords...")
		entry.config(fg='grey')  # Change text color to gray

def generate_link():
	query = entry.get()
	if query and query != "Query keywords...":
		encoded_query = quote(query)
		base_url = "https://digi.kansalliskirjasto.fi/search"
		link = f"{base_url}?query={encoded_query}"
		nlf_link_lable.config(text=f"NLF suggested link: {link}", fg='blue', cursor='hand2')
		nlf_link_lable.bind("<Button-1>", lambda e: webbrowser.open(link))
	else:
		nlf_link_lable.config(text="Enter a valid search query to proceed!", fg='red', )
		messagebox.showerror('Error', 'Enter a valid search query to proceed!')

def close_window(count=8):
	if count > 0:
		countdown_label.config(text=f"Thanks for using our service, Have a Good Day!\n\nclosing in {count} sec...")
		root.after(1000, close_window, count-1)
	else:
		root.destroy()

def recSys_cb():
	query = entry.get()
	if query and query != "Query keywords...":
		recys_label.config(text=f"Since You searched < {query} >\nYou might be interested in: TK1, TK2, TK3, TK4", fg='green')
	else:
		nlf_link_lable.config(text="Enter a valid search query first", fg='red', )

def trending_today_cb():
	trn_lemmas="Suomi | Helsinki | "
	trn_lbl=tk.Label(root, text=trn_lemmas )
	trn_lbl.pack()

# Create the main window
root = tk.Tk()
root.title("TAU | National Library of Finland Recommendation System")
root.geometry('1000x550')

icon_img = ImageTk.PhotoImage(Image.open("search-13-16.png").resize((10, 10),Image.Resampling.LANCZOS))
root.iconphoto(True, icon_img)

# Load left and right images from your local directory
left_image_path = "tau.jpg"
right_image_path = "nlf.png"

left_image = Image.open(left_image_path)
right_image = Image.open(right_image_path)

left_image = left_image.resize((150, 150), Image.Resampling.LANCZOS)
right_image = right_image.resize((150, 150), Image.Resampling.LANCZOS)

tk_left_image = ImageTk.PhotoImage(left_image)
tk_right_image = ImageTk.PhotoImage(right_image)

# Set the images on labels
left_image_label = tk.Label(root, image=tk_left_image)
right_image_label = tk.Label(root, image=tk_right_image)

# left_image_label.pack(side="top", anchor="w", pady=5)
# right_image_label.pack(side="top", anchor="e", pady=5)


# Position the images using grid
left_image_label.grid(row=0, column=0, padx=5, pady=15)
right_image_label.grid(row=0, column=1, padx=5, pady=15)


# Add a label right after the images
welcome_label = tk.Label(root, text="Welcome!\n\nWhat are you looking after, today?")

# # Create and pack widgets Set default text with gray color
entry = tk.Entry(root, width=55, fg='grey', borderwidth=5)
entry.insert(0, "Query keywords...")

# Bind events to the Entry widget
entry.bind('<FocusIn>', on_entry_click)
entry.bind('<FocusOut>', on_entry_leave)

search_btn = tk.Button(root, text="Search NLF", command=generate_link)
nlf_link_lable = tk.Label(root, text="", fg='blue', cursor='arrow')


# trending_btn = tk.Button(root, text="Popular Today!", command=trending_today_cb)
# trending_btn.pack()

rec_btn = tk.Button(root, text="Recommend Me!", command=recSys_cb)
recys_label = tk.Label(root, text="")


exit_btn = tk.Button(root, text="Exit", command=lambda: close_window())
countdown_label = tk.Label(root, text="")

welcome_label.grid(row=1, column=0, columnspan=2, pady=5)
entry.grid(row=2, column=0, columnspan=2, pady=10)

search_btn.grid(row=3, column=0, columnspan=2, pady=5)
nlf_link_lable.grid(row=4, column=0, columnspan=2, pady=5)

rec_btn.grid(row=5, column=0, columnspan=2, pady=5)
recys_label.grid(row=6, column=0, columnspan=2, pady=5)

exit_btn.grid(row=7, column=0, columnspan=2, pady=5)
countdown_label.grid(row=8, column=0, columnspan=2, pady=10)

root.mainloop()