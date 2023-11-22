import tkinter as tk
import webbrowser
from urllib.parse import quote
import time

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
	pass

# Create the main window
root = tk.Tk()
root.title("TAU | National Library of Finland Recommendation System")
root.geometry('1200x450') 

label = tk.Label(root, text="Welcome\n\nWhat are you looking after, today?")
label.pack(pady=25)

# # Create and pack widgets Set default text with gray color
entry = tk.Entry(root, width=55, fg='grey')
entry.insert(0, "Query keywords...")
entry.pack(pady=20)

# Bind events to the Entry widget
entry.bind('<FocusIn>', on_entry_click)
entry.bind('<FocusOut>', on_entry_leave)

search_button = tk.Button(root, text="Search NLF", padx=50, command=generate_link)
search_button.pack(pady=2)

nlf_link_lable = tk.Label(root, text="", fg='blue', cursor='arrow')
nlf_link_lable.pack(pady=1)

recys_label = tk.Label(root, text="")
recys_label.pack(pady=20)

search_button = tk.Button(root, text="Recommend Me!", padx=75, command=recSys_cb)
search_button.pack(pady=20)

close_button = tk.Button(root, text="Exit", command=lambda: close_window())
close_button.pack(pady=20)

countdown_label = tk.Label(root, text="")
countdown_label.pack(pady=10)

root.mainloop()