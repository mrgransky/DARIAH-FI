from utils import *
import ipywidgets as widgets
from IPython.display import display, HTML, Image
from PIL import Image as PILImage, ImageOps
from io import BytesIO

digi_base_url = "https://digi.kansalliskirjasto.fi/search"
left_image_path = "https://www.topuniversities.com/sites/default/files/profiles/logos/tampere-university_5bbf14847d023f5bc849ec9a_large.jpg"
right_image_path = "https://digi.kansalliskirjasto.fi/images/logos/logo_fi_darkblue.png"
TKs=list()
flinks=list()
left_image = PILImage.open(BytesIO(requests.get(left_image_path).content))
right_image = PILImage.open(BytesIO(requests.get(right_image_path).content))
left_image_widget = widgets.Image(value=requests.get(left_image_path).content, format='png', width=300, height=300)
right_image_widget = widgets.Image(value=requests.get(right_image_path).content, format='png', width=300, height=300)
welcome_lbl = widgets.HTML(value="<h2 style=font-family:verdana;font-size:30px;color:black;text-align:center;>Welcome!<br>What are you looking after, today?</h2>")

# Modified entry widget
entry = widgets.Text(placeholder="Enter your query keywords here...", 
										 layout=widgets.Layout(width='800px', 
																					 height='50px',
																					 font_size='30px', 
																					 padding='5px',
																					 font_weight='bold',
																					 font_family='Ubuntu',
																					)
										)

# Added vertical padding
vbox_layout = widgets.Layout(align_items='center', padding='15px')

# Modified buttons with dark gray background and enlarged font size
button_style = {'button_color': 'darkgray', 'font_weight': 'bold', 'font_size': '16px'}

search_btn = widgets.Button(description="Search NLF", layout=widgets.Layout(width='150px'), style=button_style)
clean_search_btn = widgets.Button(description="Clear", layout=widgets.Layout(width='150px'), style=button_style)
rec_btn = widgets.Button(description="Recommend Me", layout=widgets.Layout(width='150px'), style=button_style)
clean_recsys_btn = widgets.Button(description="Clear", layout=widgets.Layout(width='150px'), style=button_style)
exit_btn = widgets.Button(description="Exit", layout=widgets.Layout(width='100px'), style=button_style)

countdown_lbl = widgets.HTML()
recys_lbl = widgets.HTML()
nlf_link_lable = widgets.HTML()

# Modified slider to have a minimum value of 3 and a maximum value of 15
slider_style={'description_width': 'initial'}
slider_value = widgets.IntSlider(value=5, min=3, max=20, description='Recsys Count', style=slider_style)
slider_value.layout.visibility = 'hidden'  # Initially hidden

progress_bar_style = {'description_width': 'initial', 'bar_color': 'blue', 'background_color': 'darkgray'}
progress_bar_description_style = {'description_width': 'initial', 'font-size': '25px', 'fort_family': 'Futura'}
progress_bar = widgets.IntProgress(value=0, min=0, max=350, description='Please wait...', style=progress_bar_style)
progress_bar.description_style = progress_bar_description_style
progress_bar.layout.visibility = 'hidden'  # Initially hidden

def close_window(count=8):
	if count > 0:
		countdown_lbl.value = f"Thanks for using our service, Have a Good Day!<br><br>closing in {count} sec..."
		time.sleep(1)
		close_window(count-1)
	else:
		display(HTML("<b>Bye</b>"))

def get_nlf_link(change):
	query = entry.value
	if query and query != "Enter your query keywords here...":
		encoded_query = urllib.parse.quote(query)
		gen_link=f"{digi_base_url}?query={encoded_query}"
		nlf_link_lable.value=f"<b style=font-family:verdana;font-size:20px;color:blue><a href={gen_link} target='_blank'>Click here to open National Library Results</a></b>"
	else:
		nlf_link_lable.value = "<p style=font-family:Courier;font-size:18px;color:red>Oops! Enter a valid search query to proceed!</p>"

def on_entry_click(widget, event, data):
	if widget.value == "Enter your query keywords here...":
		widget.value = ""
		widget.style = {'description_width': 'initial', 'color': 'black'}

def clean_search_entry(change):
	nlf_link_lable.value = ""
	entry.value = ""
	entry.placeholder = "Enter your query keywords here..."

def get_recsys_result(qu: str="Tampereen seudun työväenopisto"):
	print(f"Running {__file__} using {nb.get_num_threads()} CPU core(s) query: {qu}")
	cmd=[	'python', 'concat_dfs.py', 
				'--dfsPath', '/scratch/project_2004072/Nationalbiblioteket/dataframes_x30',
				'--lmMethod', 'stanza',
				'--qphrase', f'{qu}',
			]
	process=subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	
	print(f"Wait for the script to finish executing (time consuming)...")
	t0=time.time()
	return_code=process.wait() # Wait for the script to finish executing
	print(f"elapsed_t: {time.time()-t0:.5f} sec")

	print(f"process.communicate()", end="\t")
	t0=time.time()
	stdout, stderr=process.communicate() # stderr not important!
	print(f"elapsed_t: {time.time()-t0:.5f} sec")

	if return_code != 0: # Check the return code for errors
		print(f"<!> Error in executing script: {stderr}")
		return

	print("*"*120)
	print(stdout)
	print("*"*120)

	serialized_result=re.search(r'Serialized Result: (.+)', stdout).group(1)
	recommended_tokens=json.loads(serialized_result)
	print('Captured Result:', type(recommended_tokens), len(recommended_tokens), recommended_tokens)
	# return [f"TK_{i+1}" for i in range(topK)]
	return recommended_tokens

def update_recys_lbl(_):
	query = entry.value
	if query and query != "Enter your query keywords here...":
		recys_lbl.value = generate_recys_html(query, TKs, flinks, slider_value.value)
	else:
		recys_lbl.value = "<font color='red'>Enter a valid search query first</font>"

def generate_recys_html(query, TKs, flinks, slider_value):
	recys_lines = ""
	for i in np.arange(slider_value):
		recys_lines += f"<b style=font-family:verdana;font-size:20px;color:blue><a href={flinks[i]} target='_blank'>{query} + {TKs[i]}</a></b><br>"
	return f"<p style=font-family:verdana;color:green;font-size:20px;text-align:center;>" \
				 f"Since You searched for:<br>" \
				 f"<b><i font-size:30px;>{query}</i></b><br>" \
				 f"you might be also interested in:<br>" \
				 f"{recys_lines}" \
				 f"</p>"

def clean_recsys_entry(change):
	entry.value = ""
	entry.placeholder = "Enter your query keywords here..."
	recys_lbl.value = ""
	slider_value.layout.visibility = 'hidden'  # Hide slider

def rec_btn_click(change):
	query = entry.value
	if query and query != "Enter your query keywords here...":
		progress_bar.layout.visibility = 'visible'  # Show progress bar
		global TKs, flinks
		TKs=get_recsys_result(qu=query)
		flinks=[f"{digi_base_url}?query={urllib.parse.quote(f'{query} {tk}')}" for tk in TKs]
		#print(TKs)
		progress_bar.layout.visibility = 'hidden'  # Hide progress bar
		slider_value.layout.visibility = 'visible'  # Show slider
	else:
		recys_lbl.value = "<font color='red'>Enter a valid search query first</font>"
	update_recys_lbl(None)

def run_gui():
	GUI=widgets.VBox(
		[widgets.HBox([left_image_widget, widgets.Label(value=' '), right_image_widget], layout=vbox_layout),
		 welcome_lbl,
		 entry,
		 widgets.HBox([search_btn, widgets.Label(value=' '), clean_search_btn], layout=vbox_layout),
		 nlf_link_lable,
		 widgets.HBox([rec_btn, widgets.Label(value=' '), clean_recsys_btn], layout=vbox_layout),
		 slider_value,  # Added slider
		 recys_lbl,
		 progress_bar,  # Added progress bar
		 widgets.HBox([exit_btn], layout=vbox_layout),
		 countdown_lbl],
		layout=vbox_layout
	)
	display(GUI)

search_btn.on_click(get_nlf_link)
clean_search_btn.on_click(clean_search_entry)
slider_value.observe(update_recys_lbl, names='value') # real-time behavior
rec_btn.on_click(rec_btn_click)
clean_recsys_btn.on_click(clean_recsys_entry)