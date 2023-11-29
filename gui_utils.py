import ipywidgets as widgets
from IPython.display import display, HTML, Image
import urllib.parse
import requests
from PIL import Image as PILImage, ImageOps
from io import BytesIO

def close_window(count=8):
		if count > 0:
				countdown_lbl.value = f"Thanks for using our service, Have a Good Day!<br><br>closing in {count} sec..."
				time.sleep(1)
				close_window(count-1)
		else:
				display(HTML("<b>Bye</b>"))

def generate_link(change):
		query = entry.value
		if query and query != "Query keywords...":
				encoded_query = urllib.parse.quote(query)
				base_url = "https://digi.kansalliskirjasto.fi/search"
				gen_link=f"{base_url}?query={encoded_query}"
				nlf_link_lable.value=f"<b style=font-family:verdana;font-size:20px;color:blue><a href={gen_link} target='_blank'>Click here to open National Library Results</a></b>"
		else:
				nlf_link_lable.value = "<p style=font-family:Courier;font-size:18px;color:red>Oops! Enter a valid search query to proceed!</p>"

def recSys_cb(change):
		flink="https://www.google.com/"
		query = entry.value
		if query and query != "Query keywords...":
				recys_lbl.value=f"<p style=font-family:verdana;color:green;font-size:20px;text-align:center;>"\
												f"Since You searched for:<br>"\
												f"<b><i font-size:30px;>{query}</i></b><br>"\
												f"you might be also interested in:<br>"\
												f"<b style=font-family:verdana;font-size:20px;color:blue><a href={flink} target='_blank'>{query} + TK1</a></b><br>"\
												f"<b style=font-family:verdana;font-size:20px;color:blue><a href={flink} target='_blank'>{query} + TK2</a></b><br>"\
												f"<b style=font-family:verdana;font-size:20px;color:blue><a href={flink} target='_blank'>{query} + TK3</a></b><br>"\
												f"<b style=font-family:verdana;font-size:20px;color:blue><a href={flink} target='_blank'>{query} + TK4</a></b><br>"\
												f"<b style=font-family:verdana;font-size:20px;color:blue><a href={flink} target='_blank'>{query} + TK5</a></b><br>"\
												f"</p>"
		else:
				recys_lbl.value = "<font color='red'>Enter a valid search query first</font>"

#f"<b style=font-family:verdana;font-size:20px;color:blue><a href={gen_link} target='_blank'>Click here</a></b>"

def clean_search_entry(change):
		nlf_link_lable.value = ""
		entry.value = ""
		entry.placeholder = "Query keywords..."

def clean_recsys_entry(change):
		recys_lbl.value = ""

def on_entry_submit(change):
		on_entry_click(entry, None, None)

def on_entry_click(widget, event, data):
		if widget.value == "Query keywords...":
				widget.value = ""
				widget.style = {'description_width': 'initial', 'color': 'black'}

def run_gui():
	left_image_path = "https://www.topuniversities.com/sites/default/files/profiles/logos/tampere-university_5bbf14847d023f5bc849ec9a_large.jpg"
	right_image_path = "https://digi.kansalliskirjasto.fi/images/logos/logo_fi_darkblue.png"

	left_image = PILImage.open(BytesIO(requests.get(left_image_path).content))
	right_image = PILImage.open(BytesIO(requests.get(right_image_path).content))

	left_image_widget = widgets.Image(value=requests.get(left_image_path).content, format='png', width=300, height=300)
	right_image_widget = widgets.Image(value=requests.get(right_image_path).content, format='png', width=300, height=300)

	welcome_lbl = widgets.HTML(value="<h2 style=font-family:verdana;font-size:30px;color:black;text-align:center;>Welcome!<br>What are you looking after, today?</h2>")

	entry = widgets.Text(placeholder="Query keywords...", layout=widgets.Layout(width='850px'))

	search_btn = widgets.Button(description="Search NLF", layout=widgets.Layout(width='150px'))
	clean_search_btn = widgets.Button(description="Clean", layout=widgets.Layout(width='150px'))

	rec_btn = widgets.Button(description="Recommend Me", layout=widgets.Layout(width='150px'))
	clean_recsys_btn = widgets.Button(description="Clear", layout=widgets.Layout(width='150px'))

	search_btn.on_click(generate_link)
	clean_search_btn.on_click(clean_search_entry)

	nlf_link_lable = widgets.HTML(value="")
	#nlf_link_lable.observe(on_link_clicked, names='value')

	rec_btn.on_click(recSys_cb)
	clean_recsys_btn.on_click(clean_recsys_entry)

	recys_lbl = widgets.HTML()

	exit_btn = widgets.Button(description="Exit", layout=widgets.Layout(width='100px'))
	exit_btn.on_click(lambda x: close_window())

	countdown_lbl = widgets.HTML()

	# Display ipywidgets
	widgets.VBox(
			[widgets.HBox([left_image_widget, widgets.Label(value=' '), right_image_widget], layout=widgets.Layout(align_items='center')),
			welcome_lbl,
			entry,
			widgets.HBox([search_btn, widgets.Label(value=' '), clean_search_btn], layout=widgets.Layout(align_items='center')),
			nlf_link_lable,
			widgets.HBox([rec_btn, widgets.Label(value=' '), clean_recsys_btn], layout=widgets.Layout(align_items='center')),
			recys_lbl,
			widgets.HBox([exit_btn], layout=widgets.Layout(align_items='center')),
			countdown_lbl],
			layout=widgets.Layout(align_items='center')
	)