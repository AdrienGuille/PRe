import io
import os, sys
from itertools import chain
from gensim.models import FastText
from gensim.models.utils_any2vec import _ft_hash, _compute_ngrams
from NetworkVisjs import Network
from time import time
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import sklearn
import random
import threading
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity as cosSim
from sklearn.metrics.pairwise import euclidean_distances
from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, ColumnDataSource
from bokeh.models import Slider, ColumnDataSource, WidgetBox, HoverTool, TapTool, Div, GroupFilter, Selection, LinearColorMapper, Circle, ColorBar
from bokeh.layouts import layout, column, row, Spacer, widgetbox
from bokeh.models.widgets import Button, TextInput, Panel, Tabs, Select, RadioGroup, DataTable, TableColumn
from bokeh.models.renderers import GlyphRenderer

number_of_elements = 1000
number_of_neighbors = 10

model = None
vectors = []
words = []
ngrams = []
vectors_ngrams = []
words_ngrams = []
positions = []
iterations = []

# Storing each iteration for T-SNE
# Changing the gradient_descent function to store each iteration in the positions list for animation purpose
def _gradient_descent(objective, p0, it, n_iter, n_iter_check=1, n_iter_without_progress=300, momentum=0.8, learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        positions.append(p.copy())

        error, grad = objective(p, *args, **kwargs)
        grad_norm = linalg.norm(grad)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if (i + 1) % n_iter_check == 0:
            toc = time()
            duration = toc - tic
            tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

    return p, error, i
sklearn.manifold.t_sne._gradient_descent = _gradient_descent

print("Starting execution ...")
modelsList = [] # List of al model files found in the gensimModels folder
path = __file__[0:len(__file__)-7]+"gensimModels/"
for file in os.listdir(path):
    if file.endswith(".bin"):
        pair = (os.path.abspath(path+file), file[0:len(file)-4])
        modelsList.append(pair)

# Bokeh Data Sources
source = ColumnDataSource(data=dict(
    x=[],
    y=[],
    mots=[],
    color=[],
))
sourceTSNE = ColumnDataSource(data=dict(
    x=[],
    y=[],
    mots=[],
    color=[],
))
sourceTemp = ColumnDataSource(data=dict(
    x=[],
    y=[],
    mots=[],
    color=[],
))
sourceNetwork = ColumnDataSource(data=dict(
    label=[],
    edges=[],
    values=[],
    index=[],
    color=[]
))
sourceAnalogy = ColumnDataSource(data=dict(
    words=[],
    similarity=[]
))

## Preparing Plots and UI elements
TOOLS = "pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,tap,save".split(
    ',')
hover = HoverTool(tooltips=[
    ("Indice", "$index"),
    ("Mot", "@mots"),
])
TOOLS.append(hover)
# Selection colors
Cpalette = ['#1434B8','#147CBD','#14C2BC','#15C776','#15CC2D','#4CD216','#9FD716','#DCC317','#E17317','#E72018']
lcm = LinearColorMapper(palette=Cpalette, low=0, high=1)
color_bar_p = ColorBar(color_mapper=lcm, location=(0, 0))
color_bar_p2 = ColorBar(color_mapper=lcm, location=(0, 0))
selected_circle = Circle(fill_alpha=1, fill_color={'field' : 'color', 'transform':lcm}, line_color={'field' : 'color', 'transform':lcm})

#PCA Plot
p = figure(plot_width=600, plot_height=400, tools=TOOLS, output_backend="webgl", active_scroll='wheel_zoom')
p_circle = p.circle('x', 'y', size=7, source=source, color='#053061', fill_alpha=0.5)
p_circle.selection_glyph = selected_circle
p.add_layout(color_bar_p, 'left')
p.axis.visible = False
p.grid.visible = False
tab1 = Panel(child=p, title="PCA")

#TSNE Plot
p2 = figure(plot_width=600, plot_height=400, tools=TOOLS, output_backend="webgl", active_scroll='wheel_zoom')
p2_circle = p2.circle('x', 'y', size=7, source=sourceTSNE, color='#053061', fill_alpha=0.5)
p2_circle.selection_glyph = selected_circle
p2.add_layout(color_bar_p2, 'left')
p2.axis.visible = False
p2.grid.visible = False

#Network
p3 = Network(label="label", edges="edges", values="values", color="color", data_source=sourceNetwork, width=650, height=450)

#Widgets
tsnePerplexity = Slider(start=5, end=100, value=30, step=1, width=120, title="Perplexity")
tsneLearning = Slider(start=10, end=1000, value=200, step=1, width=120, title="Learning Rate")
tsneIteration = Slider(start=300, end=5000, value=500, step=50, width=120, title="Iterations")
tsneAnimationPosition = Slider(start=0, end=tsneIteration.value, step=1, width=400, title="Aller à Iteration")
tsneSpeed = Slider(start=10, end=100, value=70, step=1, width=120, title="Speed")
neighborsNumber = Slider (start=3, end=20, value=10, width=120, title="N° Voisinages")
neighborsApply = Button(label='Apply', button_type='success', width=60)
tsneApply = Button(label='Apply', button_type='success', width=80)
pauseB = Button(label='Pause', button_type='success', width=60)
startB = Button(label='Start', button_type='success', width=60)
stopB = Button(label='Stop', button_type='success', width=60)
modelSelect = Select(value="Choisir un modèle" ,options=modelsList)
tsneMetricSelect = Select(title="metrique", value='cosine', width=120, options=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'])
tsneLoading = Div()
LoadingDiv = Div()
informationDiv = Div(width=600)
NumberElementsDiv = Div(width=250)
iterationCount = Div(width=100)
minus = Div(text='-',width=15)
plus = Div(text='+',width=15)
word1 = TextInput(width=200, title="Analogie")
word2 = TextInput(width=200, title="-")
word3 = TextInput(width=200, title="+")
calculateAnalogy = Button(label='Calculer', button_type='success', width=60)
equals = Div(text=" ", width=120)
searchBox = TextInput(width=250, placeholder="Rechercher ...")
searchButton = Button(label='Rechercher', button_type='success', width=100)
equals.css_classes = ["center"]
# p3.css_classes = ["blackBorder"]
analogyColumns = [
    TableColumn(field="words", title="Mots"),
    TableColumn(field="similarity", title="Similarit\u00E9"),
]
analogyDataTable = DataTable(source=sourceAnalogy, columns=analogyColumns, index_position=None, width=500, height=200)
analogy = row(column(word1, word2, word3, row(calculateAnalogy, Spacer(width=20), equals)), Spacer(width=40), analogyDataTable)
tsneLayout = layout([
    [p2],
    [tsnePerplexity, Spacer(width=10), tsneLearning, Spacer(width=10), tsneIteration, Spacer(width=10), widgetbox(tsneMetricSelect, width=120), Spacer(width=20), widgetbox(tsneLoading, width=30)],
    [tsneSpeed, Spacer(width=10), pauseB, Spacer(width=10), startB, Spacer(width=10), stopB, Spacer(width=10), widgetbox(iterationCount, width=60), Spacer(width=10), tsneApply],
    [tsneAnimationPosition]
])
tab2 = Panel(child=tsneLayout, title="t-SNE")
renderer = p2.select(dict(type=GlyphRenderer))
ds = renderer[0].data_source

tab3 = Panel(child=analogy, title="Analogie")
tabs = Tabs(tabs=[tab1, tab2, tab3])

l = layout([
  [p3, tabs],
  [widgetbox(searchBox, searchButton), Spacer(width=20), widgetbox(neighborsNumber, width=120), Spacer(width=20), widgetbox(modelSelect, width=80)],
  [LoadingDiv]
], sizing_mode='fixed')

d1 = Div(text="<h2>Choix d'un mod\u00E8le</h2>", width=500)
d2 = Div(text="<h2>Choix d'un mot</h2>", width=500)
d3 = Div(text="<h2>Visualisation globale des repr\u00E9sentations</h2><br><h3>Vecteurs de dimension 100 projet\u00E9s dans le plan selon :</h3>", width=500)
projectionMethode = RadioGroup(labels=["la m\u00E9thode t-SNE en deux dimensions", "les deux premiers axes principaux"], active=0)
d4 = Div(text="<h2>Exploration des voisinages</h2><br><h3>Voisinages \u00E9tablis selon :</h3>", width=500)
similarityMethode = RadioGroup(labels=["la similarit\u00E9 cosinus", "la distance euclidienne"], active=0)

newLayout = layout([
    [d1],
    [modelSelect, Spacer(width=20), NumberElementsDiv],
    [d2],
    [searchBox],
    [searchButton, Spacer(width=20), informationDiv],
    [d3],
    [projectionMethode],
    [p2],
    [d4],
    [similarityMethode],
    [p3],
    [analogy],
], sizing_mode='fixed')

# Template for the generated HTML document using jinja2 templating language
template = """
{% block preamble %}
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css">
<style>
    .loader {
        border: 8px solid #f3f3f3;
        border-radius: 50%;
        border-top: 8px solid #3498db;
        width: 20px;
        height: 20px;
        -webkit-animation: spin 2s linear infinite;
        /* Safari */
        animation: spin 2s linear infinite;
    }

    .center {
        padding: 5px 0;
        text-align: center;
        border: 3px solid green;
    }

    .DivWithScroll {
        height: 120px;
        overflow: scroll;
        overflow-x: hidden;
    }

    .blackBorder {
        border:1px solid black
    }

    /* Safari */

    @-webkit-keyframes spin {
        0% {
            -webkit-transform: rotate(0deg);
        }
        100% {
            -webkit-transform: rotate(360deg);
        }
    }

    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }
    /* Absolute Center Spinner */
    .loading {
      position: fixed;
      z-index: 999;
      height: 2em;
      width: 2em;
      overflow: show;
      margin: auto;
      top: 40%;
      left: 0;
      bottom: 0;
      right: 0;
    }

    /* Transparent Overlay */
    .loading:before {
      content: '';
      display: block;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background-color: rgba(0,0,0,0.3);
    }

    /* :not(:required) hides these rules from IE9 and below */
    .loading:not(:required) {
      /* hide "loading..." text */
      font: 0/0 a;
      color: transparent;
      text-shadow: none;
      background-color: transparent;
      border: 0;
    }

    .loading:not(:required):after {
      content: '';
      display: block;
      font-size: 10px;
      width: 1em;
      height: 1em;
      margin-top: -0.5em;
      -webkit-animation: spinner 1500ms infinite linear;
      -moz-animation: spinner 1500ms infinite linear;
      -ms-animation: spinner 1500ms infinite linear;
      -o-animation: spinner 1500ms infinite linear;
      animation: spinner 1500ms infinite linear;
      border-radius: 0.5em;
      -webkit-box-shadow: rgba(0, 0, 0, 0.75) 1.5em 0 0 0, rgba(0, 0, 0, 0.75) 1.1em 1.1em 0 0, rgba(0, 0, 0, 0.75) 0 1.5em 0 0, rgba(0, 0, 0, 0.75) -1.1em 1.1em 0 0, rgba(0, 0, 0, 0.5) -1.5em 0 0 0, rgba(0, 0, 0, 0.5) -1.1em -1.1em 0 0, rgba(0, 0, 0, 0.75) 0 -1.5em 0 0, rgba(0, 0, 0, 0.75) 1.1em -1.1em 0 0;
      box-shadow: rgba(0, 0, 0, 0.75) 1.5em 0 0 0, rgba(0, 0, 0, 0.75) 1.1em 1.1em 0 0, rgba(0, 0, 0, 0.75) 0 1.5em 0 0, rgba(0, 0, 0, 0.75) -1.1em 1.1em 0 0, rgba(0, 0, 0, 0.75) -1.5em 0 0 0, rgba(0, 0, 0, 0.75) -1.1em -1.1em 0 0, rgba(0, 0, 0, 0.75) 0 -1.5em 0 0, rgba(0, 0, 0, 0.75) 1.1em -1.1em 0 0;
    }

    /* Animation */

    @-webkit-keyframes spinner {
      0% {
        -webkit-transform: rotate(0deg);
        -moz-transform: rotate(0deg);
        -ms-transform: rotate(0deg);
        -o-transform: rotate(0deg);
        transform: rotate(0deg);
      }
      100% {
        -webkit-transform: rotate(360deg);
        -moz-transform: rotate(360deg);
        -ms-transform: rotate(360deg);
        -o-transform: rotate(360deg);
        transform: rotate(360deg);
      }
    }
    @-moz-keyframes spinner {
      0% {
        -webkit-transform: rotate(0deg);
        -moz-transform: rotate(0deg);
        -ms-transform: rotate(0deg);
        -o-transform: rotate(0deg);
        transform: rotate(0deg);
      }
      100% {
        -webkit-transform: rotate(360deg);
        -moz-transform: rotate(360deg);
        -ms-transform: rotate(360deg);
        -o-transform: rotate(360deg);
        transform: rotate(360deg);
      }
    }
    @-o-keyframes spinner {
      0% {
        -webkit-transform: rotate(0deg);
        -moz-transform: rotate(0deg);
        -ms-transform: rotate(0deg);
        -o-transform: rotate(0deg);
        transform: rotate(0deg);
      }
      100% {
        -webkit-transform: rotate(360deg);
        -moz-transform: rotate(360deg);
        -ms-transform: rotate(360deg);
        -o-transform: rotate(360deg);
        transform: rotate(360deg);
      }
    }
    @keyframes spinner {
      0% {
        -webkit-transform: rotate(0deg);
        -moz-transform: rotate(0deg);
        -ms-transform: rotate(0deg);
        -o-transform: rotate(0deg);
        transform: rotate(0deg);
      }
      100% {
        -webkit-transform: rotate(360deg);
        -moz-transform: rotate(360deg);
        -ms-transform: rotate(360deg);
        -o-transform: rotate(360deg);
        transform: rotate(360deg);
      }
    }
</style>
{% endblock %}

{% block body %}
<body>
    <h1 align="center">Exploration des repr&#233sentation vectorielles des mots</h1>
    <h6 align="center"><i>Laboratoire ERIC</i></h6>
    <h6 align="center"><i>par : Abderahmen Masmoudi</i></h6>
    {{ self.inner_body() }}
</body>
{% endblock %}
"""

# Adding elements to the document
curdoc().add_root(LoadingDiv)
curdoc().title = "Embedding"
curdoc().template = template
curdoc().add_root(newLayout)

# Neighbors
def generateColor():
    """
        Generate and return a random color
    """
    color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color
def selectionTSNE(attr, old, new):
    """
        Handler function called when the user selects a word from the TSNE plot
        Finds nearest neighbors to the selected word and creates the corresponding network
    """
    global vectors
    if (len(new.indices) != 0) and (selectionTSNE.update == False) :
        selectionTSNE.update = True
        wordIndex = new.indices[0]
        if(tsneMetricSelect.value == 'euclidean'):
            v = euclidean_distances(vectors[0:number_of_elements],[vectors[wordIndex]])
            v = list(chain.from_iterable(v))
        else:
            v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors[0:number_of_elements]]
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        p2_circle.data_source.add(data=v,name='color')
        p_circle.data_source.add(data=v,name='color')
        sourceTemp.add(data=v,name='color')
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=(tsneMetricSelect.value != 'euclidean'))
        l = [new.indices[0]]
        sourceNetwork.data['label'] = [words[wordIndex]]
        sourceNetwork.data['edges'] = [[]]
        sourceNetwork.data['values'] = [[]]
        sourceNetwork.data['index'] = [wordIndex]
        sourceNetwork.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors+1):
            l.append(sortedSim[i][0])
            sourceNetwork.data['label'].append(words[sortedSim[i][0]])
            sourceNetwork.data['edges'].append([1])
            sourceNetwork.data['values'].append([sortedSim[i][1]])
            sourceNetwork.data['index'].append(sortedSim[i][0])
            sourceNetwork.data['color'].append(color)
        p2_circle.data_source.selected.indices = l
        p2_circle.data_source.trigger('selected',None,p2_circle.data_source.selected)
        sourceNetwork.trigger('data', None, sourceNetwork)
        selectionTSNE.update = False

def selection(attr, old, new):
    """
        Handler function called when the user selects a word from the TSNE plot
        Finds nearest neighbors to the selected word and creates the corresponding network
    """
    global vectors
    if (len(new.indices) != 0) and (selection.update == False) :
        selection.update = True
        wordIndex = new.indices[0]
        if(tsneMetricSelect.value == 'euclidean'):
            v = euclidean_distances(vectors[0:number_of_elements],[vectors[wordIndex]])
            v = list(chain.from_iterable(v))
        else:
            v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors[0:number_of_elements]]
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        p_circle.data_source.add(data=v,name='color')
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=(tsneMetricSelect.value != 'euclidean'))
        l = [new.indices[0]]
        sourceNetwork.data['label'] = [words[wordIndex]]
        sourceNetwork.data['edges'] = [[]]
        sourceNetwork.data['values'] = [[]]
        sourceNetwork.data['index'] = [wordIndex]
        sourceNetwork.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors+1):
            l.append(sortedSim[i][0])
            sourceNetwork.data['label'].append(words[sortedSim[i][0]])
            sourceNetwork.data['edges'].append([1])
            sourceNetwork.data['values'].append([sortedSim[i][1]])
            sourceNetwork.data['index'].append(sortedSim[i][0])
            sourceNetwork.data['color'].append(color)
        p_circle.data_source.selected.indices = l
        p_circle.data_source.trigger('selected',None,p_circle.data_source.selected)
        sourceNetwork.trigger('data', None, sourceNetwork)
        selection.update = False
        
selectionTSNE.update = False
selection.update = False

def pcaProcess():
    """
        Apply PCA on the first 'number_of_elements' word vectors and store the result in source
    """
    pca = PCA(n_components=2)
    pca.fit(vectors[0:number_of_elements])
    transform = pca.transform(vectors[0:number_of_elements])
    print("PCA done ...")
    source.data['x'] = transform[:, 0]
    source.data['y'] = transform[:, 1]
    source.data['mots'] = words[0:number_of_elements]
    source.data['color'] = ['#053061' for i in range(0,number_of_elements)]
    source.selected.indices = []

    source.trigger('data', None, source.data)

def loadModelandPCA(modelName=modelSelect.value):
    """
        Load selected model from the selection box and applies PCA on the first 'number_of_elements' word vectors
    """
    global model
    global vectors
    global words
    global ngrams
    global vectors_ngrams
    global words_ngrams
    global source
    global sourceTSNE
    global sourceNetwork

    LoadingDiv.css_classes = ["loading"]
    model = FastText.load(modelName)
    # model = FastText.load_fasttext_format('PRe_git/model/'+modelName+'.bin', encoding='ISO-8859-15')
    ## model = fasttext.load_model('PRe/model/model.bin')
    print("Data loaded ...")
    # vectors = [list(line) for line in data.values()]
    # words = list(data)
    vectors = model.wv.syn0
    words = model.wv.index2word
    ngrams = []
    vectors_ngrams = []
    words_ngrams = []
    for word in words:
        ngrams += _compute_ngrams(word, model.min_n, model.max_n)
    ngrams = set(ngrams)
    print('Ngrams done ...')
    i=0
    for ngram in ngrams:
        ngram_hash = _ft_hash(ngram) % model.bucket
        if ngram_hash in model.wv.hash2index:
            i += 1
            words_ngrams.append(ngram)
            vectors_ngrams.append(model.wv.vectors_ngrams[model.wv.hash2index[ngram_hash]])
    gr = _compute_ngrams(words[466], model.min_n, model.max_n)
    for g in gr:
        print(words[466], g, model.wv.similarity(words[466],g))
    
    NumberElementsDiv.text = "Nombre total de mots : "+str(len(words))
    d3.text = "<h2>Visualisation globale des repr\u00E9sentations</h2><br><h3>Vecteurs de dimension "+str(len(vectors[0]))+" projet\u00E9s dans le plan selon :</h3>"

    # PCA
    pcaProcess()

    sourceTSNE.data['x'] = [0 for i in range(0,number_of_elements)]
    sourceTSNE.data['y'] = [0 for i in range(0,number_of_elements)]
    sourceTSNE.data['mots'] = words[0:number_of_elements]
    sourceTSNE.data['color'] = ['#053061' for i in range(0,number_of_elements)]
    sourceTSNE.selected.indices = []

    sourceNetwork.data['label'] = []
    sourceNetwork.data['edges'] = []
    sourceNetwork.data['values'] = []
    sourceNetwork.data['index'] = []
    sourceNetwork.data['color'] = []

    print("Source done ...")
    source.trigger('data', None, source)
    sourceTSNE.trigger('data', None, sourceTSNE)
    sourceNetwork.trigger('data', None, sourceNetwork)


    tabs.active = 0
    tabChange.first = False

    LoadingDiv.css_classes = []

#T-SNE representation Function
def tsneProcess():
    """
        Apply t-SNE reduction on the first 'number_of_elements' word vectors
    """
    global positions
    global iterations
    global vectors
    global sourceTemp

    positions = []
    LoadingDiv.css_classes = ["loading"]
    TSNE_transform = TSNE(n_components=2, n_iter=tsneIteration.value, perplexity=tsnePerplexity.value, learning_rate=tsneLearning.value, metric=tsneMetricSelect.value).fit_transform(vectors[0:number_of_elements])
    print("TSNE done ...")
    iterations = np.dstack(position.reshape(-1, 2) for position in positions)
    print("iterations done ...")
    sourceTSNE.data['x'] = TSNE_transform[:, 0]
    sourceTSNE.data['y'] = TSNE_transform[:, 1]
    sourceTSNE.data['mots'] = words[0:number_of_elements]
    sourceTSNE.trigger('data',None,sourceTSNE.data)
    sourceTemp.data = sourceTSNE.data
    tsneAnimationPosition.end = tsneIteration.value
    tsneAnimationPosition.value = tsneIteration.value
    LoadingDiv.css_classes = []

#T-SNE Animation Function
iterationNumber = 0
def tsne_animation(length=number_of_elements, div=iterationCount, goto=False):
    """
        Create animation from the t-SNE transformation
    """
    global ds
    global iterationNumber
    global iterations
    if iterationNumber < iterations.shape[2]:
        for i in range(0,length):
            ds.data['x'][i] = iterations[i][0][iterationNumber]
            ds.data['y'][i] = iterations[i][1][iterationNumber]
        
        ds.trigger('data',ds.data,ds.data)
        if(goto==False):
            tsneAnimationPosition.value = iterationNumber
        iterationNumber += 1
        div.text = "Iteration N&#176 : " + str(iterationNumber)
    else:
        stopAnimation()

animation = None
def animate():
    """
        Function used to start or resume animation when user clicks on start/resume
    """
    global animation
    if (animation == None):
        animation = curdoc().add_periodic_callback(tsne_animation,110-tsneSpeed.value)
def destroyAnimation():
    """
        Used to pause animation when user clicks on stop button
    """
    global animation
    if (animation != None):
        curdoc().remove_periodic_callback(animation)
        animation = None
        startB.label = "Resume"
def stopAnimation():
    """
        Used to stop animation when user clicks on stop button
    """
    global animation
    global iterationNumber
    if (animation != None):
        curdoc().remove_periodic_callback(animation)
        animation = None
        iterationNumber = 0
        startB.label = "Start"
def changeSpeed(attr, old, new):
    """
        Change the speed of the t-SNE animation based on the value of the Slider tsneSpeed
    """
    global animation
    if (animation != None):
        curdoc().remove_periodic_callback(animation)
        animation = None
        animation = curdoc().add_periodic_callback(tsne_animation,int(110-new))

def gotoPosition(attr, old, new):
    """
        Go to specific iteration in the t-SNE animation
    """
    global iterationNumber
    iterationNumber = new
    tsne_animation(goto=True)
def tabChange(attr, old, new):
    """
        Starts the t-SNE transformation when changing to the t-SNE tab for the first time
    """
    if(new == 1) and (tabChange.first == False):
        tabChange.first = True
        tsneProcess()
tabChange.first = False

def addNeighborNodes(index):
    """
        Find neighbor words when double clicking on a node and adds the new neighbor nodes to the Netwrok representation
    """
    global vectors
    wordIndex = sourceNetwork.data['index'][index]
    if(tsneMetricSelect.value == 'euclidean'):
        v = euclidean_distances(vectors[0:number_of_elements],[vectors[wordIndex]])
        v = list(chain.from_iterable(v))
    else:
        v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors[0:number_of_elements]]
    similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
    sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=(tsneMetricSelect.value != 'euclidean'))
    color = generateColor()
    for i in range(1, number_of_neighbors+1):
        if (sortedSim[i][0] in sourceNetwork.data['index']) and (sourceNetwork.data['index'].index(sortedSim[i][0])+1 not in sourceNetwork.data['edges'][index]):
            indice = sourceNetwork.data['index'].index(sortedSim[i][0])
            sourceNetwork.data['edges'][indice].append(index+1)
            sourceNetwork.data['values'][indice].append(sortedSim[i][1])
        elif (sortedSim[i][0] not in sourceNetwork.data['index']):
            sourceNetwork.data['label'].append(words[sortedSim[i][0]])
            sourceNetwork.data['edges'].append([index+1])
            sourceNetwork.data['values'].append([sortedSim[i][1]])
            sourceNetwork.data['index'].append(sortedSim[i][0])
            sourceNetwork.data['color'].append(color)
    sourceNetwork.trigger('data', None, sourceNetwork)

def selectNode(attr, old, new):
    addNeighborNodes(new)

def changeNeighbors(attr, old, new):
    """
        Change the number of neighbors according to the value of the Slider 'neighborsNumber'
    """
    global number_of_neighbors
    number_of_neighbors = new

def chargeModel():
    """
        Charge model at the start of execution
    """
    loadModelandPCA(modelName=modelSelect.value)
    tsneProcess()

def changeModel(attr, old, new):
    """
        Charge selected model from the selection box
    """
    loadModelandPCA(modelName=new)
    tsneProcess()

def calcAnalogy():
    """
        Find most similar word
    """
    word = model.wv.most_similar(positive=[word1.value,word3.value], negative=[word2.value])
    equals.text = "<b> <center>"+word[0][0]+"</center> </b>"
    sourceAnalogy.data["words"] = [word[i][0] for i in range(0,10)]
    sourceAnalogy.data["similarity"] = [word[i][1] for i in range(0,10)]
    sourceAnalogy.trigger('data', None, sourceAnalogy.data)

def searchWord():
    """
        Search for first 'number_of_elements' most similar words to the typed word and and applies PCA and t-SNE on thein vectors
    """
    global vectors
    global words
    word = searchBox.value
    selectionTSNE.update = True
    selection.update = True
    if (word in words):
        informationDiv.text = "Ce mot existe dans le vocabulaire."
        wordIndex = words.index(word)
        v = []
        LoadingDiv.css_classes = ["loading"]
        if(tsneMetricSelect.value == 'euclidean'):
            v = euclidean_distances(vectors,[vectors[wordIndex]])
            v = list(chain.from_iterable(v))
        else:
            v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors]
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=(tsneMetricSelect.value != 'euclidean'))
        changedVectors = [vectors[i[0]] for i in sortedSim]
        changedWords = [words[i[0]] for i in sortedSim]
        changedValues = [v[i[0]] for i in sortedSim]
        vectors = changedVectors
        words = changedWords
        v = changedValues
        pcaProcess()
        tsneProcess()
        wordIndex = 0
        l = [i for i in range(0,number_of_neighbors+1)]
        sourceNetwork.data['label'] = [words[wordIndex]]
        sourceNetwork.data['edges'] = [[]]
        sourceNetwork.data['values'] = [[]]
        sourceNetwork.data['index'] = [wordIndex]
        sourceNetwork.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors+1):
            # l.append(sortedSim[i][0])
            sourceNetwork.data['label'].append(words[i])
            sourceNetwork.data['edges'].append([1])
            sourceNetwork.data['values'].append([sortedSim[i][1]])
            sourceNetwork.data['index'].append(i)
            sourceNetwork.data['color'].append(color)
        p2_circle.data_source.add(data=v,name='color')
        p2_circle.data_source.selected.indices = l
        p2_circle.data_source.trigger('selected',None,p2_circle.data_source.selected)
        p_circle.data_source.add(data=v,name='color')
        p_circle.data_source.selected.indices = l
        p_circle.data_source.trigger('selected',None,p_circle.data_source.selected)
        sourceNetwork.trigger('data', None, sourceNetwork)
    else:
        informationDiv.text = "Ce mot n'existe pas dans le vocabulaire."
        newWordVector = model[searchBox.value]
        LoadingDiv.css_classes = ["loading"]
        if(tsneMetricSelect.value == 'euclidean'):
            v = euclidean_distances(vectors,[newWordVector])
            v = list(chain.from_iterable(v))
        else:
            v = [cosSim( np.asarray([newWordVector]), np.asarray([b]) )[0][0] for b in vectors]
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=(tsneMetricSelect.value != 'euclidean'))
        changedVectors = [vectors[i[0]] for i in sortedSim]
        changedWords = [words[i[0]] for i in sortedSim]
        vectors = changedVectors
        words = changedWords
        informationDiv.text =  informationDiv.text+" Le mot le plus proche trouv\u00E9 est : "+words[sortedSim[0][0]]
        pcaProcess()
        tsneProcess()
        l = [i for i in range(0,number_of_neighbors+1)]
        sourceNetwork.data['label'] = [words[sortedSim[0][0]]]
        sourceNetwork.data['edges'] = [[]]
        sourceNetwork.data['values'] = [[]]
        sourceNetwork.data['index'] = [sortedSim[0][0]]
        sourceNetwork.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors+1):
            # l.append(sortedSim[i][0])
            sourceNetwork.data['label'].append(words[sortedSim[i][0]])
            sourceNetwork.data['edges'].append([1])
            sourceNetwork.data['values'].append([sortedSim[i][1]])
            sourceNetwork.data['index'].append(sortedSim[i][0])
            sourceNetwork.data['color'].append(color)
        p2_circle.data_source.add(data=v,name='color')
        p2_circle.data_source.selected.indices = l
        p2_circle.data_source.trigger('selected',None,p2_circle.data_source.selected)
        p_circle.data_source.add(data=v,name='color')
        p_circle.data_source.selected.indices = l
        p_circle.data_source.trigger('selected',None,p_circle.data_source.selected)
        sourceNetwork.trigger('data', None, sourceNetwork)

    selectionTSNE.update = False
    selection.update = False

def radioPCA_TSNE(attr, old, new):
    if (new == 1):
        sourceTSNE.data = source.data
        sourceTSNE.trigger('data', None, sourceTSNE)
    else:
        sourceTSNE.data = sourceTemp.data
        sourceTSNE.trigger('data', None, sourceTSNE)

def radioSimilarity(attr, old, new):
    if(new == 1):
        tsneMetricSelect.value = 'euclidean'
        changeColorPalette(0,4)
    else:
        tsneMetricSelect.value = 'cosine'
        changeColorPalette(0,1)
    chargeModel()

def changeColorPalette(low, high):
    global Cpalette

    lcm.low = low
    lcm.high = high
    Cpalette = list(reversed(Cpalette))
    lcm.palette = Cpalette

pauseB.on_click(destroyAnimation)
startB.on_click(animate)
stopB.on_click(stopAnimation)
tsneSpeed.on_change('value', changeSpeed)
tsneAnimationPosition.on_change('value', gotoPosition)
neighborsNumber.on_change('value',changeNeighbors)
modelSelect.on_change('value', changeModel)
tabs.on_change('active', tabChange)
tsneApply.on_click(tsneProcess)
p_circle.data_source.on_change('selected',selection)
p2_circle.data_source.on_change('selected',selectionTSNE)
p3.on_change('selected',selectNode)
calculateAnalogy.on_click(calcAnalogy)
searchButton.on_click(searchWord)
projectionMethode.on_change('active', radioPCA_TSNE)
similarityMethode.on_change('active', radioSimilarity)

#curdoc().add_timeout_callback(chargeModel,3000)