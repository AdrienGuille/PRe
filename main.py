import io
import os
from itertools import chain
from gensim.models import FastText
from gensim.models.utils_any2vec import _ft_hash, _compute_ngrams
from NetworkVisjs import Network
from time import time
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sklearn
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity as cosSim
from sklearn.metrics.pairwise import euclidean_distances
from bokeh.io import curdoc
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import CustomJS, Slider, ColumnDataSource, WidgetBox, HoverTool, TapTool, Div, CDSView, GroupFilter, Selection, LinearColorMapper, Circle, ColorBar
from bokeh.layouts import layout, column, row, Spacer, widgetbox
from bokeh.models.widgets import Button, AutocompleteInput, TextInput, Panel, Tabs, Select, RadioGroup
from bokeh.models.callbacks import CustomJS
from bokeh.models.renderers import GlyphRenderer
from bokeh.server.server import BaseServer
from bokeh.events import SelectionGeometry

number_of_elements = 2000
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

def to_lowercase(sentences):
    """Convert all characters to lowercase from list of tokenized words"""
    new_sentences = []
    for sentence in sentences:
        new_words = []
        for word in sentence:
            new_word = word.lower()
            new_words.append(new_word)
        new_sentences.append(new_words)
    return new_sentences

print("Starting execution ...")
modelsList = []
for file in os.listdir("new_layout/gensimModels"):
    if file.endswith(".bin"):
        pair = (os.path.abspath("new_layout/gensimModels/"+file), file[0:len(file)-4])
        modelsList.append(pair)
print(modelsList)
#data = load_vectors("PRe/vectors/newWiki300.vec")
## Training data with fasttext
# model = fasttext.skipgram('PRe/text/wikipediaTXT.txt','model',bucket=50000)
'''
f = open("PRe_new/text/text.txt", encoding='ISO-8859-1')
txt = f.read()
List = [word_tokenize(t) for t in sent_tokenize(txt)]
List = to_lowercase(List)
model = FastText(List, sg=1, size=300, workers=4, min_count=1)
'''

# Representation with Bokeh
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

## Preparing Plots and UI elements
TOOLS = "pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,tap,save".split(
    ',')
hover = HoverTool(tooltips=[
    ("Indice", "$index"),
    ("Mot", "@mots"),
])
TOOLS.append(hover)
Cpalette = ['#E72018','#E17317','#DCC317','#9FD716','#4CD216','#15CC2D','#15C776','#14C2BC','#147CBD','#1434B8']
lcm = LinearColorMapper(palette=Cpalette, low=0, high=1)
color_bar_p = ColorBar(color_mapper=lcm, location=(0, 0))
color_bar_p2 = ColorBar(color_mapper=lcm, location=(0, 0))
selected_circle = Circle(radius=2000, fill_alpha=1, fill_color={'field' : 'color', 'transform':lcm}, line_color={'field' : 'color', 'transform':lcm})

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
tsneIteration = Slider(start=300, end=5000, value=1000, step=50, width=120, title="Iterations")
tsneAnimationPosition = Slider(start=0, end=tsneIteration.value, step=1, width=400, title="Goto Iteration")
tsneSpeed = Slider(start=10, end=100, value=70, step=1, width=120, title="Speed")
neighborsNumber = Slider (start=3, end=20, value=10, width=120, title="N° Neighbors")
neighborsApply = Button(label='Apply', button_type='success', width=60)
tsneApply = Button(label='Apply', button_type='success', width=80)
pauseB = Button(label='Pause', button_type='success', width=60)
startB = Button(label='Start', button_type='success', width=60)
stopB = Button(label='Stop', button_type='success', width=60)
modelSelect = Select(value=modelsList[0][0], options=modelsList)
tsneMetricSelect = Select(title="metric", value='cosine', width=120, options=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'])
tsneLoading = Div()
LoadingDiv = Div()
iterationCount = Div(width=100)
minus = Div(text='-',width=15)
plus = Div(text='+',width=15)
word1 = TextInput(width=200, title="Analogy")
word2 = TextInput(width=200, title="-")
word3 = TextInput(width=200, title="+")
calculateAnalogy = Button(label='Equals', button_type='success', width=60)
equals = Div(text=" ", width=120)
searchBox = TextInput(width=150, placeholder="Search ...")
searchButton = Button(label='Search', button_type='success', width=60)
equals.css_classes = ["center"]
# p3.css_classes = ["blackBorder"]
analogy = column(word1, word2, word3, row(calculateAnalogy, Spacer(width=20), equals))
tsneLayout = layout([
    [p2],
    [tsnePerplexity, Spacer(width=10), tsneLearning, Spacer(width=10), tsneIteration, Spacer(width=10), widgetbox(tsneMetricSelect, width=120), Spacer(width=20), widgetbox(tsneLoading, width=30)],
    [tsneSpeed, Spacer(width=10), pauseB, Spacer(width=10), startB, Spacer(width=10), stopB, Spacer(width=10), widgetbox(iterationCount, width=60), Spacer(width=10), tsneApply],
    [tsneAnimationPosition]
])
tab2 = Panel(child=tsneLayout, title="t-SNE")
renderer = p2.select(dict(type=GlyphRenderer))
ds = renderer[0].data_source

tab3 = Panel(child=analogy, title="Analogy")
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
	[modelSelect],
	[d2],
	[widgetbox(searchBox, searchButton)],
	[d3],
	[projectionMethode],
	[p2],
	[d4],
	[similarityMethode],
	[p3],
], sizing_mode='fixed')

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
    {{ self.inner_body() }}
</body>
{% endblock %}
"""
curdoc().add_root(LoadingDiv)
curdoc().title = "Embedding"
curdoc().template = template
curdoc().add_root(newLayout)

nltk.download('punkt')

# Neighbors
def generateColor():
    color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color
def handlerTSNE(attr, old, new):
    global vectors
    if (len(new.indices) != 0) and (handlerTSNE.update == False) :
        handlerTSNE.update = True
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
        handlerTSNE.update = False

def handler(attr, old, new):
    global vectors
    if (len(new.indices) != 0) and (handler.update == False) :
        handler.update = True
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
        handler.update = False
        
handlerTSNE.update = False
handler.update = False

def loadModelandPCA(modelName=modelSelect.value):
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
    print(len(vectors))
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
    # PCA Test
    pca = PCA(n_components=2)
    pca.fit(vectors[0:number_of_elements])
    transform = pca.transform(vectors[0:number_of_elements])
    print("PCA done ...")

    source.data['x'] = transform[:, 0]
    source.data['y'] = transform[:, 1]
    source.data['mots'] = words[0:number_of_elements]
    source.data['color'] = ['#053061' for i in range(0,number_of_elements)]
    source.selected.indices = []

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

# T-SNE representation Function
def tsneProcess():
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
    ds.data['x'] = TSNE_transform[:, 0]
    ds.data['y'] = TSNE_transform[:, 1]
    ds.trigger('data',ds.data,ds.data)
    sourceTemp.data = sourceTSNE.data
    tsneAnimationPosition.end = tsneIteration.value
    tsneAnimationPosition.value = tsneIteration.value
    LoadingDiv.css_classes = []

#T-SNE Animation Function
iterationNumber = 0
def tsne_animation(length=number_of_elements, div=iterationCount, goto=False):
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
    global animation
    if (animation == None):
        animation = curdoc().add_periodic_callback(tsne_animation,110-tsneSpeed.value)
def destroyAnimation():
    global animation
    if (animation != None):
        curdoc().remove_periodic_callback(animation)
        animation = None
        startB.label = "Resume"
def stopAnimation():
    global animation
    global iterationNumber
    if (animation != None):
        curdoc().remove_periodic_callback(animation)
        animation = None
        iterationNumber = 0
        startB.label = "Start"
def changeSpeed(attr, old, new):
    global animation
    if (animation != None):
        curdoc().remove_periodic_callback(animation)
        animation = None
        animation = curdoc().add_periodic_callback(tsne_animation,int(110-new))
def gotoPosition(attr, old, new):
	global iterationNumber
	iterationNumber = new
	tsne_animation(goto=True)
def tabChange(attr, old, new):
    if(new == 1) and (tabChange.first == False):
        tabChange.first = True
        tsneProcess()
tabChange.first = False

def addNeighborNodes(index):
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
	global number_of_neighbors
	number_of_neighbors = new

def chargeModel():
    loadModelandPCA(modelName=modelSelect.value)
    tsneProcess()

def changeModel(attr, old, new):
    loadModelandPCA(modelName=new)
    tsneProcess()

def calcAnalogy():
    word = model.wv.most_similar(positive=[word1.value,word3.value], negative=[word2.value])
    equals.text = "<b> <center>"+word[0][0]+"</center> </b>"

def searchWord():
    global vectors
    global words
    word = searchBox.value
    if (word in words[0:number_of_elements]):
        handlerTSNE.update = True
        handler.update = True
        wordIndex = words.index(word)
        v = []
        if(tsneMetricSelect.value == 'euclidean'):
            v = euclidean_distances(vectors[0:number_of_elements],[vectors[wordIndex]])
            v = list(chain.from_iterable(v))
        else:
            v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors[0:number_of_elements]]
        print(v)
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=(tsneMetricSelect.value != 'euclidean'))
        l = [wordIndex]
        sourceNetwork.data['label'] = [words[wordIndex]]
        sourceNetwork.data['edges'] = [[]]
        sourceNetwork.data['values'] = [[]]
        sourceNetwork.data['index'] = [wordIndex]
        sourceNetwork.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors):
            l.append(sortedSim[i][0])
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
        handlerTSNE.update = False
        handler.update = False
    else:
        handlerTSNE.update = True
        handler.update = True
        newWordVector = model[searchBox.value]
        if(tsneMetricSelect.value == 'euclidean'):
            v = euclidean_distances(vectors[0:number_of_elements],[newWordVector])
            v = list(chain.from_iterable(v))
        else:
            v = [cosSim( np.asarray([newWordVector]), np.asarray([b]) )[0][0] for b in vectors[0:number_of_elements]]
        print(v)
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=(tsneMetricSelect.value != 'euclidean'))
        l = [sortedSim[0][0]]
        sourceNetwork.data['label'] = [words[sortedSim[0][0]]]
        sourceNetwork.data['edges'] = [[]]
        sourceNetwork.data['values'] = [[]]
        sourceNetwork.data['index'] = [sortedSim[0][0]]
        sourceNetwork.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors):
            l.append(sortedSim[i][0])
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
        handlerTSNE.update = False
        handler.update = False

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
    else:
        tsneMetricSelect.value = 'cosine'
    print(tsneMetricSelect.value)
    chargeModel()

pauseB.on_click(destroyAnimation)
startB.on_click(animate)
stopB.on_click(stopAnimation)
tsneSpeed.on_change('value', changeSpeed)
tsneAnimationPosition.on_change('value', gotoPosition)
neighborsNumber.on_change('value',changeNeighbors)
modelSelect.on_change('value', changeModel)
tabs.on_change('active', tabChange)
tsneApply.on_click(tsneProcess)
p_circle.data_source.on_change('selected',handler)
p2_circle.data_source.on_change('selected',handlerTSNE)
p3.on_change('selected',selectNode)
calculateAnalogy.on_click(calcAnalogy)
searchButton.on_click(searchWord)
projectionMethode.on_change('active', radioPCA_TSNE)
similarityMethode.on_change('active', radioSimilarity)

curdoc().add_timeout_callback(chargeModel,3000)