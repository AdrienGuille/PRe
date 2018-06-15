import io
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
from bokeh.io import curdoc
from bokeh.plotting import figure, show, output_file, ColumnDataSource
from bokeh.models import CustomJS, Slider, ColumnDataSource, WidgetBox, HoverTool, TapTool, Div, CDSView, GroupFilter, Selection, LinearColorMapper, Circle, ColorBar
from bokeh.layouts import layout, column, row, Spacer, widgetbox
from bokeh.models.widgets import Button, AutocompleteInput, TextInput, Panel, Tabs
from bokeh.models.callbacks import CustomJS
from bokeh.models.renderers import GlyphRenderer
from bokeh.palettes import Plasma256
from bokeh.server.server import BaseServer
from bokeh.events import SelectionGeometry

nltk.download('punkt')

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

# Storing each iteration for T-SNE
positions = []
X_iter = []
def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
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
model = FastText.load_fasttext_format('PRe_new/model/model.bin', encoding='ISO-8859-15')
## model = fasttext.load_model('PRe/model/model.bin')
##
print("Data loaded ...")
# vectors = [list(line) for line in data.values()]
# words = list(data)
number_of_elements = 1000
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
gr = _compute_ngrams(words[1799], model.min_n, model.max_n)
for g in gr:
    print(words[1799], g, model.wv.similarity(words[100],g))
# del model
# PCA Test
pca = PCA(n_components=2)
pca.fit(vectors[0:number_of_elements])
# print(pca.explained_variance_ratio_)
transform = pca.transform(vectors[0:number_of_elements])
print("PCA done ...")

# Representation with Bokeh
source = ColumnDataSource(data=dict(
    x=transform[:, 0],
    y=transform[:, 1],
    mots=words[0:number_of_elements],
    color=['#053061' for i in range(0,number_of_elements)],
))
sourceTSNE = ColumnDataSource(data=dict(
    x=[0 for i in range(0,number_of_elements)],
    y=[0 for i in range(0,number_of_elements)],
    mots=words[0:number_of_elements],
    color=['#053061' for i in range(0,number_of_elements)],
))
sour = ColumnDataSource(data=dict(
    label=['one','two','three'],
    edges=[[2,3],[1],[3]],
    values=[[0.5,0.2],[0.3],[0.78]],
    index=[0,1,2],
    color=['red', 'blue', 'red']
))
print('source done ...')
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

p = figure(plot_height=400, tools=TOOLS, output_backend="webgl", active_scroll='wheel_zoom')
p_circle = p.circle('x', 'y', size=5, source=source, color='#053061', fill_alpha=0.5)
p_circle.selection_glyph = selected_circle
p.add_layout(color_bar_p, 'left')
p.axis.visible = False
tab1 = Panel(child=p, title="PCA")

p2 = figure(plot_height=400, x_range=(-50, 50), y_range=(-50, 50), tools=TOOLS, output_backend="webgl", active_scroll='wheel_zoom')
p2_circle = p2.circle('x', 'y', size=5, source=sourceTSNE, color='#053061', fill_alpha=0.5)
p2_circle.selection_glyph = selected_circle
p2.add_layout(color_bar_p2, 'left')
p2.axis.visible = False
p3 = Network(label="label", edges="edges", values="values", color="color", data_source=sour, width=650, height=390)
tsnePerplexity = Slider(start=5, end=100, value=30, step=1, width=120, title="Perplexity")
tsneLearning = Slider(start=10, end=1000, value=200, step=1, width=120, title="Learning Rate")
tsneIteration = Slider(start=300, end=5000, value=500, step=50, width=120, title="Iterations")
tsneSpeed = Slider(start=10, end=100, value=70, step=1, width=120, title="Speed")
tsneApply = Button(label='Apply', button_type='success', width=80)
pauseB = Button(label='Pause', button_type='success', width=60)
startB = Button(label='Start', button_type='success', width=60)
stopB = Button(label='Stop', button_type='success', width=60)
tsneLoading = Div()
iterationCount = Div()
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
p3.css_classes = ["blackBorder"]
analogy = column(word1, word2, word3, row(calculateAnalogy, Spacer(width=20), equals))
tsneLayout = layout([
    [p2, p3],
    [tsnePerplexity, Spacer(width=20), tsneLearning, Spacer(width=20), tsneIteration, Spacer(width=20), tsneApply, Spacer(width=20), widgetbox(tsneLoading, width=30), Spacer(width=100), widgetbox(searchBox, searchButton)],
    [tsneSpeed, Spacer(width=20), pauseB, Spacer(width=10), startB, Spacer(width=10), stopB, Spacer(width=20), widgetbox(iterationCount, width=60)],
])
tab2 = Panel(child=tsneLayout, title="t-SNE")
renderer = p2.select(dict(type=GlyphRenderer))
ds = renderer[0].data_source

tab3 = Panel(child=analogy, title="Analogy")
tabs = Tabs(tabs=[tab1, tab2, tab3])
'''
#testing analogies
vec = [x1-x2+x3 for x1,x2,x3 in zip(vectors[563],vectors[313],vectors[513])]
analog = [cosSim(np.asarray([vec]), np.asarray([b]) )[0][0] for b in vectors]
analogyList = list(zip([i for i in range(0,len(vectors)-1)], analog))
sortedAna = sorted(analogyList,key=lambda l:l[1], reverse=True)
print(words[563], words[313], words[513])
for i in range(1,10):
    print(words[sortedAna[i][0]])
'''
# Neighbors
number_of_neighbors = 10
#allSimilarity = cosSim(vectors)

'''
def neighbors(source=source, div=resultText, vectors=vectors[0:number_of_elements], cosSim=cosSim):
    word = source.data['mots'][cb_data.source.selected.indices[0]]
    wordIndex = cb_data.source.selected.indices[0]
    print(cb_data)
    v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors]
    #similarityList = [[i,cosSim(vectors[wordIndex],vectors[i])] for i in range(0,len(vectors)) if i!=wordIndex]
    similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
    sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=True)
    similarities = "10 neighbors of " + word + " : <br/>"
    for i in range(1, 10):
        cb_data.source.selected.indices[i] = sortedSim[i][0]
        similarities = similarities + \
            words[sortedSim[i][0]] + " : " + \
            "{0:.2f}".format(sortedSim[i][1]) + "<br/>"
    div.text = similarities

# Selection Event
taptool = p.select(type=TapTool)
taptool.callback = CustomJS.from_py_func(neighbors)

taptool_TSNE = p2.select(type=TapTool)
taptool_TSNE.callback = CustomJS.from_py_func(neighbors)
'''
def generateColor():
    color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
    return color
def handlerTSNE(attr, old, new, vectors=vectors[0:number_of_elements]):
    if (len(new.indices) != 0) and (handlerTSNE.update == False) :
        handlerTSNE.update = True
        wordIndex = new.indices[0]
        v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors]
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        p2_circle.data_source.add(data=v,name='color')
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=True)
        l = [new.indices[0]]
        sour.data['label'] = [words[wordIndex]]
        sour.data['edges'] = [[]]
        sour.data['values'] = [[]]
        sour.data['index'] = [wordIndex]
        sour.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors):
            l.append(sortedSim[i][0])
            sour.data['label'].append(words[sortedSim[i][0]])
            sour.data['edges'].append([1])
            sour.data['values'].append([sortedSim[i][1]])
            sour.data['index'].append(sortedSim[i][0])
            sour.data['color'].append(color)
        p2_circle.data_source.selected.indices = l
        p2_circle.data_source.trigger('selected',None,p2_circle.data_source.selected)
        # print(model.wv.most_similar(words[wordIndex]))
        sour.trigger('data', None, sour)
        handlerTSNE.update = False

def handler(attr, old, new, vectors=vectors[0:number_of_elements]):
    if (len(new.indices) != 0) and (handler.update == False) :
        handler.update = True
        wordIndex = new.indices[0]
        v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors]
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        p_circle.data_source.add(data=v,name='color')
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=True)
        sortedSim = np.asarray([list(i) for i in sortedSim])
        l = [new.indices[0]]
        for i in range(1, number_of_neighbors):
            l.append(int(sortedSim[i][0]))
        p_circle.data_source.selected.indices = l
        p_circle.data_source.trigger('selected',None,p_circle.data_source.selected)
        # print(model.wv.most_similar(words[wordIndex]))
        handler.update = False
        
handlerTSNE.update = False
handler.update = False

# T-SNE representation Function
def tsneProcess(source=ds, vectors=vectors):
    global positions
    global X_iter
    positions = []
    tsneLoading.css_classes = ["loader"]
    TSNE_transform = TSNE(n_components=2, n_iter=tsneIteration.value, perplexity=tsnePerplexity.value, learning_rate=tsneLearning.value).fit_transform(vectors[0:number_of_elements])
    print("TSNE done ...")
    X_iter = np.dstack(position.reshape(-1, 2) for position in positions)
    print("X_iter done ...")
    ds.data['x'] = TSNE_transform[:, 0]
    ds.data['y'] = TSNE_transform[:, 1]
    ds.trigger('data',ds.data,ds.data)
    tsneLoading.css_classes = []

#T-SNE Animation Function
iteration = 0
def tsne_animation(length=number_of_elements, div=iterationCount):
    global ds
    global iteration
    global X_iter
    if iteration < X_iter.shape[2]:
        for i in range(0,length):
            ds.data['x'][i] = X_iter[i][0][iteration]
            ds.data['y'][i] = X_iter[i][1][iteration]
        
        ds.trigger('data',ds.data,ds.data)
        iteration += 1
        div.text = "Iteration N&#176 : " + str(iteration)
    else:
        stopAnimation()

numberElement = TextInput(width=200, placeholder="Nombre d'iterations")
changeNumberB = Button(label='Change')

# output_file("Vector representation.html", title="Example of vector representation")
l2 = layout([
    [numberElement],
], sizing_mode='scale_width')
l = layout([
  [tabs],
], sizing_mode='scale_width')
curdoc().add_root(l)
curdoc().title = "Embedding"

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
    global iteration
    if (animation != None):
        curdoc().remove_periodic_callback(animation)
        animation = None
        iteration = 0
        startB.label = "Start"
def changeSpeed(attr, old, new):
    global animation
    if (animation != None):
        curdoc().remove_periodic_callback(animation)
        animation = None
        animation = curdoc().add_periodic_callback(tsne_animation,int(110-new))
def tabChange(attr, old, new):
    if(new == 1) and (tabChange.first == False):
        tabChange.first = True
        tsneProcess()
tabChange.first = False

def addNeighborNodes(index, vectors=vectors[0:number_of_elements]):
    wordIndex = sour.data['index'][index]
    v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors]
    similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
    sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=True)
    color = generateColor()
    for i in range(1, number_of_neighbors):
        #print(sortedSim[i][0], sour.data['index'], sour.data['edges'][index])
        if (sortedSim[i][0] in sour.data['index']) and (sour.data['index'].index(sortedSim[i][0])+1 not in sour.data['edges'][index]):
            indice = sour.data['index'].index(sortedSim[i][0])
            sour.data['edges'][indice].append(index+1)
            sour.data['values'][indice].append(sortedSim[i][1])
        elif (sortedSim[i][0] not in sour.data['index']):
            sour.data['label'].append(words[sortedSim[i][0]])
            sour.data['edges'].append([index+1])
            sour.data['values'].append([sortedSim[i][1]])
            sour.data['index'].append(sortedSim[i][0])
            sour.data['color'].append(color)
    sour.trigger('data', None, sour)

def selectNode(attr, old, new):
    addNeighborNodes(new)

def calcAnalogy():
    word = model.wv.most_similar(positive=[word1.value,word3.value], negative=[word2.value])
    # print(word)
    equals.text = "<b> <center>"+word[0][0]+"</center> </b>"

def searchWord(vectors=vectors[0:number_of_elements], words=words[0:number_of_elements]):
    word = searchBox.value
    if (word in words):
        handlerTSNE.update = True
        wordIndex = words.index(word)
        v = [cosSim( np.asarray([vectors[wordIndex]]), np.asarray([b]) )[0][0] for b in vectors]
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        p2_circle.data_source.add(data=v,name='color')
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=True)
        l = [wordIndex]
        sour.data['label'] = [words[wordIndex]]
        sour.data['edges'] = [[]]
        sour.data['values'] = [[]]
        sour.data['index'] = [wordIndex]
        sour.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors):
            l.append(sortedSim[i][0])
            sour.data['label'].append(words[sortedSim[i][0]])
            sour.data['edges'].append([1])
            sour.data['values'].append([sortedSim[i][1]])
            sour.data['index'].append(sortedSim[i][0])
            sour.data['color'].append(color)
        p2_circle.data_source.selected.indices = l
        p2_circle.data_source.trigger('selected',None,p2_circle.data_source.selected)
        # print(model.wv.most_similar(words[wordIndex]))
        sour.trigger('data', None, sour)
        handlerTSNE.update = False
    else:
        handlerTSNE.update = True
        newWordVector = model[searchBox.value]
        v = [cosSim( np.asarray([newWordVector]), np.asarray([b]) )[0][0] for b in vectors]
        similarityList = list(zip([i for i in range(0, len(vectors)-1)], v))
        p2_circle.data_source.add(data=v,name='color')
        sortedSim = sorted(similarityList, key=lambda l:l[1], reverse=True)
        l = [sortedSim[0][0]]
        sour.data['label'] = [words[sortedSim[0][0]]]
        sour.data['edges'] = [[]]
        sour.data['values'] = [[]]
        sour.data['index'] = [sortedSim[0][0]]
        sour.data['color'] = [generateColor()]
        color = generateColor()
        for i in range(1, number_of_neighbors):
            l.append(sortedSim[i][0])
            sour.data['label'].append(words[sortedSim[i][0]])
            sour.data['edges'].append([1])
            sour.data['values'].append([sortedSim[i][1]])
            sour.data['index'].append(sortedSim[i][0])
            sour.data['color'].append(color)
        p2_circle.data_source.selected.indices = l
        p2_circle.data_source.trigger('selected',None,p2_circle.data_source.selected)
        # print(model.wv.most_similar(words[wordIndex]))
        sour.trigger('data', None, sour)
        handlerTSNE.update = False

pauseB.on_click(destroyAnimation)
startB.on_click(animate)
stopB.on_click(stopAnimation)
tsneSpeed.on_change('value', changeSpeed)
tabs.on_change('active', tabChange)
tsneApply.on_click(tsneProcess)
p_circle.data_source.on_change('selected',handler)
p2_circle.data_source.on_change('selected',handlerTSNE)
p3.on_change('selected',selectNode)
calculateAnalogy.on_click(calcAnalogy)
searchButton.on_click(searchWord)
