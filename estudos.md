## 25/03/2018
Teste GPU

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
Verificar GPU
```console
$ nvcc --version
$ nvidia-smi
```
[Mais informações](https://medium.com/@srini.x/beginners-guide-to-setting-up-a-tensorflow-deep-learning-environment-in-ubuntu-16-04-2-e31164e3d638)

## 23/03/2018
[midi melodies](http://en.midimelody.ru/category/midi-melodies)
[midis](http://antaresmidis.com.br/heitor_villa_lobos.html)

## 14/03/2018

[Download all files in a path on Jupyter notebook server](https://stackoverflow.com/questions/43042793/download-all-files-in-a-path-on-jupyter-notebook-server)

## 10/03/2018

DeepJazz Coursera instalar módulo [qa](https://github.com/bickfordb/qa)

instalar também grammar 

      pip install anovelmous_grammar

## 09/03/2018
[Python online sem precisar de instalar - repl.it](https://repl.it/repls/MelodicWhoppingPhases)

## 07/03/2018 

[Deep Learning for Music (DL4M)](https://github.com/ybayle/awesome-deep-learning-music)

[Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)

## 05/03/2018 

#### Configuração music21 

``` 
us = environment.UserSettings()
us.getSettingsPath()
'/home/usuario/.music21rc'
us['musicxmlPath']='/usr/bin/musescore'
us['musescoreDirectPNGPath']='/usr/bin/musescore'
```




## 26/02/2018
[Vários artigos - Music Generation With DL](https://github.com/umbrellabeach/music-generation-with-DL)

[Excelentes materiais de estudos de Machine Learning](https://matheusfacure.github.io/tutoriais/)

[Tutorial GAN](https://github.com/matheusfacure/Tutoriais-de-AM/blob/master/Redes%20Neurais%20Artificiais/GAN_MNIST.ipynb)
## 20/02/2018
[Tutorial Keras](http://cv-tricks.com/tensorflow-tutorial/keras/)
## 19/02/2018
[Ótimo artigo sobre Multi-Label Classification - Guide To Multi-Class Multi-Label Classification With Neural Networks In Python](https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/)
## 17/02/2018
[Resolver o problema do Kernel não aparecer no jupyter notebook](https://github.com/jupyter/jupyter/issues/245)

      source activate myenv
      python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
      source activate other-env
      python -m ipykernel install --user --name other-env --display-name "Python (other-env)"

## 07/02/2018
[Text Generation using Generative Adversarial Networks (GAN) - Core challenges](https://www.linkedin.com/pulse/text-generation-using-generative-adversarial-networks-p-n/)

## 15/01/2018

[Filtro convolucional](http://setosa.io/ev/image-kernels/)

## 10/01/2018 

Depois de ter instalado o Cuda 9 (errado porque o tensorflow ainda nao funciona com essa versao). Deu muito trabalho para conseguir remover esta e instalar o Cuda 8. O que funcionou foi a dica desse [site](http://dhaneshr.net/2016/11/09/upgrading-to-cuda-8-0-on-ubuntu-16-04/).

## 04/01/2018

NETWORK FOR SYMBOLIC-DOMAIN MUSIC GENERATION](https://arxiv.org/pdf/1703.10847.pdf) [Código](https://github.com/RichardYang40148/MidiNet)


## 03/01/2018
Para rodar o projeto  https://github.com/llSourcell/Music_Generator_Demo além do tensorflow deve instalar tdm e o python-midi


      $ pip install https://pypi.python.org/packages/71/3c/341b4fa23cb3abc335207dba057c790f3bb329f6757e1fcd5d347bcf8308/tqdm-4.19.5-py2.py3-none-any.whl#md5=99fe93af82488226c1afadfaec0e1498
      
      $ # testei no python3 e deu certo
      $ git clone https://github.com/louisabraham/python3-midi.git
      $ #dentro do diretório clonado rodar o comando
      $ python setup.py install




## 30/12/2017
# saber o caminho onde o python está instalado
      $ import os
      $ import sys
      $ os.path.dirname(sys.executable)

## 24/12/2017
* [Gan com Tensorflow](https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/)
## 13/12/2017
* [MIDI NET](https://arxiv.org/abs/1703.10847)
* [MUSEGAN: DEMONSTRATION OF A CONVOLUTIONAL GAN BASED
MODEL FOR GENERATING MULTI-TRACK PIANO-ROLLS](https://ismir2017.smcnus.org/lbds/Dong2017.pdf)
* [Tutorial Pretty-Midi](https://github.com/craffel/pretty-midi/blob/master/Tutorial.ipynb)

## 05/12/2017
* [The innovative approach with Generative Adversarial Networks (GAN) is to replace the human evaluator
with an additional neural network](https://openreview.net/pdf?id=SyBPtQfAZ)


## 28/11/2017
* Para fazer rodar o código https://github.com/olofmogren/c-rnn-gan só
         com o python 2.7 e tensorflow 0.12.1 e python-midi
         
         
        $ conda create -n rnn-gan python=2.7
        $ source activate rnn_gan
        $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
        $ pip install --upgrade $TF_BINARY_URL
        $ pip install python-midi
        $ python rnn_gan.py --datadir datadir --traindir traindir -testdir test
        

## 22/10/2017
[Leitura do artigo Algorithmic Composition of Melodies with Deep
Recurrent Neural Networks](https://arxiv.org/pdf/1606.07251.pdf)
## 16/10/2017
[Variational Autoencoder: Intuition and Implementation in Keras](https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/)
## 14/10/2017
* [LSTM in numpy](http://chris-chris.ai/2017/10/10/LSTM-LayerNorm-breakdown-eng/)
* [Livro pense em Python](https://penseallen.github.io/PensePython2e/)
## 12/10/2017
* [Redes adversarias generativas em TensorFlow](https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/)
## 08/10/2017
* [Vanilla LSTM with numpy](http://blog.varunajayasiri.com/numpy_lstm.html)
* [Generating Random Vectors from the Multivariate Normal
Distribution](ftp://ftp.dca.fee.unicamp.br/pub/docs/vonzuben/ia013_2s09/material_de_apoio/gen_rand_multivar.pdf)
* [Normal Distribution](https://www.mathsisfun.com/data/standard-normal-distribution.html)

## 28/09/2018 Estudos GAN Udemy
## 22/09/2017
* (Effective TensorFlow)[Machine Learning Top 10 Articles For the Past Month (v.Sep 2017)](https://medium.mybridge.co/machine-learning-top-10-articles-for-the-past-month-v-sep-2017-c68f4b0b5e72)
* [](https://github.com/vahidk/EffectiveTensorflow)
## 21/09/2017
* [data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks#scikit-learn)
## 18/09/2017
* [Creating A Language Translation Model Using Sequence To Sequence Learning Approach](https://chunml.github.io/ChunML.github.io/project/Sequence-To-Sequence/)
* [Matemática para Deep Learning](http://aima.cs.berkeley.edu/newchapa.pdf)
* [Algebra Linear Livros](http://www.sedis.ufrn.br/bibliotecadigital/site/pdf/TICS/Alg_I_ECT_Livro_Z_WEB.pdf)(http://www.uesb.br/professor/flaulles/download/softwares/IntAlgebraLinear.pdf)
## 13/09/2017
* [Understanding LSTM in Tensorflow(MNIST dataset)](https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)
## 12/09/2017
* [Generative Adversarial Networks for Beginners](https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners) [Código](https://github.com/jonbruner/generative-adversarial-networks/blob/master/gan-notebook.ipynb)
## 11/09/2017
* [Backprop is very simple. Who made it Complicated?](https://github.com/Prakashvanapalli/TensorFlow/blob/master/Blogposts/Backpropogation_with_Images.ipynb)
* [Writing Mathematic Fomulars in Markdown](http://csrgxtu.github.io/2015/03/20/Writing-Mathematic-Fomulars-in-Markdown/)
## 07/09/2017
* [Deep Learning Techniques for Music Generation - A Survey](https://arxiv.org/abs/1709.01620)
## 05/09/2017
* [Embedding and Tokenizer in Keras](http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/)
## 04/09/2017
* [Deep Learning Mindmap / Cheatsheet](https://github.com/dformoso/deeplearning-mindmap)
* [Encoder-Decoder Long Short-Term Memory Networks](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)
* [Music Generation Using Rnn](http://qihqi.github.io/machine/learning/music-generation-using-rnn/)
* [Delving deep into Generative Adversarial Networks (GANs)](https://github.com/GKalliatakis/Delving-deep-into-GANs)
* [https://pythonprogramming.net/machine-learning-tutorial-python-introduction/](https://pythonprogramming.net/machine-learning-tutorial-python-introduction/)
* [How to Use an Encoder Decoder Lstm to Echo Sequences of Random Integers](https://machinelearningmastery.com/how-to-use-an-encoder-decoder-lstm-to-echo-sequences-of-random-integers/)
* [Tutorials Keras Recurrent](http://vict0rsch.github.io/tutorials/keras/recurrent/)
## 02/09/2017
* [Python Cheat Sheet Basic](https://www.dataquest.io/blog/images/cheat-sheets/python-cheat-sheet-basic.pdf)
* [Python Cheat Sheet Intermediate](https://www.dataquest.io/blog/images/cheat-sheets/python-cheat-sheet-intermediate.pdf)
* [Pandas Cheat Sheet](https://www.dataquest.io/blog/images/cheat-sheets/pandas-cheat-sheet.pdf)
* [Pandas Tutorial Python 2](https://www.dataquest.io/blog/pandas-tutorial-python-2/)
* [Naive Bayes Tutorial](https://www.dataquest.io/blog/naive-bayes-tutorial/)
## 01/09/2017
* [Tutorial git](https://gist.github.com/leocomelli/2545add34e4fec21ec16)
## 29/08/2017
* [Vários links de projetos de música usando deep learning](https://handong1587.github.io/deep_learning/2015/10/09/fun-with-deep-learning.html#music)
* Leitura do artigo [Song From PI A Musically Plausible Network for Pop Music Generation](http://www.cs.toronto.edu/songfrompi/)
## 28/08/2017
* [Deep Learning for Music - https://amundtveit.com/2016/11/22/deep-learning-for-music/](https://amundtveit.com/2016/11/22/deep-learning-for-music/)
## 26/08/2017
* [Programming instrumental music from scratch - Ensina o funcionamento da biblioteca Python MIDI](https://www.r-bloggers.com/programming-instrumental-music-from-scratch/) 
* [Music Note Symbols](http://www.musiclearningworkshop.com/music-note-symbols/)
## 25/08/2017
* [Deep Learning From Scratch: Theory and Implementation](http://www.deepideas.net/deep-learning-from-scratch-theory-and-implementation/)
* [DEEP LEARNING METALLICA WITH RECURRENT NEURAL NETWORKS](http://www.mattmoocar.me/blog/tabPlayer/)
* [Isolated guitar transcription using a deep
belief network](https://peerj.com/articles/cs-109.pdf)
## 21/08/2017
* [Modeling Musical Context with Word2vec](http://dorienherremans.com/word2vec)
* [Projetos](http://dorienherremans.com/software)
* [Bibliografia Geração de Musicas](http://dorienherremans.com/biblio)
## 19/08/2017 
* Configurar o Music21 para usar o Musescore no Windows:
        
        $  us = environment.UserSettings()
        $  us['musicxmlPath']="C:\\Program Files (x86)\\MuseScore 2\\bin\\MuseScore.exe"
        $  us["musescoreDirectPNGPath"]="C:\\Program Files (x86)\\MuseScore 2\\bin\\MuseScore.exe"
## 16/08/2017
* [https://henri.io/posts/chord-classification-using-neural-networks.html](https://henri.io/posts/chord-classification-using-neural-networks.html)
* [https://www.thisismetis.com/made-at-metis](https://www.thisismetis.com/made-at-metis)
* [Geração de música JohaNN após treinar a rede](https://github.com/eurismarpires/exercicios_python/blob/master/RunJohaNN.ipynb)

## 14/08/2017 - 15/08/2017
* [The 5 Step Life-Cycle for Long Short-Term Memory Models in Keras](http://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)
* [Generative Adversarial Networks Part 2 - Implementation with Keras 2.0](http://www.rricard.me/machine/learning/generative/adversarial/networks/keras/tensorflow/2017/04/05/gans-part2.html)
* [https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/](https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/)
## 13/08/2017
* [Vídeo aula modelos generativos - Stanford](https://www.youtube.com/watch?v=5WoItGTWV54)
## 11/12/2017 - 12/08/2017
* Estudo do [código](https://github.com/naoyak/JohaNN)
* Exemplo com [music21 mid_online_music21](https://github.com/eurismarpires/exercicios_python/blob/master/mid_online_music21.ipynb)

![](https://github.com/eurismarpires/exercicios_python/blob/master/mapeamento_johaNN.png)

## 10/08/2017
* Estudos da biblioteca [Music21 e Pyknon](https://github.com/eurismarpires/exercicios_python/blob/master/MidiMusic21Pyknon.ipynb) baseados no [tutorial](https://blog.ouseful.info/2016/09/13/making-music-and-embedding-sounds-in-jupyter-notebooks/)
* Consegui fazer funcionar o áudio no Musescore instalando o gawk:

        $ sudo apt-get install gawk
                
## 09/08/2017
* Leitura [5 Genius Python Deep Learning Libraries](https://elitedatascience.com/python-deep-learning-libraries)
## 08/08/2017
* Estudando como transformar MIDI em rolo de Piano. [midiToNoteStateMatrix](https://github.com/llSourcell/Music_Generator_Demo). Não quer funcionar para arquivos MIDI para violão. Mas para os datasets do projeto de exemplo funciona. É interessante a representaço onde a metade da matriz contém os bits um's para indicar quando a nota foi tocada.
* Uma forma interessante de representar um [rolo de piano](https://stackoverflow.com/questions/44661688/converting-piano-roll-to-midi-in-music21)
* Visualização de rolo de piano em [music21 ](http://web.mit.edu/music21/doc/usersGuide/usersGuide_22_graphing.html)

## 07/08/2017 
* Estudo da biblioteca [pretty_midi](https://github.com/craffel/pretty-midi) Paper [INTUITIVE ANALYSIS, CREATION AND MANIPULATION OF MIDI
DATA WITH pretty_midi](http://colinraffel.com/publications/ismir2014intuitive.pdf) 
## 04/08/2017 - 06/08/2017
* [Estudo GAN Deep Learning Brasil no Jupyter Notebook](https://github.com/deeplearningbrasil/gan-generative-adversarial-networks-parte-I)

## 03/08/2017
* [Estudo GAN Deep Learning Brasil](https://github.com/deeplearningbrasil/gan-generative-adversarial-networks-parte-I) [1](https://github.com/eurismarpires/exercicios_python/blob/master/gan.ipynb) [2](https://github.com/eurismarpires/exercicios_python/blob/master/gan_nova.ipynb)
* [TensorFlow Tutorial](https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf)
* [Variable sharing in Tensorflow](https://jasdeep06.github.io/posts/variable-sharing-in-tensorflow/)
## 02/08/2017
* Estudo do código [GAN-Sandbox](https://github.com/wayaai/GAN-Sandbox) em Keras/Tensorflow
## 01/08/2017 
* Leitura do artigo [Text Generation using Generative Adversarial Training](https://web.stanford.edu/class/cs224n/reports/2761133.pdf)
* Estudo do código [TextGAN](https://github.com/AustinStoneProjects/TextGAN)
## 31/07/2017

* [Estudo do código - music-rnn](https://github.com/jamong/music-rnn)
* [Quando e Como usar TimeDistributedDense](https://github.com/fchollet/keras/issues/1029)
* [TimeDistributed Keras](https://keras.io/layers/wrappers/)

## 30/07/2017
* Pesquisa sobre as bibliotecas [music21](http://web.mit.edu/music21/), [unroll](https://github.com/Zulko/unroll) e [lilypond](http://lilypond.org/doc/v2.18/Documentation/learning/tutorial)
* [Leitura sobre transposição para ser utilizada em DeepLearning](http://nickkellyresearch.com/python-script-transpose-midi-files-c-minor/)
* [Leitura - Metis Final Project Music Composition](http://blog.naoya.io/metis-final-project-music-composition-with-lstms/)
* [Leitura parcial do projeto musicNet](https://cm-gitlab.stanford.edu/tsob/musicNet/) 

## 29/07/2017
* Estudo do código [sequence_gan](https://github.com/ofirnachum/sequence_gan)
* Estudo do código [Modelling-and-Generating-Sequences-of-Polyphonic-Music-With-RNN-RBM](https://github.com/SiddharthTiwari/Modelling-and-Generating-Sequences-of-Polyphonic-Music-With-RNN-RBM)
* Estudo do código[Music Generator Demo](https://github.com/llSourcell/Music_Generator_Demo). É necessário instalar os pacotes no python2:        

        $ pip install Numpy==1.11.0 
        $ pip install tensorflow==0.12.0 
        $ pip install pandas==0.19.2 
        $ pip install msgpack-python==0.4.8 
        $ pip install glob2==0.5 
        $ pip install tqdm==4.11.2 
        $ pip install python-midi==0.2.4
* Estudo do código [How-to-Generate-Music-Demo](https://github.com/llSourcell/How-to-Generate-Music-Demo) Para funcionar tem que remover a parte fixa mode='major' do arquivo preprocess.py [Video](https://www.youtube.com/watch?v=4DMm5Lhey1U). Se gerar com mais de 25 épocas dá erro o qual pode ser corrigido conforme essa [dica](https://github.com/llSourcell/How-to-Generate-Music-Demo/issues/4). É o mesmo código de [DeepJazz](https://github.com/jisungk/deepjazz)

* Leitura [Aprendizado por reforço](http://edirlei.3dgb.com.br/aulas/ia_2011_2/IA_Aula_19_Aprendizado_Por_Reforco.pdf)
* Estudo [Set em Python](https://www.programiz.com/python-programming/set)
## 28/07/2017
* Leitura do artigo [Deep Learning for Music](https://arxiv.org/pdf/1705.10843.pdf)
* Leitura do artigo [Objective-Reinforced Generative Adversarial
Networks (ORGAN) for Sequence Generation Models](https://arxiv.org/pdf/1606.04930.pdf) [Código](https://github.com/gablg1/ORGAN)
        
        $ conda create -c rdkit -n organ rdkit python=2.7
        $ source activate organ
        $ pip install tqdm

* Estudo de [função de verossimilhança](http://www.leg.ufpr.br/~paulojus/embrapa/Rembrapa/Rembrapase16.html) [Código](https://github.com/eurismarpires/DeepLearning/blob/master/fun%C3%A7%C3%A3o%2Bde%2Bverossimilhan%C3%A7a.ipynb)
* [Estudo R](http://www.leg.ufpr.br/~paulojus/embrapa/Rembrapa/Rembrapa.html#Rembrapase6.html)
* Debug [GAN](https://github.com/deeplearningbrasil/gan-generative-adversarial-networks-parte-I)
* Debug [SeqGAN](https://github.com/LantaoYu/SeqGAN)
* Estudo Tensorflow [Getting Started With TensorFlow](https://www.tensorflow.org/get_started/get_started) e [MNIST For ML Beginners](https://www.tensorflow.org/get_started/mnist/beginners)
## 27/07/2017
* Leitura do projeto final Deep-Rock https://cs224d.stanford.edu/reports/Ilan.pdf
## 26/07/2017
* Teste do Melody_RNN do projeto magenta: https://github.com/tensorflow/magenta/blob/master/magenta/models/melody_rnn/README.md Para isso foi criado as variáveis de ambiente abaixo e depois seguiu-se o tutorial.
####        
        $ docker pull tensorflow/magenta #download o docker do projeto magenta
        $ docker run -it tensorflow/magenta bash
        $ export BUNDLE_PATH=/magenta-models/basic_rnn.mag
        $ export CONFIG=basic_rnn
        $ melody_rnn_generate \
        --config=${CONFIG} \
        --bundle_file=${BUNDLE_PATH} \
        --output_dir=/tmp/melody_rnn/generated \
        --num_outputs=10 \
        --num_steps=128 \
        --primer_melody="[60]"

* Leitura https://magenta.tensorflow.org/2016/07/15/lookback-rnn-attention-rnn/
* Links docker http://stefanteixeira.com.br/2015/03/17/comandos-essenciais-docker-monitoramento-containers/ https://woliveiras.com.br/posts/imagem-docker-ou-um-container-docker/ Obs. Para sair do container sem matar a sessão digitar Ctrl+P+Q. Para entrar novamente digirar docker attach + containerId

#### Deep Jammer
* blog https://medium.com/towards-data-science/can-a-deep-neural-network-compose-music-f89b6ba4978d
* poster https://www.justinsvegliato.com/pdf/deep-jammer-poster.pdf
* artigo https://www.justinsvegliato.com/pdf/deep-jammer-report.pdf
* código https://github.com/justinsvegliato/deep-jammer

#### Biaxial Recurrent Neural Network for Music Composition
* Código https://github.com/hexahedria/biaxial-rnn-music-composition
* Blog http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/
* Artigo http://www.hexahedria.com/files/2017generatingpolyphonic.pdf
* Leituras relacionadas:[RNN Music Composer](http://imatv.me/classes/Psych186BReportRNNMusic.pdf), [Creativity in Machine Learning](https://arxiv.org/pdf/1601.03642.pdf), [Composing Classical Music with Recurrent Neural Network](https://deepdarklearning.wordpress.com/2016/12/15/composing-classical-music-with-recurrent-neural-network/)

## Links Importantes
* https://libraries.io/
* https://magenta.tensorflow.org/2016/12/16/nips-demo/
* [programiz.com](https://www.programiz.com/)

## Comandos Úteis

#### Instalar Magenta
    $ sudo apt-get install libasound2
    $ sudo apt-get install libasound2-dev
    $ pip install python-rtmidi
    $ pip install magenta 
    
#### Conda
    $ conda create -n deep-improvisation python=2.7
    $ conda-env remove -n deep-improvisation
    $ source activate deep-improvisation

#### Coisas para aprender
 * Maxima verossimilhança
 * GAN
 #### Geração de Músicas
 | Titulo | Description |Site|Biblioteca|
| --- | --- |---|---|
|LSTM network for algorithmic music composition|Este é o repositório de código para o trabalho de tese de mestrado, em Composição de música algorítmica utilizando redes neuronais recorrentes (RNN)|[https://github.com/bernatfp/LSTM-Composer](https://github.com/bernatfp/LSTM-Composer)|Keras e [mido](https://github.com/olemb/mido)|
|LSTM-RNN-Melody-Composer| Esta é a implementação de uma Rede Neural Recorrente LSTM que compõe uma melodia para uma determinada sequência de acordes.|[https://github.com/konstilackner/LSTM-RNN-Melody-Composer](https://github.com/konstilackner/LSTM-RNN-Melody-Composer)|Keras e Mido|
|JohaNN|2-layer LSTM music generator trained on Bach Cello Suites|[https://github.com/naoyak/JohaNN](https://github.com/naoyak/JohaNN)|Keras e [music21](http://web.mit.edu/music21/doc/about/what.html)|
|SCHUMANN: RECURRENT NEURAL NETWORKS COMPOSING MUSIC||[http://inspiratron.org/blog/2017/01/01/schumann-rnn-composing-music/](http://inspiratron.org/blog/2017/01/01/schumann-rnn-composing-music/)||
|Music Language Modeling with Recurrent Neural Networks||(https://github.com/yoavz/music_rnn) [http://yoavz.com/music_rnn/](http://yoavz.com/music_rnn/)||
|StyleNet|Neural Translation of Musical Style|[Dissertação](http://imanmalik.com/documents/dissertation.pdf) [Blog](http://imanmalik.com/cs/2017/06/05/neural-style.html) [Código](https://github.com/imalikshake/StyleNet/)|Tensorflow,mido,pretty_midi,fluidsynth|
|Metis Final Project: Music Composition with LSTMs|a recurrent neural network utilizing Long Short-Term Memory nodes (LSTMs) to learn patterns in the Six Cello Suites by J.S. Bach|[Blog](http://blog.naoya.io/metis-final-project-music-composition-with-lstms/)||
|neuralnetmusic|The myparser.py module turns musicxml files into training data|https://github.com/fephsun/neuralnetmusic|Keras|
|Music Generator Demo - Siraj Raval|Overview Use TensorFlow to generate short sequences of music with a Restricted Boltzmann Machine. This is the code for Generate Music in TensorFlow on YouTube.|https://github.com/llSourcell/Music_Generator_Demo|Python midi e Tensorflow|
|Music Generator Demo - Siraj Raval|This is the code for this video on Youtube by Siraj Raval as part of the the Udacity Deep Learning Nanodegree. It uses Keras & Theano, two deep learning libraries, to generate jazz music. Specifically, it builds a two-layer LSTM, learning from the given MIDI file.|https://github.com/llSourcell/How-to-Generate-Music-Demo|Keras, Theano, Music21|
|LSTM-RNN-Melody-Composer|Esta é a implementação de uma Rede Neural Recorrente LSTM que compõe uma melodia para uma determinada sequência de acordes|https://github.com/konstilackner/LSTM-RNN-Melody-Composer|mido e Keras
|Implementation of C-RNN-GAN.|Aplicado a geração de música|https://github.com/olofmogren/c-rnn-gan|Tensorflow










