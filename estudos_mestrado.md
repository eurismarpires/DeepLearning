## 31/07/2017

* [Estudo do código - music-rnn](https://github.com/jamong/music-rnn)
* [Quando e Como usar TimeDistributedDense](https://github.com/fchollet/keras/issues/1029)

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
