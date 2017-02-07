## Instalação 

## Principais comandos


### docker sem sudo https://coderwall.com/p/4zeaqq/run-docker-commands-without-sudo

### docker ps [args]
Lista os containers em execução

> *Nota:* a maior parte dos comandos aceitam argumentros, execute `docker ps --help` para ver uma listagem das opções.

### docker stats [container]
Informa o tempo de execução e detalhes do consumo de recursos da maquina host

### docker images
Lista as imagens disponiveis na maquina

### docker rm [container]
Remove um container que não estava mais em execução.

### docker rmi [imagem]
Remove uma imagem salva. 

### docker run [options] [image] [command] [args]
Coloca um container em execução, por exemplo:
```
docker run -it ubuntu bash
```
se tudo der certo deve ser visualizado um terminal, de dentro do container em execução
```
root@13d2fb2f08e1:/#
```
#### Opções mais utilizadas para o comando `docker run`
 - `-i` permite interagir com o container
 - `-t` associa o seu terminal ao terminal do container
 - `-it` é apenas uma forma reduzida de escrever `-i -t`
 - `--name algum-nome` permite atribuir um nome ao container em execução
 - `-p 8080:80` mapeia a porta 80 do container para a porta 8080 do host
 - `-d` executa o container em background
 - `-v /pasta/host:/pasta/container` cria um volume '/pasta/container' dentro do container com o conteudo da pasta '/pasta/host' do host
 
### docker commit [hash-do-container] [nome-da-imagem]
Cria uma nova imagem baseada na inicial, mantendo o que foi criado/alterado durante a execução desta nova imagem.

### docker stop|start [args] [container]
Inicia ou pausa a execução de um container

> Alguns comandos que facilitam o uso do docker

> docker rm \`docker ps -qa\`      # deleta todos os containers retornados na consulta

> docker rmi $(docker images -q) # deleta todas as imagens retornadas na consulta

## Dockerfile
