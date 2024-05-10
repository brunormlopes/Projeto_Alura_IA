# Geração de texto com Gemini API & Google AI Studio

Este repositório demonstra como usar a Gemini API e o Google AI Studio para gerar texto de forma criativa e informativa.

## Pré-requisitos

* Conta do Google Cloud Platform (GCP)
* Habilitar a API do Generative AI
* Instalar a biblioteca `google-generativeai`

## Instalação

```bash
!pip install -q -U google-generativeai
```

## Importações e configurações iniciais

```bash
import numpy as np
import pandas as pd
import google.generativeai as genai

from google.colab import userdata
apy_key = userdata.get('SECRET_KEY')
GOOGLE_API_KEY= apy_key
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
  if 'embedContent' in m.supported_generation_methods:
    print(m.name)
```
## Documentos para busca 
```bash
  DOCUMENT1 = {
      "Título": "Operação do sistema de controle climático",
      "Conteúdo": "O Googlecar tem um sistema de controle climático que permite ajustar a temperatura e o fluxo de ar no carro. Para operar o sistema de controle climático, use os botões e botões localizados no console central.  Temperatura: O botão de temperatura controla a temperatura dentro do carro. Gire o botão no sentido horário para aumentar a temperatura ou no sentido anti-horário para diminuir a temperatura. Fluxo de ar: O botão de fluxo de ar controla a quantidade de fluxo de ar dentro do carro. Gire o botão no sentido horário para aumentar o fluxo de ar ou no sentido anti-horário para diminuir o fluxo de ar. Velocidade do ventilador: O botão de velocidade do ventilador controla a velocidade do ventilador. Gire o botão no sentido horário para aumentar a velocidade do ventilador ou no sentido anti-horário para diminuir a velocidade do ventilador. Modo: O botão de modo permite que você selecione o modo desejado. Os modos disponíveis são: Auto: O carro ajustará automaticamente a temperatura e o fluxo de ar para manter um nível confortável. Cool (Frio): O carro soprará ar frio para dentro do carro. Heat: O carro soprará ar quente para dentro do carro. Defrost (Descongelamento): O carro soprará ar quente no para-brisa para descongelá-lo."}
  
  DOCUMENT2 = {
      "Título": "Touchscreen",
      "Conteúdo": "O seu Googlecar tem uma grande tela sensível ao toque que fornece acesso a uma variedade de recursos, incluindo navegação, entretenimento e controle climático. Para usar a tela sensível ao toque, basta tocar no ícone desejado.  Por exemplo, você pode tocar no ícone \"Navigation\" (Navegação) para obter direções para o seu destino ou tocar no ícone \"Music\" (Música) para reproduzir suas músicas favoritas."}
  
  DOCUMENT3 = {
      "Título": "Mudança de marchas",
      "Conteúdo": "Seu Googlecar tem uma transmissão automática. Para trocar as marchas, basta mover a alavanca de câmbio para a posição desejada.  Park (Estacionar): Essa posição é usada quando você está estacionado. As rodas são travadas e o carro não pode se mover. Marcha à ré: Essa posição é usada para dar ré. Neutro: Essa posição é usada quando você está parado em um semáforo ou no trânsito. O carro não está em marcha e não se moverá a menos que você pressione o pedal do acelerador. Drive (Dirigir): Essa posição é usada para dirigir para frente. Low: essa posição é usada para dirigir na neve ou em outras condições escorregadias."}
  
  documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]
```
## Criando um dataframe com os documentos anteriores
```bash
df = pd.DataFrame(documents)
df.columns = ["Titulo", "Conteudo"]
df
```
## Definindo o modelo do google AI para text embedding
```bash
model = "models/embedding-001"
```
## Realizando o Embedding
```bash
def embed_fn(title, text):
  return genai.embed_content(model=model,
                                 content=text,
                                 title=title,
                                 task_type="RETRIEVAL_DOCUMENT")["embedding"]
```
## Adiciona ao Data Frame os Embeddings, criando uma nova coluna 
```bash
  df["Embeddings"] = df.apply(lambda row: embed_fn(row["Titulo"], row["Conteudo"]), axis=1)
  df
```
## Função para gerar e buscar a consulta
```bash
  def gerar_e_buscar_consulta(consulta, base, model):
  embedding_da_consulta = genai.embed_content(model=model,
                                 content=consulta,
                                 task_type="RETRIEVAL_QUERY")["embedding"]

  produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)

  indice = np.argmax(produtos_escalares)
  return df.iloc[indice]["Conteudo"]
```
## Chama a função passando a consulta, o data frame e o modelo
```bash
consulta = "Como faço para trocar marchas em um carro do Google?"

trecho = gerar_e_buscar_consulta(consulta, df, model)
print(trecho)
```
## Configurações do Gemini para nova busca
```bash
  generation_config = {
    "temperature": 0,
    "candidate_count": 1
  }
```
## Usando o Gemini 1.0 Pro para reescrever o texto passando as configurações 
```bash
  prompt = f"Reescreva esse texto de uma forma mais descontraída, sem adicionar informações que não façam parte do texto: {trecho}"
  
  model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                  generation_config=generation_config)
  response = model_2.generate_content(prompt)
  print(response.text)

```
