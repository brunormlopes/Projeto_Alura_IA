# Projeto de IA para apoio a situaçãoes de desastre.

Este projeto utiliza as LLMS e IA Generativa para analisar documentos relacionados a uma crise e dar respostas rápidas às pessoas atingidas e todos que se propuserem a ajudar.

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
## Documentos de auxilio
```bash
  DOCUMENT1 = {
      "Título": "Abrigos para Vítimas das Enchentes no Rio Grande do Sul",
      "Conteúdo": "Introdução: As fortes chuvas que atingiram o Rio Grande do Sul em maio de 2024 causaram inundações e deslocamentos em massa, afetando milhares de pessoas. Para auxiliar na resposta a esse   desastre, diversos abrigos foram criados em todo o estado para acolher as vítimas das enchentes. Este documento tem como objetivo fornecer um guia completo sobre esses abrigos, incluindo informações sobre sua localização, serviços oferecidos e como entrar em contato. Localização dos Abrigos: Abrigos foram instalados em diversos municípios do Rio Grande do Sul, com foco nas áreas mais afetadas pelas inundações. Uma lista completa e atualizada dos abrigos, incluindo seus endereços e informações de contato, pode ser encontrada nos seguintes sites: Secretaria da Segurança Pública do Rio Grande do Sul: https://www.ssp.rs.gov.br/
Defesa Civil do Rio Grande do Sul: [URL inválido removido], Cruz Vermelha Brasileira: https://www.cruzvermelha.org.br/ .Serviços Oferecidos: Os abrigos para vítimas das enchentes no Rio Grande do Sul oferecem uma variedade de serviços essenciais para atender às necessidades básicas dos desabrigados. Entre os serviços oferecidos, estão: Alojamento: As pessoas desabrigadas podem dormir em camas ou colchões em um ambiente seguro e protegido. Alimentação: Três refeições por dia são fornecidas aos abrigados, além de lanches entre as refeições. Cuidados médicos: Equipes médicas estão disponíveis nos abrigos para fornecer atendimento médico e odontológico básico. Apoio psicológico: Psicólogos e assistentes sociais estão disponíveis para oferecer apoio emocional e psicológico aos abrigados.
Serviços sociais: Assistentes sociais podem ajudar os abrigados a entrar em contato com familiares e amigos, obter documentos perdidos e acessar outros serviços de assistência social.
Como Entrar em Contato: Secretaria da Segurança Pública do Rio Grande do Sul: (51) 3289-3100. Defesa Civil do Rio Grande do Sul: 0800-643-1992. Cruz Vermelha Brasileira: (51) 3217-4500. Informações Adicionais:
Doações: Se você deseja ajudar as vítimas das enchentes no Rio Grande do Sul, você pode fazer doações em dinheiro ou itens de primeira necessidade para as organizações humanitárias que estão atuando no estado.
Voluntariado: Você também pode se voluntariar para ajudar nos abrigos ou em outras atividades de apoio às vítimas das enchentes."}
  
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
