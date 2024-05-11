# Projeto de IA para apoio a situaçãoes de desastre.

Este projeto utiliza as LLMS e IA Generativa para analisar documentos relacionados a uma crise e dar respostas rápidas às pessoas atingidas e todos que se propuserem a ajudar. Vamos usar o caso da atual tragédia do Rio Grande do Sul para exemplificar, onde serão inputados textos mais próximos da realidade, com o objetivo de responder adequadamente ajudando os as pessoas que precisam de informaçãoes o mais rápido possível.

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
## Documentos informativos
```bash
  DOCUMENT1 = {
      "Título": "Abrigos para Vítimas das Enchentes no Rio Grande do Sul",
      "Conteúdo": "Para auxiliar na resposta ao desastre, diversos abrigos foram criados em todo o estado para acolher as vítimas das enchentes. Este documento tem como objetivo fornecer um guia completo sobre esses abrigos, incluindo informações sobre sua localização, serviços oferecidos e como entrar em contato. Localização dos Abrigos: Abrigos foram instalados em diversos municípios do Rio Grande do Sul, com foco nas áreas mais afetadas pelas inundações. Localização: Porto Alegre: Clube Geraldo Santana - Rua Luiz de Camões, 337 – bairro Santo Antônio. Canoas Ulbra: Av. Farroupilha, 8.001 – bairro São JoséUma. Lista completa e atualizada dos abrigos, incluindo seus endereços e informações de contato, pode ser encontrada nos seguintes sites: Secretaria da Segurança Pública do Rio Grande do Sul: https://www.ssp.rs.gov.br/
Defesa Civil do Rio Grande do Sul: [URL inválido removido], Cruz Vermelha Brasileira: https://www.cruzvermelha.org.br/ .Serviços Oferecidos: Os abrigos para vítimas das enchentes no Rio Grande do Sul oferecem uma variedade de serviços essenciais para atender às necessidades básicas dos desabrigados. Entre os serviços oferecidos, estão: Alojamento: As pessoas desabrigadas podem dormir em camas ou colchões em um ambiente seguro e protegido. Alimentação: Três refeições por dia são fornecidas aos abrigados, além de lanches entre as refeições. Cuidados médicos: Equipes médicas estão disponíveis nos abrigos para fornecer atendimento médico e odontológico básico. Apoio psicológico: Psicólogos e assistentes sociais estão disponíveis para oferecer apoio emocional e psicológico aos abrigados.
Serviços sociais: Assistentes sociais podem ajudar os abrigados a entrar em contato com familiares e amigos, obter documentos perdidos e acessar outros serviços de assistência social.
Como Entrar em Contato: Secretaria da Segurança Pública do Rio Grande do Sul: (51) 3289-3100. Defesa Civil do Rio Grande do Sul: 0800-643-1992. Cruz Vermelha Brasileira: (51) 3217-4500. Informações Adicionais:
Doações: Se você deseja ajudar as vítimas das enchentes no Rio Grande do Sul, você pode fazer doações em dinheiro ou itens de primeira necessidade para as organizações humanitárias que estão atuando no estado.
Voluntariado: Você também pode se voluntariar para ajudar nos abrigos ou em outras atividades de apoio às vítimas das enchentes."}
  
  DOCUMENT2 = {
      "Título": "Centros de doação oficiais",
      "Conteúdo": "
	  Defesa Civil do Rio Grande do Sul:
		Porto Alegre: 
	  	Centro Administrativo Fernando Ferrari (Caff) - Avenida Borges de Medeiros, 1.501 - Praia de Belas.
	  	Palácio Piratini - Praça Marechal Deodoro, s/nº - Centro.
          Fundação de Assistência Social e Esporte (FASE):
	  	Porto Alegre:
		Ginásio Municipal de Esportes Alceu Carvalho - Rua Barão de Urussuanga, 1.560 - Passo das Pedras.	
          Postos de coleta em órgãos públicos:
	  	Prefeituras Municipais: Diversas prefeituras no Rio Grande do Sul estão recebendo doações.
	  	Câmaras de Vereadores: Algumas Câmaras de Vereadores também estão servindo como pontos de coleta de doações.
          Instituições de caridade:
            	Cruz Vermelha Brasileira: http://www.cruzvermelha.org.br/pb/institucional/doacoes/
            	Caritas Arquidiocesana de Porto Alegre: https://rs.caritas.org.br/
            	Legião da Boa Vontade: https://lbv.org/
          Campanhas de doação online:
            	VOAA: https://voaa.me/
            	Benfeitoria: https://benfeitoria.com/.
	"}
  
  DOCUMENT3 = {
      "Título": "O que doar",
      "Conteúdo": "
		Alimentos: Alimentos não perecíveis, como enlatados, massas, arroz, feijão, leite em pó e óleo.
		Água potável: Garrafas de água mineral ou água potável em galões.
		Itens de higiene pessoal: Sabonete, shampoo, creme dental, escova de dentes, desodorante, papel higiênico e fraldas infantis.
		Itens de limpeza: Detergente, água sanitária, vassouras, rodos e panos de chão.
		Roupas e calçados: Roupas e calçados em bom estado para todas as idades.
		Cobertores e colchões: Cobertores, mantas e colchões para auxiliar no abrigo das vítimas.
		Utensílios domésticos: Pratos, copos, talheres, panelas e outros utensílios básicos para cozinha.
		Lembre-se:Antes de doar, verifique se os itens estão em boas condições e adequados para o uso pelas vítimas das enchentes.
		Ao doar roupas e calçados, certifique-se de que estejam limpos e secos.
		Organize as doações por tipo de item para facilitar a triagem e distribuição.
		Se possível, doe dinheiro. As doações em dinheiro permitem que as organizações humanitárias comprem os itens mais necessários no momento.
"}
  
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
## Função para comparar a consulta do usuario com os textos
```bash
  def gerar_e_buscar_consulta(consulta, base, model):
  embedding_da_consulta = genai.embed_content(model=model,
                                 content=consulta,
                                 task_type="RETRIEVAL_QUERY")["embedding"]

  produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)

  indice = np.argmax(produtos_escalares)
  return df.iloc[indice]["Conteudo"]
```
## Chama a função passando o prompt (consulta do usuário), o data frame e o modelo
```bash
prompt = input('Esperando Prompt: ')

trecho = gerar_e_buscar_consulta(prompt, df, model)
print(trecho)
```
## Configurações do Gemini para nova busca
```bash
  generation_config = {
    "temperature": 0,
    "candidate_count": 1
  }
```
## Usando o Gemini 1.0 Pro para reescrever o texto passando as configurações para que o mesmo fique mais amigável ao usuário.
```bash
  prompt = f"Reescreva esse texto de uma forma mais amigável, sem adicionar informações que não façam parte do texto: {trecho}"
  
  model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                  generation_config=generation_config)
  response = model_2.generate_content(prompt)
  print(response.text)

```
