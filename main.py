import cx_Oracle
import pandas as pd
from langchain.llms import AutoLM
from langchain.llms.prompts import ChainGPTLMTemplatePrompt
from langchain.pipelines import Pipeline
from langchain.storage import SQLiteTarget

# Credenciais e detalhes do banco de dados
con_string = "oracle+cx_oracle://user:password@host:port/database"

# Conectar ao banco de dados
connection = cx_Oracle.connect(con_string)

# Criar cursor para executar consultas
cursor = connection.cursor()

# Consulta SQL para selecionar dados do produto
query = "SELECT * FROM produtos"

# Executar a consulta e recuperar os dados
cursor.execute(query)
produtos = cursor.fetchall()

# Fechando o cursor
cursor.close()

# Convertendo os dados em um DataFrame do Pandas
produtos_df = pd.DataFrame(produtos, columns=["coluna1", "coluna2", ...])

def load_produtos(ctx):
    produtos = ctx.get(produtos_df)
    return produtos

def generate_qa(produto):
    # Exemplos de perguntas em português
    pergunta = f"Qual a cor do {produto['nome']}?"
    resposta = f"A cor do {produto['nome']} é {produto['cor']}"
    return pergunta, resposta

def create_qa_pairs(produtos):
    qa_pairs = []
    for produto in produtos:
        pergunta, resposta = generate_qa(produto)
        qa_pairs.append({"question": pergunta, "answer": resposta})
    return qa_pairs

# Criar banco de dados SQLite
sqlite_target = SQLiteTarget("treinamento_qa.sqlite")

# Criar tabelas e colunas
sqlite_target.create_table(
    "qa_pairs",
    columns={"id": "INTEGER PRIMARY KEY", "question": "TEXT", "answer": "TEXT"},
)

# Modelo LLM em português (substitua por sua LLM preferida)
llm = AutoLM("gpt3-portuguese")

# Template de prompt em português
prompt_template = ChainGPTLMTemplatePrompt(
    input_text="Pergunta: {{ question }}\nResposta: {{ answer }}",
    instruction="Responda à seguinte pergunta sobre o produto:",
)

# Construir o pipeline
pipeline = Pipeline(
    load_produtos,
    llm.run,
    prompt_template=prompt_template,
    preprocessor=create_qa_pairs,
    storage=sqlite_target,
)

# Executar o pipeline
pipeline.run()

print("Dados de treinamento em português gerados e armazenados em treinamento_qa.sqlite!")
