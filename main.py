from langchain.llms import AutoLM
from langchain.llms.prompts import ChainGPTLMTemplatePrompt
from langchain.pipelines import Pipeline
from langchain.storage import SQLSource, SQLiteTarget

# Conectar ao banco de dados SQL (origem)
sql_source = SQLSource(
    "mysql+pymysql://user:password@host:port/database",
    query="SELECT * FROM produtos"
)

# Criar banco de dados SQLite (destino)
sqlite_target = SQLiteTarget("treinamento_qa.sqlite")

# Carregar os dados do produto
def load_produtos(ctx):
    produtos = ctx.get(sql_source)
    return produtos

# Criar modelo LLM (substitua por sua LLM preferida)
llm = AutoLM("gpt3")  # Exemplo, substitua por endpoint ou classe LLM compatível

# Criar template de prompt para perguntas e respostas
prompt_template = ChainGPTLMTemplatePrompt(
    input_text="Pergunta: {{ question }}\nResposta: {{ answer }}",
    instruction="Respond to the following question about the product:",
)

# Função para gerar pares pergunta-resposta
def generate_qa(produto):
    # Exemplos de perguntas
    pergunta = f"Qual a cor do {produto['nome']}?"
    resposta = f"A cor do {produto['nome']} é {produto['cor']}"
    return pergunta, resposta

# Pipeline para processar dados e gerar pares QA
def create_qa_pairs(produtos):
    qa_pairs = []
    for produto in produtos:
        pergunta, resposta = generate_qa(produto)
        qa_pairs.append({"question": pergunta, "answer": resposta})
    return qa_pairs

# Construir o pipeline LangChain
pipeline = Pipeline(
    load_produtos,
    llm.run,
    prompt_template=prompt_template,
    preprocessor=create_qa_pairs,
    storage=sqlite_target,
)

# Criar tabelas e colunas no banco de dados SQLite
pipeline.storage.create_table(
    "qa_pairs",
    columns={"id": "INTEGER PRIMARY KEY", "question": "TEXT", "answer": "TEXT"},
)

# Executar o pipeline para gerar pares pergunta-resposta
pipeline.run()

print("Dados de treinamento gerados e armazenados em treinamento_qa.sqlite!")
