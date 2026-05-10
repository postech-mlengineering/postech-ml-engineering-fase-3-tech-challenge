# Postech ML Engineering - Fase 3 - Tech Challenge

Este repositório contém os notebooks e scripts relacionados aos modelos supervisionados e não supervisionados para a análise e predição de atrasos de voos na malha aérea.

## Pré-requisitos

Recomendamos o uso do **[uv](https://github.com/astral-sh/uv)**, um instalador e resolvedor de pacotes Python extremamente rápido escrito em Rust, para gerenciar o ambiente virtual e as dependências deste projeto.

Se você ainda não tem o `uv` instalado, pode instalá-arlo via `pip`:
```bash
pip install uv
```

## Configuração do Ambiente Virtual

Siga os passos abaixo para criar o ambiente, ativá-lo e instalar as dependências necessárias listadas no `requirements.txt`.

### 1. Criar o Ambiente Virtual (.venv)
Na raiz do projeto, execute o seguinte comando:
```bash
uv venv
```
Isso criará uma pasta `.venv` na raiz do seu projeto contendo o ambiente Python isolado.

### 2. Ativar o Ambiente Virtual
**No Windows (PowerShell/CMD):**
```bash
.\.venv\Scripts\activate
```

**No Linux/macOS:**
```bash
source .venv/bin/activate
```

*(Você saberá que o ambiente está ativado quando vir o prefixo `(.venv)` na sua linha de comando).*

### 3. Instalar as Dependências
Com o ambiente ativado, instale os pacotes listados no arquivo `requirements.txt`:
```bash
uv pip install -r requirements.txt
```
O `uv` fará o download e a instalação de pacotes como `pandas`, `scikit-learn`, `plotly`, `xgboost` e o `ipykernel` (necessário para rodar os notebooks) de forma muito rápida.

---

## Execução dos Notebooks

Os notebooks contendo as análises e treinamentos dos modelos estão na pasta `notebooks/`.

### Opção 1: Usando o VS Code (Recomendado)
1. Abra o arquivo do notebook (ex: `notebooks/modelo_nao_supervisionado.ipynb` ou `notebooks/modelo_supervisionado.ipynb`) no VS Code.
2. No canto superior direito, clique em **Select Kernel** (Selecionar Kernel).
3. Vá em **Python Environments** (Ambientes Python) e selecione a opção que contém o ambiente recém-criado, normalmente indicada como `Python 3.x.x ('.venv': venv)`.
4. Execute as células normalmente.

### Opção 2: Usando o Jupyter via Terminal
Se preferir utilizar a interface web do Jupyter, execute o comando abaixo (certifique-se de que o ambiente `.venv` esteja ativado):

```bash
jupyter notebook
```
Isso abrirá o navegador onde você poderá navegar até a pasta `notebooks/` e executar os arquivos `.ipynb`.
