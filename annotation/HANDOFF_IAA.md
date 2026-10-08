# Handoff — Anotação Manual para Cálculo de Cohen's Kappa

## Contexto

Este pacote contém tudo que você precisa para anotar **180 sentenças** extraídas de relatórios epidemiológicos do SIREVA-SUS (vigilância de meningite/pneumonia, 2013–2024).  
Sua anotação será comparada com a do primeiro anotador para calcular o **Cohen's Kappa** (concordância inter-anotador, IAA).

> **Importante:** anote de forma independente, sem consultar resultados de outra pessoa.

---

## Arquivos deste pacote

| Arquivo | O que é |
|---|---|
| `HANDOFF_IAA.md` | Este documento |
| `guidelines.md` | Definição das 8 entidades com exemplos e regras de borda |
| `samples/label_studio_iaa_import.json` | Import para o Label Studio — **180 sentenças de texto narrativo** |
| `samples/label_studio_iaa_tables_import.json` | Import para o Label Studio — **100 linhas de tabela** |

> Crie dois projetos separados no Label Studio (um para texto, um para tabelas) e importe o arquivo correspondente em cada um.

---

## Entidades a anotar

| Tag | O que anotar |
|---|---|
| `PATOGENO` | Organismo causador da doença (*Streptococcus pneumoniae*, pneumococo…) |
| `SOROTIPO` | Classificação sorológica (sorotipo 19A, sorogrupo B…) |
| `FAIXA_ETARIA` | Estrato etário epidemiológico (menores de 5 anos, idosos…) |
| `LOCAL` | Entidade geográfica (Brasil, Sudeste, São Paulo…) |
| `PERIODO_TEMPORAL` | Janela temporal explícita (2024, jan–jun 2022, semana epi 35…) |
| `METODO_LAB` | Método laboratorial/diagnóstico (PCR, cultura, MALDI-TOF…) |
| `METRICA_EPI` | Indicador epidemiológico **sem o valor numérico** (taxa de incidência, letalidade…) |
| `MANIFESTACAO_CLINICA` | Doença ou síndrome (meningite bacteriana, pneumonia, sepse…) |

Leia o `guidelines.md` antes de começar — ele tem exemplos e regras de borda para cada entidade.

---

## Passo a passo no Label Studio

### 1. Criar o projeto

1. Abra o Label Studio (`http://localhost:8080` ou onde estiver instalado).
2. Clique em **Create Project** → dê o nome `SIREVA-IAA`.
3. Aba **Labeling Setup** → escolha **Natural Language Processing → Named Entity Recognition**.
4. Substitua o XML de configuração pelo bloco abaixo e salve:

```xml
<View>
  <Labels name="label" toName="text">
    <Label value="PATOGENO"            background="#FF6B6B"/>
    <Label value="SOROTIPO"            background="#4ECDC4"/>
    <Label value="FAIXA_ETARIA"        background="#45B7D1"/>
    <Label value="LOCAL"               background="#96CEB4"/>
    <Label value="PERIODO_TEMPORAL"    background="#FFEAA7"/>
    <Label value="METODO_LAB"          background="#DDA0DD"/>
    <Label value="METRICA_EPI"         background="#98D8C8"/>
    <Label value="MANIFESTACAO_CLINICA" background="#F7DC6F"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```

### 2. Importar as sentenças

1. Na aba **Data Import**, clique em **Upload Files**.
2. Selecione o arquivo `samples/label_studio_iaa_import.json`.
3. Confirme a importação — devem aparecer **180 tarefas**.

### 3. Anotar

- Selecione um trecho de texto e escolha a tag correspondente.
- Cada sentença pode ter zero ou múltiplas entidades.
- Quando terminar uma tarefa, clique em **Submit** (não em Skip).
- Sessões podem ser salvas e retomadas — as tarefas já submetidas ficam marcadas como concluídas.

### 4. Exportar quando terminar

1. No projeto, clique em **Export**.
2. Escolha o formato **JSON-MIN** (ou **JSON**).
3. Salve o arquivo e envie de volta — nome sugerido: `iaa_annotator2.json`.

---

## Dúvidas frequentes

**Posso anotar entidades aninhadas?**  
Não. Escolha o span mais específico. Ex.: "meningite bacteriana por *S. pneumoniae*" → anote "meningite bacteriana" como MANIFESTACAO_CLINICA e "*S. pneumoniae*" como PATOGENO separadamente.

**E se o texto tiver ruído (caracteres estranhos)?**  
Anote o que for interpretável. Se a sentença for completamente ilegível, submeta sem anotações.

**Quanto tempo leva?**  
Esperamos ~3–4 horas no total (≈ 1–2 min por sentença).

---

## Contato

Dúvidas sobre as diretrizes: christian.freitas@unifesp.br
