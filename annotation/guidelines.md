# Annotation Guidelines -- SIREVA-SUS NER

Version 0.1 -- draft for internal review.

## Entity Types

### PATOGENO
Disease-causing organism identified at the species or genus level.

- Include: binomial names (*Streptococcus pneumoniae*), abbreviations (*S. pneumoniae*),
  common names (pneumococo, meningococo), and plural forms.
- Exclude: serological identifiers -- those are SOROTIPO.

### SOROTIPO
Serological classification attached to a PATOGENO when the context makes
the identity unambiguous.

- Include: sorotipo 19A, sorogrupo B, tipo b, sorogrupo W135.
- Note: annotate SOROTIPO as a separate span even when it appears directly
  after the pathogen name (*S. pneumoniae* **sorotipo 19A**).

### FAIXA_ETARIA
Named age stratum used for epidemiological stratification.

- Include: explicit intervals (12 a 23 meses), comparative expressions
  (menor que 5 anos, acima de 60 anos), demographic labels (lactentes,
  idosos) when used as a stratum.
- Exclude: isolated ages without stratum meaning (o paciente tinha 3 anos).

### LOCAL
Named geographic entity at any administrative level.

- Include: countries, states, municipalities, macro-regions
  (regiao Sudeste), health regions.
- Exclude: non-geographic institutional names (see METRICA_EPI for SINAN).

### PERIODO_TEMPORAL
An explicit temporal reference used to delimit an observation window.

- Include: years (2024), year ranges (2013 a 2024), months
  (janeiro de 2023), epidemiological weeks (semana epidemiologica 35),
  semesters (primeiro semestre de 2022).
- Exclude: relative expressions without explicit markers (recentemente,
  nos ultimos anos).

### METODO_LAB
A named laboratory or diagnostic method or technique.

- Include: cultura, PCR em tempo real, MALDI-TOF, antibiograma,
  teste de aglutinacao, sorotipagem molecular.
- Exclude: generic references to "laboratorio" or "exame" without
  specifying the technique.

### METRICA_EPI
A named epidemiological measure or indicator -- the concept, not the
numeric value.

- Include: taxa de incidencia, numero de casos, cobertura vacinal,
  letalidade, prevalencia, coeficiente de mortalidade.
- Exclude: the numeric value itself (e.g. "3,2" or "45%").

### MANIFESTACAO_CLINICA
A named disease, syndrome, or clinical condition.

- Include: meningite, meningite bacteriana, pneumonia, doenca pneumococica
  invasiva (DPI), sepse, bacteremia.
- Exclude: generic terms (doenca, infeccao) without a qualifying modifier
  that names a specific condition.

## General Principles

1. **Nested entities**: do not annotate; choose the most specific span.
2. **Coreference**: annotate each occurrence independently; do not resolve
   coreference chains.
3. **Abbreviations**: annotate with the same type as the full form.
4. **Articles and prepositions**: exclude from spans
   (e.g. annotate *meningite* not *a meningite*).
5. **Numeric values**: not annotated in this version.
6. **Discontinuous spans**: not supported; annotate the core noun phrase.
