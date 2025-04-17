# Otimização por Enxame de Partículas (PSO) na calibração de Modelos de Física de Rochas (RPMs)

Este projeto é parte do Trabalho de Conclusão de Curso (TCC) em Ciência da Computação da Universidade Federal de Santa Catarina (UFSC). O trabalho implementa a calibração de diferentes modelos de física de rochas (RPMs) utilizando o algoritmo de Otimização por Enxame de Partículas (PSO). O objetivo é estimar a porosidade a partir de dados de poços, utilizando a velocidade de ondas P e porosidade real como entrada.

## 📋 Descrição

O projeto utiliza o algoritmo PSO para otimizar os parâmetros dos RPMs, buscando a melhor correspondência entre os dados observados e os resultados obtidos. Os RPMs implementados incluem:

- Wyllie
- Raymer
- SoftSand
- StiffSand
- SphericalInclusion
- BerrymanInclusion

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone git@github.com:eduVPetry/tcc.git
```

2. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuração

O projeto utiliza três arquivos de configuração na pasta `config/`:

1. `param_general.txt`: Parâmetros gerais do projeto
2. `param_pso.txt`: Parâmetros específicos do algoritmo PSO
3. `param_rpms.txt`: Parâmetros dos modelos de física de rochas

## 📊 Uso

Para executar os experimentos:

```bash
python src/experiments_driver.py
```

O script irá:
1. Gerar combinações de parâmetros do PSO (iterações, número de partículas, coeficientes cognitivo e social, modo de atualização dos coeficientes)
2. Para cada combinação:
   - Criar um experimento com os parâmetros específicos
   - Executar a calibração para cada modelo de física de rochas
   - Salvar os resultados e métricas de desempenho
4. Gerar gráficos e resultados na pasta `results/`

## 📁 Estrutura do Projeto

```
.
├── config/                 # Arquivos de configuração
│   ├── param_general.txt
│   ├── param_pso.txt
│   └── param_rpms.txt
├── data/                  # Dados de entrada
├── results/               # Resultados e gráficos
├── src/                   # Código fonte
│   ├── PSO.py
│   ├── PSORPInversion_Phi_v2_eval.py
│   ├── experiment.py
│   ├── experiment_utils.py
│   ├── experiments_driver.py
│   ├── main_dm_las_forms_plots_separados.py
│   └── utils.py
└── requirements.txt       # Dependências do projeto
```

## 📈 Resultados

Os resultados são salvos na pasta `results/` e incluem:
- Gráficos de comparação entre dados observados e sintéticos
- Valores de porosidade estimados
- Métricas de qualidade do ajuste

## 👨‍🎓 Autor

Eduardo Vicente Petry - Graduando em Ciência da Computação  
Universidade Federal de Santa Catarina (UFSC)
