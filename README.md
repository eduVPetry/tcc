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

O projeto utiliza três tipos de arquivos de configuração na pasta `config/`:

1. `param_general.txt`: Parâmetros gerais do projeto
2. `param_pso_<experiment_id>.txt`: Parâmetros específicos do algoritmo PSO para cada experimento
3. `param_rpms.txt`: Parâmetros dos modelos de física de rochas

## 📊 Uso

Para executar um único experimento (teste):

```bash
python3 -m src.run_single_experiment
```

Para executar todos os experimentos, você tem duas opções:

1. Execução Sequencial:
```bash
python3 -m src.experiments_driver_sequential
```

2. Execução Paralela (recomendado para melhor performance):
```bash
python3 -m src.experiments_driver_parallel
```

O driver de experimentos irá:
1. Gerar combinações de parâmetros do PSO (iterações, número de partículas, coeficientes cognitivo e social, modo de atualização dos coeficientes)
2. Para cada combinação, dado uma quantidade de repetições:
   - Criar um experimento com os parâmetros específicos
   - Executar a calibração para cada modelo de física de rochas
   - Salvar os resultados e métricas de desempenho
3. Gerar gráficos e resultados na pasta `results/`

## 📁 Estrutura do Projeto

```
.
├── config/                 # Arquivos de configuração
│   ├── param_general.txt
│   ├── param_pso_<experiment_id>.txt
│   └── param_rpms.txt
├── data/                  # Dados de entrada
├── results/               # Resultados e gráficos
├── src/                   # Código fonte
│   ├── PSO.py
│   ├── PSORPInversion_Phi_v2_eval.py
│   ├── experiment.py
│   ├── experiment_utils.py
│   ├── experiments_driver_parallel.py         # Driver de experimentos paralelos
│   ├── experiments_driver_sequential.py       # Driver de experimentos sequenciais
│   ├── main_dm_las_forms_plots_separados.py
│   ├── run_single_experiment.py               # Execução de experimento único
│   └── utils.py
└── requirements.txt       # Dependências do projeto
```

## 📈 Resultados

Os resultados são salvos na pasta `results/` e incluem:
- CSV contendo os parâmetros de cada experimento e o melhor erro obtido 
- Gráficos de comparação entre dados reais e resultados experimentais
- Valores de porosidade estimados
- Métricas de qualidade do ajuste

## 👨‍🎓 Autor

Eduardo Vicente Petry - Graduando em Ciência da Computação  
Universidade Federal de Santa Catarina (UFSC)
