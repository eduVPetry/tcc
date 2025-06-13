import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carregar os resultados
df = pd.read_csv("results/results.csv")

# Criar identificador único de combinação de parâmetros
df["config_id"] = df.apply(
    lambda row: f"iter={row['iter']}, part={row['particles']}, cmode={row['cmode']}, "
                f"c1=[{row['c1_start']:.1f}-{row['c1_end']:.1f}], c2=[{row['c2_start']:.1f}-{row['c2_end']:.1f}]",
    axis=1
)

# Agrupar pelas 164 combinações únicas
grouped = df.groupby("config_id").agg(
    mean_error=("best_error", "mean"),
    std_error=("best_error", "std"),
    mean_runtime=("runtime", "mean"),
    std_runtime=("runtime", "std"),
    iter=("iter", "first"),
    particles=("particles", "first"),
    cmode=("cmode", "first"),
    c1_start=("c1_start", "first"),
    c1_end=("c1_end", "first"),
    c2_start=("c2_start", "first"),
    c2_end=("c2_end", "first")
).reset_index()

# Ordenar por erro médio crescente
grouped_sorted = grouped.sort_values(by="mean_error").reset_index(drop=True)
columns_order = ["iter", "particles", "cmode", "c1_start", "c1_end", "c2_start", "c2_end", 
                "mean_error", "std_error", "mean_runtime", "std_runtime"]

# Salvar todas as 164 combinações agregadas em um arquivo CSV
grouped_sorted[columns_order].to_csv("results/results_grouped.csv", index=False)
print("Todas as 164 combinações agregadas salvas em 'results/results_grouped.csv'")

# Salvar as 10 melhores combinações em um arquivo CSV com a ordem especificada
top10 = grouped_sorted.head(10)
top10[columns_order].to_csv("results/top10_grouped.csv", index=False)
print("\nTop 10 combinações salvas em 'results/top10_grouped.csv'")

# Análise gráfica: relação entre erro médio e tempo médio
plt.figure(figsize=(10, 6))
sns.scatterplot(data=grouped, x="mean_runtime", y="mean_error", hue="cmode", palette="Set1")
plt.xlabel("Tempo médio de execução (s)")
plt.ylabel("Erro médio")
plt.title("Relação entre tempo de execução e erro médio por configuração")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/relacao_tempo_erro.png")
plt.close()

# Análise da variabilidade do erro
print(f"\nDesvio padrão médio do erro entre todas as combinações: {grouped['std_error'].mean():.6f}")
print(f"Desvio padrão máximo do erro: {grouped['std_error'].max():.6f}")
print(f"Erro médio mínimo global: {grouped['mean_error'].min():.6f}")
print(f"Erro médio máximo global: {grouped['mean_error'].max():.6f}")

# Análise da influência dos parâmetros
# Boxplots para erro e tempo por parâmetro
plt.figure(figsize=(12, 6))
sns.boxplot(data=grouped, x="particles", y="mean_runtime")
plt.xlabel("Número de partículas")
plt.ylabel("Tempo médio de execução (s)")
plt.title("Tempo médio de execução por número de partículas")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/boxplot_tempo_particulas.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=grouped, x="iter", y="mean_runtime")
plt.xlabel("Número de iterações")
plt.ylabel("Tempo médio de execução (s)")
plt.title("Tempo médio de execução por número de iterações")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/boxplot_tempo_iteracoes.png")
plt.close()
