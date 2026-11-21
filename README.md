# vintage_analyzer.py
Script Python que realiza Análise de Safra (Vintage Analysis) de carteira de crédito. Agrupa operações por mês de concessão e acompanha a inadimplência (30/60/90+ dias) ao longo do tempo em carteira (MOB), gerando curvas de maturação, heatmaps e diagnóstico automático de deterioração das safras recentes.
"""
===============================================================================
 ANÁLISE DE SAFRA (VINTAGE ANALYSIS) - CARTEIRA DE CRÉDITO
===============================================================================
Autor  : [seu nome]
Objetivo:
    Agrupar operações de crédito pela safra (mês/trimestre de concessão) e
    acompanhar a evolução da inadimplência ao longo do tempo de vida (MOB -
    Months On Book). Permite responder:
      * A qualidade das safras recentes está piorando ou melhorando?
      * Onde ocorre o "pico" de inadimplência de cada safra?
      * Qual é a velocidade de maturação (curva de "onda") de cada coorte?

Métricas principais:
    - Taxa de atraso (30 / 60 / 90+ dias) por safra x MOB
    - Curva de maturação (vintage curve)
    - Pico de inadimplência por safra
    - Comparação entre safras recentes vs antigas

Bibliotecas: pandas, numpy, matplotlib, seaborn
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from datetime import date, timedelta

# Estilo visual
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (13, 7)
plt.rcParams["axes.titleweight"] = "bold"

# Semente para reprodutibilidade
RNG = np.random.default_rng(seed=42)
PD   = np.random.default_rng(seed=42)  # alias

# ---------------------------------------------------------------------------
# 1. GERAÇÃO DE DADOS SINTÉTICOS (simulando um data warehouse de crédito)
# ---------------------------------------------------------------------------
# Cada linha = uma parcela (installment) de um contrato.
# Colunas mínimas esperadas em produção:
#   id_contrato | data_concessao | data_vencimento | valor_parcela
#   | data_pagamento (NaN se não pagou) | saldo_devedor | status

def gerar_carteira(
    n_contratos: int = 6000,
    data_inicio: str = "2022-01-01",
    data_fim: str    = "2024-12-31",
    prazo_min: int   = 6,
    prazo_max: int   = 36,
) -> pd.DataFrame:
    """Gera carteira sintética com deterioração gradual das safras recentes
    (para que o gráfico mostre claramente uma piora nas safras novas)."""

    # Datas de concessão distribuídas uniformemente entre início e fim
    dias_total = (pd.Timestamp(data_fim) - pd.Timestamp(data_inicio)).days
    dias_concessao = RNG.integers(0, dias_total, size=n_contratos)
    datas_concessao = pd.Timestamp(data_inicio) + pd.to_timedelta(dias_concessao, unit="D")

    # Prazo do contrato (meses)
    prazos = RNG.integers(prazo_min, prazo_max + 1, size=n_contratos)

    # Valor do contrato (log-normal — típico de carteira de crédito)
    valores = np.round(RNG.lognormal(mean=8.5, sigma=0.6, size=n_contratos), 2)

    # ---- Deterioração das safras recentes (o "enredo" da análise) ----------
    # Quanto mais recente a concessão, maior a probabilidade-base de default.
    meses_do_inicio = (
        (pd.to_datetime(datas_concessao) - pd.Timestamp(data_inicio)).days / 30
    ).astype(int)
    # Probabilidade de default cresce ~1.2 p.p. por mês de vintage
    pd_base = 0.03 + 0.012 * meses_do_inicio
    pd_base = np.clip(pd_base, 0.01, 0.55)

    # ---- Simulando o "momento" do default (MOB em que quebra) --------------
    # Usamos uma distribuição beta deslocada: pico ao redor do MOB 6-10.
    mob_default = RNG.beta(a=3, b=6, size=n_contratos) * prazos
    # O contrato quebra se um uniforme < PD base relativa (ajustada pelo prazo)
    u = RNG.uniform(size=n_contratos)
    quebra = u < pd_base * (prazos / prazos.max())  # contratos longos quebram mais
    mob_default = np.where(quebra, mob_default, np.nan)

    # ---- Explodindo contratos em parcelas mensais --------------------------
    registros = []
    for i in range(n_contratos):
        dt_conc   = datas_concessao[i]
        prazo     = int(prazos[i])
        valor     = valores[i]
        mob_queb  = mob_default[i]

        for m in range(prazo):
            venc = dt_conc + pd.DateOffset(months=m + 1)
            # Consideramos apenas parcelas já vencidas até "hoje"
            if venc > pd.Timestamp(data_fim):
                continue

            mob = m + 1  # meses em carteira na parcela
            # Se houve quebra: pagamentos cessam após mob_queb
            if not np.isnan(mob_queb):
                if mob < mob_queb:
                    pagou = True
                else:
                    # pode haver arrears parciais antes do 90+
                    pagou = RNG.uniform() < 0.15 if mob == mob_queb else False
            else:
                pagou = RNG.uniform() < 0.97  # pequena inadimplência residual

            dias_atraso = 0 if pagou else int(RNG.integers(30, 180))
            dt_pgto     = venc + pd.Timedelta(days=dias_atraso) if pagou else pd.NaT

            registros.append(
                {
                    "id_contrato":     i,
                    "data_concessao":  dt_conc,
                    "mob":             mob,
                    "data_vencimento": venc,
                    "data_pagamento":  dt_pgto,
                    "valor_parcela":   round(valor / prazo, 2),
                    "dias_atraso":     dias_atraso,
                }
            )

    df = pd.DataFrame(registros)
    df["safra"] = df["data_concessao"].dt.to_period("M").dt.to_timestamp()
    return df


# ---------------------------------------------------------------------------
# 2. CÁLCULO DAS MÉTRICAS DE SAFRA
# ---------------------------------------------------------------------------
def calcular_vintage(df: pd.DataFrame, bucket: str = "30") -> pd.DataFrame:
    """
    Retorna DataFrame com taxa de atraso por safra x MOB.

    bucket: '30' -> parcela com >=30 dias de atraso
            '60' -> >=60
            '90' -> >=90 (NPL clássico)
    """
    limite = int(bucket)
    df = df.copy()
    df["em_atraso"] = (df["dias_atraso"] >= limite).astype(int)

    # Denominador = todas as parcelas vencidas daquela safra naquele MOB
    grp = (
        df.groupby(["safra", "mob"])
          .agg(parcelas=("id_contrato", "size"),
               em_atraso=("em_atraso", "sum"))
          .reset_index()
    )
    grp["taxa_atraso"] = grp["em_atraso"] / grp["parcelas"]
    return grp


def metricas_resumo(vintage: pd.DataFrame) -> pd.DataFrame:
    """Resumo por safra: MOB em que ocorre o pico e valor do pico (90+)."""
    idx_pico = vintage.groupby("safra")["taxa_atraso"].idxmax()
    picos = vintage.loc[idx_pico, ["safra", "mob", "taxa_atraso"]]
    picos = picos.rename(
        columns={"mob": "mob_pico", "taxa_atraso": "taxa_pico"}
    ).reset_index(drop=True)
    # MOB máximo observado (maturação da safra)
    maturacao = vintage.groupby("safra")["mob"].max().reset_index(
        name="mob_max"
    )
    resumo = picos.merge(maturacao, on="safra").sort_values("safra")
    return resumo


# ---------------------------------------------------------------------------
# 3. VISUALIZAÇÕES
# ---------------------------------------------------------------------------
def plot_curvas_maturacao(vintage_30, vintage_60, vintage_90,
                          destaque: int = 4):
    """Plota as 3 curvas de maturação (30/60/90) no formato clássico de 'onda'."""

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
    paleta = sns.color_palette("viridis", n_colors=vintage_30["safra"].nunique())
    safras = sorted(vintage_30["safra"].unique())
    cores = dict(zip(safras, paleta))

    for ax, dfv, titulo in zip(
        axes,
        [vintage_30, vintage_60, vintage_90],
        ["Atraso 30+ dias", "Atraso 60+ dias", "Atraso 90+ dias (NPL)"],
    ):
        for safra in safras:
            sub = dfv[dfv["safra"] == safra]
            ax.plot(
                sub["mob"], sub["taxa_atraso"] * 100,
                marker="o", markersize=3, linewidth=1.8,
                color=cores[safra],
                alpha=0.9 if safra in safras[-destaque:] else 0.55,
            )
        ax.set_title(titulo)
        ax.set_xlabel("MOB (meses em carteira)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Taxa de atraso (%)")
    # Legenda apenas no primeiro gráfico
    handles = [
        plt.Line2D([0], [0], color=cores[s], lw=2,
                   label=s.strftime("%Y-%m"))
        for s in safras
    ]
    axes[0].legend(
        handles=handles, title="Safra", bbox_to_anchor=(1.01, 1),
        loc="upper left", fontsize=9, title_fontsize=10, ncol=1,
    )
    fig.suptitle(
        "Curvas de Maturação por Safra — Quanto mais 'alta/íngreme' a curva, "
        "pior a qualidade da safra",
        y=1.02, fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_heatmap(vintage_90: pd.DataFrame):
    """Heatmap Safra x MOB — leitura visual rápida de toda a carteira."""
    pivot = vintage_90.pivot_table(
        index="safra", columns="mob", values="taxa_atraso"
    )
    plt.figure(figsize=(16, 8))
    sns.heatmap(
        pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd",
        cbar_kws={"label": "Taxa atraso 90+ (%)"},
        linewidths=0.5, linecolor="white",
    )
    plt.title("Heatmap de Inadimplência 90+ — Safra x MOB")
    plt.xlabel("MOB (meses em carteira)")
    plt.ylabel("Safra (mês de concessão)")
    plt.tight_layout()
    plt.show()


def plot_pico_por_safra(resumo: pd.DataFrame):
    """Barras do pico de inadimplência por safra — visão executiva."""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    resumo = resumo.copy()
    resumo["safra_str"] = resumo["safra"].dt.strftime("%Y-%m")
    bars = ax1.bar(
        resumo["safra_str"], resumo["taxa_pico"] * 100,
        color=sns.color_palette("rocket_r", len(resumo)),
    )
    ax1.set_ylabel("Pico de atraso 90+ (%)")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_xticklabels(resumo["safra_str"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(
        resumo["safra_str"], resumo["mob_pico"],
        color="black", marker="o", linestyle="--", label="MOB do pico",
    )
    ax2.set_ylabel("MOB em que o pico ocorre")
    ax2.legend(loc="upper left")

    for bar, val in zip(bars, resumo["taxa_pico"] * 100):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
        )

    plt.title("Pico de Inadimplência 90+ por Safra e MOB do Pico")
    plt.tight_layout()
    plt.show()


def plot_comparativo_recentes_vs_antigas(vintage_90, n_recentes=3):
    """Compara curva média das safras antigas x safras recentes."""
    safras = sorted(vintage_90["safra"].unique())
    antigas  = safras[:-n_recentes]
    recentes = safras[-n_recentes:]

    def curva_media(df, safras_grupo):
        sub = df[df["safra"].isin(safras_grupo)]
        return sub.groupby("mob")["taxa_atraso"].mean() * 100

    m_ant = curva_media(vintage_90, antigas)
    m_rec = curva_media(vintage_90, recentes)

    plt.figure(figsize=(12, 6))
    plt.plot(m_ant.index, m_ant.values, marker="o",
             label=f"Média safras antigas ({antigas[0].strftime('%Y-%m')} a "
                   f"{antigas[-1].strftime('%Y-%m')})",
             color="steelblue", linewidth=2.5)
    plt.plot(m_rec.index, m_rec.values, marker="s",
             label=f"Média safras recentes ({recentes[0].strftime('%Y-%m')} a "
                   f"{recentes[-1].strftime('%Y-%m')})",
             color="crimson", linewidth=2.5)
    plt.fill_between(m_rec.index, m_ant.reindex(m_rec.index, method="ffill"),
                     m_rec.values, where=m_rec.values >
                     m_ant.reindex(m_rec.index, method="ffill"),
                     color="crimson", alpha=0.15, label="Deterioração")
    plt.title("Safras Recentes vs Safras Antigas — A qualidade está piorando?")
    plt.xlabel("MOB")
    plt.ylabel("Atraso 90+ (%)")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print(" ANÁLISE DE SAFRA (VINTAGE ANALYSIS) — CARTEIRA DE CRÉDITO")
    print("=" * 78)

    # -- 1. Dados ----------------------------------------------------------
    print("\n[1/4] Gerando carteira sintética...")
    df = gerar_carteira(n_contratos=6000)
    print(f"      Contratos: {df['id_contrato'].nunique():,}")
    print(f"      Parcelas : {len(df):,}")
    print(f"      Safras   : {df['safra'].nunique()} meses "
          f"({df['safra'].min():%Y-%m} a {df['safra'].max():%Y-%m})")

    # -- 2. Vintage --------------------------------------------------------
    print("\n[2/4] Calculando curvas de maturação (30 / 60 / 90+)...")
    v30 = calcular_vintage(df, bucket="30")
    v60 = calcular_vintage(df, bucket="60")
    v90 = calcular_vintage(df, bucket="90")

    resumo = metricas_resumo(v90)
    print("\n      Resumo por safra (pico 90+):")
    resumo_print = resumo.copy()
    resumo_print["safra"] = resumo_print["safra"].dt.strftime("%Y-%m")
    resumo_print["taxa_pico"] = (
        resumo_print["taxa_pico"] * 100
    ).round(2).astype(str) + "%"
    print(resumo_print.to_string(index=False))

    # -- 3. Gráficos -------------------------------------------------------
    print("\n[3/4] Gerando curvas de maturação (onda)...")
    plot_curvas_maturacao(v30, v60, v90)

    print("      Gerando heatmap Safra x MOB...")
    plot_heatmap(v90)

    print("      Gerando pico por safra...")
    plot_pico_por_safra(resumo)

    print("      Gerando comparativo recentes vs antigas...")
    plot_comparativo_recentes_vs_antigas(v90, n_recentes=3)

    # -- 4. Conclusão ------------------------------------------------------
    print("\n[4/4] Diagnóstico automático:")
    antigas  = resumo["taxa_pico"].iloc[:-3].mean()
    recentes = resumo["taxa_pico"].iloc[-3:].mean()
    delta_pp = (recentes - antigas) * 100
    tendencia = "PIORANDO" if delta_pp > 0 else "MELHORANDO"
    print(f"      Pico médio (safras antigas) : {antigas*100:.2f}%")
    print(f"      Pico médio (3 mais recentes): {recentes*100:.2f}%")
    print(f"      Delta: {delta_pp:+.2f} p.p.  ->  Tendência: {tendencia}")
    print("\nConcluído.")


if __name__ == "__main__":
    main()
