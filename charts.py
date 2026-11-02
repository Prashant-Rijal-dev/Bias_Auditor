"""charts.py - Matplotlib / Seaborn figures (embedded in Tkinter) and Plotly (browser)."""
import numpy as np
import seaborn as sns
import plotly.express as px
from matplotlib.figure import Figure


def _fig():
    fig = Figure(figsize=(6, 4), dpi=100)
    return fig, fig.add_subplot(111)


def selection_bar(before, after=None, group="group"):
    fig, ax = _fig()
    x = np.arange(len(before))
    if after is None:
        ax.bar(x, before.values, color="#4C72B0", label="Model")
    else:
        ax.bar(x - 0.2, before.values, 0.4, color="#C44E52", label="Before fix")
        ax.bar(x + 0.2, after.values, 0.4, color="#55A868", label="After fix")
    ax.set_xticks(x)
    ax.set_xticklabels(before.index)
    ax.axhline(max(before.max(), after.max() if after is not None else 0) * 0.8,
               color="gray", ls="--", label="80% of top group")
    ax.set_title(f"Approval rate by {group}")
    ax.set_xlabel(group.capitalize())
    ax.set_ylabel("Approval rate")
    ax.legend()
    fig.tight_layout()
    return fig


def heatmap(df, row="gender", col="region", label="approved"):
    fig, ax = _fig()
    table = df.pivot_table(index=row, columns=col, values=label, aggfunc="mean")
    sns.heatmap(table, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title(f"Approval rate: {row} x {col}")
    fig.tight_layout()
    return fig


def violin(df, group, feature):
    fig, ax = _fig()
    sns.violinplot(data=df, x=group, y=feature, hue=group, ax=ax, legend=False)
    ax.set_title(f"{feature} by {group}")
    fig.tight_layout()
    return fig


def pie(df, group, label="approved"):
    fig, ax = _fig()
    share = df[df[label] == 1][group].value_counts()
    ax.pie(share.values, labels=share.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Share of approvals by {group}")
    fig.tight_layout()
    return fig


def correlation(df):
    fig, ax = _fig()
    sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    return fig


# def plotly_dashboard(df, group, numeric, path, label="approved"):
#     """Interactive 3D scatter + histogram written to one HTML file."""
#     a, b, c = (numeric + numeric * 3)[:3]
#     f1 = px.scatter_3d(df, x=a, y=b, z=c, color=group, symbol=label, opacity=0.7,
#                        title="3D view: who gets approved?")
#     f2 = px.histogram(df, x=a, color=group, barmode="overlay", facet_col=label,
#                       title=f"{a} by {group} (0 = rejected, 1 = approved)")
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(f1.to_html(include_plotlyjs="cdn"))
#         f.write(f2.to_html(full_html=False, include_plotlyjs=False))
#     return path
