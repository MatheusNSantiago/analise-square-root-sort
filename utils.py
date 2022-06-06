from tqdm import trange
from rich.console import Console
from rich.table import Table
import numpy as np


console = Console()


def rprint(text="", style=None):
    print()
    console.print(text, style=style)


def show_results_table(resultados: dict):
    quadratico, heap = resultados.values()

    # repeticoes = len(list(quadratico.values())[0])

    # print(tuple(zip(quadratico.items())))
    # for size, tempos in quadratico.items():
        # print(size, tempos)

    sizes, tempos = zip(*quadratico.items())

    table = Table(
        title="Ordenacão por Selecão de Raiz Quadrada - Média de tempo (em [italic]s[/italic])",
        title_style="bold white",
        min_width=70,
        show_lines=True,
        caption=f"Média de {len(tempos[0])} repetições para cada tamanho de array",
    )
    to_exp = lambda x: f"10^{int(np.log10(x))}"

    table.add_column("Tamanhos", justify="center", style="cyan")
    table.add_column("Insertion Sort", style="magenta", justify="center")
    table.add_column("Heap", style="green", justify="center")

    for size, tempos in zip(sizes, tempos):
        media = np.round(np.mean(tempos), 3)
        table.add_row(str(to_exp(size)), str(media), "0.2")
    # table.add_row("10^5", "1.12", "1.23")
    # table.add_row("10^6", "12.2", "12.3")
    # table.add_row("10^7", "40.2", "24.3")

    console.print(table, new_line_start=True)


# show_results_table(1)


def progress_bar(
    iterable,
    size,
    bar_format="{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{rate_inv_fmt}]  ",
):
    return trange(
        iterable,
        unit=" ordenação",
        desc=f"  Para n = 10^{int(np.log10(size))}",
        bar_format=bar_format,
    )
