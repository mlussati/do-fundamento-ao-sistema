# Do Fundamento ao Sistema

Código da série **Do Fundamento ao Sistema**, publicada em [mlussati.substack.com](https://mlussati.substack.com).

Cada episódio pega um fundamento da computação (algoritmo, estrutura de dados, sistemas distribuídos) e mostra onde ele aparece num sistema de IA real. O código é pequeno, roda na sua máquina e usa o mínimo de dependências: a ideia é ver o mecanismo, não usar uma biblioteca pronta.

| # | Episódio | Código |
|---|---|---|
| 0 | [Construí um Vector Store do Zero para Entender RAG de Verdade](https://mlussati.substack.com/p/construi-um-vector-store-do-zero) | (no post) |
| 1 | Busca aproximada e por que o seu banco vetorial não compara com tudo | [`01-busca-aproximada/`](01-busca-aproximada/) |

## Como rodar

Requer Python 3.10+ e NumPy.

```bash
pip install numpy
python3 01-busca-aproximada/busca_aproximada.py
python3 01-busca-aproximada/busca_aproximada.py --dim 768   # mais perto de um embedding real
```

Os números mudam com a máquina e com os parâmetros; o que importa é a forma da curva entre recall e distâncias calculadas.

## Autor

Manilson Lussati · [Substack](https://substack.com/@mlussati) · [LinkedIn](https://www.linkedin.com/in/manilson-lussati/)
