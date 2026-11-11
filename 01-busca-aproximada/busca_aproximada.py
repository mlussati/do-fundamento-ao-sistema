"""
Do Fundamento ao Sistema #1: busca aproximada.

Três peças, só com NumPy e a biblioteca padrão:
  (a) força bruta: compara a pergunta com todos os N vetores (a verdade-terreno);
  (b) um grafo de vizinhos simples (cada vetor ligado aos seus M mais próximos)
      e uma busca gulosa com lista de candidatos de tamanho ef;
  (c) recall@10 da busca no grafo contra a força bruta, e quantas distâncias
      cada abordagem precisou calcular.

Não é HNSW: não tem camadas nem heurística de seleção de vizinhos.
É a camada 0 dele, da forma mais crua possível, para ver o mecanismo.

Uso:
    python3 busca_aproximada.py
    python3 busca_aproximada.py --n 20000 --dim 64 --m 16
"""

import argparse
import heapq
import time

import numpy as np


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def gerar_vetores(n, dim, rng):
    """Vetores aleatórios normalizados (norma 1), como embeddings prontos para cosseno."""
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


# ---------------------------------------------------------------------------
# (a) Força bruta
# ---------------------------------------------------------------------------

def forca_bruta(base, consulta, k):
    """Similaridade com todos os vetores, devolve os índices dos k mais parecidos.
    Calcula exatamente len(base) distâncias."""
    scores = base @ consulta                      # produto escalar = cosseno (vetores normalizados)
    top = np.argpartition(-scores, k)[:k]         # os k maiores, fora de ordem
    return top[np.argsort(-scores[top])], len(base)


# ---------------------------------------------------------------------------
# (b) Grafo de vizinhos + busca gulosa com lista de candidatos (ef)
# ---------------------------------------------------------------------------

def construir_grafo_knn(base, m, bloco=1024):
    """Liga cada vetor aos seus m vizinhos mais próximos (grafo kNN).
    A construção aqui usa força bruta em blocos: é cara (N x N), e tudo bem,
    o ponto do exemplo é a busca, não a construção."""
    n = len(base)
    vizinhos = np.empty((n, m), dtype=np.int64)
    for ini in range(0, n, bloco):
        fim = min(ini + bloco, n)
        scores = base[ini:fim] @ base.T
        scores[np.arange(fim - ini), np.arange(ini, fim)] = -np.inf   # ninguém é vizinho de si mesmo
        top = np.argpartition(-scores, m, axis=1)[:, :m]
        vizinhos[ini:fim] = top
    # Torna as arestas de ida e volta: se A aponta para B, B também aponta para A.
    # Sem isso, alguns nós ficam inalcançáveis.
    adj = [set(linha.tolist()) for linha in vizinhos]
    for a in range(n):
        for b in vizinhos[a]:
            adj[b].add(a)
    return [np.fromiter(s, dtype=np.int64) for s in adj]


def busca_no_grafo(base, grafo, consulta, k, ef, entrada):
    """Busca gulosa com lista de candidatos, no espírito do search_layer do HNSW.

    - candidatos: quem ainda vale a pena expandir (o mais perto primeiro)
    - resultados: os ef melhores vistos até agora
    Para quando o melhor candidato restante já está mais longe que o pior
    dos resultados. Com ef=1 é a busca gulosa pura: segue um caminho só
    e para no primeiro mínimo local.
    Devolve (k melhores índices, nº de distâncias calculadas)."""
    def dist(i):
        return 1.0 - float(base[i] @ consulta)    # distância de cosseno

    calculadas = 1
    d0 = dist(entrada)
    visitados = {entrada}
    candidatos = [(d0, entrada)]                  # heap mínimo por distância
    resultados = [(-d0, entrada)]                 # heap máximo (distância negativa)

    while candidatos:
        d_c, c = heapq.heappop(candidatos)
        pior = -resultados[0][0]
        if d_c > pior:
            break                                 # nada aqui pode melhorar a lista
        for v in grafo[c]:
            v = int(v)
            if v in visitados:
                continue
            visitados.add(v)
            d_v = dist(v)
            calculadas += 1
            pior = -resultados[0][0]
            if len(resultados) < ef or d_v < pior:
                heapq.heappush(candidatos, (d_v, v))
                heapq.heappush(resultados, (-d_v, v))
                if len(resultados) > ef:
                    heapq.heappop(resultados)

    melhores = sorted((-d, i) for d, i in resultados)
    return [i for _, i in melhores[:k]], calculadas


# ---------------------------------------------------------------------------
# (c) Recall@k contra a força bruta
# ---------------------------------------------------------------------------

def recall_at_k(aproximado, exato, k):
    """Dos k vizinhos verdadeiros, quantos a busca aproximada trouxe."""
    return len(set(aproximado[:k]) & set(exato[:k])) / k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000, help="vetores na base")
    p.add_argument("--dim", type=int, default=32, help="dimensão")
    p.add_argument("--m", type=int, default=16, help="vizinhos por nó no grafo")
    p.add_argument("--consultas", type=int, default=200)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--semente", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.semente)
    base = gerar_vetores(args.n, args.dim, rng)
    consultas = gerar_vetores(args.consultas, args.dim, rng)
    k = args.k

    print(f"base: {args.n} vetores, dim {args.dim} | grafo: m={args.m} | "
          f"{args.consultas} consultas | k={k} | semente {args.semente}\n")

    # (a) verdade-terreno
    t0 = time.perf_counter()
    exatos = [forca_bruta(base, q, k)[0] for q in consultas]
    t_fb = (time.perf_counter() - t0) / len(consultas) * 1000
    print(f"força bruta: recall@{k} = 1.000 | distâncias/consulta = {args.n} "
          f"| {t_fb:.3f} ms/consulta")

    # (b) grafo
    t0 = time.perf_counter()
    grafo = construir_grafo_knn(base, args.m)
    grau_medio = sum(len(g) for g in grafo) / len(grafo)
    print(f"grafo construído em {time.perf_counter() - t0:.1f} s "
          f"(grau médio {grau_medio:.1f})\n")

    # entrada fixa para todas as consultas, como o ponto de entrada de um índice
    entrada = int(rng.integers(args.n))

    # Busca gulosa pura (ef=1): acha o vizinho mais próximo de verdade?
    acertos, dists = 0, 0
    for q, ex in zip(consultas, exatos):
        achado, calc = busca_no_grafo(base, grafo, q, 1, 1, entrada)
        acertos += int(achado[0] == ex[0])
        dists += calc
    print(f"gulosa pura (ef=1): achou o vizinho mais próximo verdadeiro em "
          f"{acertos}/{len(consultas)} consultas | "
          f"distâncias/consulta = {dists / len(consultas):.0f}\n")

    # (c) recall@k variando ef (ef >= k, senão a lista não cabe k resultados)
    print(f"{'ef':>5} | {'recall@' + str(k):>10} | {'distâncias/consulta':>20} | "
          f"{'% da base':>9} | {'ms/consulta':>11}")
    print("-" * 68)
    for ef in (10, 20, 40, 80, 160, 320):
        rec, dists = 0.0, 0
        t0 = time.perf_counter()
        for q, ex in zip(consultas, exatos):
            achado, calc = busca_no_grafo(base, grafo, q, k, ef, entrada)
            rec += recall_at_k(achado, ex, k)
            dists += calc
        ms = (time.perf_counter() - t0) / len(consultas) * 1000
        media = dists / len(consultas)
        print(f"{ef:>5} | {rec / len(consultas):>10.3f} | {media:>20.0f} | "
              f"{100 * media / args.n:>8.1f}% | {ms:>11.3f}")

    print("\nObs.: o tempo em ms da busca no grafo é Python puro e não se compara "
          "com o da força bruta (NumPy vetorizado). Compare as distâncias.")


if __name__ == "__main__":
    main()
