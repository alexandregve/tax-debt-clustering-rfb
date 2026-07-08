#!/usr/bin/env python3
"""
classify_debtors.py — RFB 9ª Região Fiscal
Collections Segmentation Cockpit  |  versão kmeans_k4_20260316

Uso:
    python classify_debtors.py --input novos_devedores.csv \
                               --output resultado.csv \
                               --model deployment/model/artefacto_v20260316.pkl
"""
import argparse, pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def classificar(df, artefacto, id_col='id_cnpj'):
    feat  = artefacto['feature_names']
    sc    = artefacto['scaler']
    cents = artefacto['centroids_normalised']
    nomes = artefacto['cluster_names']
    short = artefacto['cluster_short']

    miss = [f for f in feat if f not in df.columns]
    if miss:
        raise ValueError(f"Colunas em falta: {miss}")

    X     = sc.transform(df[feat].values)
    dists = euclidean_distances(X, cents)
    ids   = dists.argmin(axis=1)

    res = pd.DataFrame()
    if id_col in df.columns:
        res[id_col] = df[id_col].values
    res['cluster_id']             = ids
    res['cluster_name']           = [nomes[c] for c in ids]
    res['cluster_short']          = [short[c]  for c in ids]
    res['dist_cluster_atribuido'] = dists[np.arange(len(dists)), ids]
    res['confianca_relativa']     = 1 - res['dist_cluster_atribuido'] / dists.sum(axis=1)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input',  required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--model',  required=True)
    ap.add_argument('--id_col', default='id_cnpj')
    args = ap.parse_args()

    with open(args.model, 'rb') as f:
        art = pickle.load(f)
    print(f"Modelo  : {art['model_version']}")
    print(f"Treino  : {art['training_date']}")

    df  = pd.read_csv(args.input)
    res = classificar(df, art, id_col=args.id_col)
    res.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"Output  : {args.output}  ({len(res):,} registos)")

    for c, name in art['cluster_names'].items():
        n = (res['cluster_id'] == c).sum()
        print(f"  C{c} {art['cluster_short'][c]}: {n:,} ({n/len(res)*100:.1f}%)")

if __name__ == '__main__':
    main()
