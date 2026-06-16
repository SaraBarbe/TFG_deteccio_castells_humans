import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


#Drectoris
csv_path = r"C:\Users\34639\OneDrive\Escriptori\TFG\RESULTS\HIBRID\TEST\results_test_FINAL3.csv"

output_dir = r"C:\Users\34639\OneDrive\Escriptori\TFG\RESULTS\HIBRID\TEST\EVALUATION_FINAL_NS"

# Crear carpeta si no existeix
os.makedirs(output_dir, exist_ok=True)


#Funcions

def extract_amplada(castell):
    match = re.match(r"(\d+)d", str(castell))
    return match.group(1) if match else "?"

def extract_alcada(castell):
    match = re.search(r"d(\d+)", str(castell))
    return match.group(1) if match else "?"

AMPLADES_VALIDADES = {"2", "3", "4", "5", "7"}

def agrupar_amplades_invalides(valor):

    valor = str(valor)

    if valor in AMPLADES_VALIDADES:
        return valor

    return "Invàlida"


#classes vàlides 

CLASSES_VALIDES = {
    "2d6", "2d7", "2d7f", "2d8", "2d8f",
    "3d6", "3d7", "3d8", "3d8f", "3d9f",
    "4d6", "4d7", "4d8", "4d9f",
    "5d6", "5d7", "5d8",
    "7d6", "7d7", "7d8"
}


def agrupar_castells_invalids(castell):

    castell = str(castell)

    if castell in CLASSES_VALIDES:
        return castell

    return "Invàlid"

def plot_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename,
    rows=None,
    cols=None
):

    if rows is None:
        rows = sorted(list(set(y_true)))

    if cols is None:
        cols = sorted(list(set(y_pred)))

    cm = pd.crosstab(
        y_true,
        y_pred,
        normalize="index"
    )

    cm = cm.reindex(
        index=rows,
        columns=cols,
        fill_value=0
    )

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        annot_kws={"size": 14}
    )

    plt.xlabel("Predicció")
    plt.ylabel("Valor real")
    plt.title(title, fontsize=16)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300
    )

    plt.close()

def plot_confusion_matrix_amplada(
    y_true,
    y_pred,
    title,
    filename,
    rows=None,
    cols=None
):

    if rows is None:
        rows = sorted(list(set(y_true)))

    if cols is None:
        cols = sorted(list(set(y_pred)))

    cm = pd.crosstab(
        y_true,
        y_pred,
        normalize="index"
    )

    cm = cm.reindex(
        index=rows,
        columns=cols,
        fill_value=0
    )

    plt.figure(figsize=(8,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues"
    )

    plt.xlabel("Predicció")
    plt.ylabel("Valor real")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300
    )

    plt.close()


#carreguem el csv
df = pd.read_csv(csv_path, encoding="cp1252")

# resultats globals
y_true = (
    df["castell_real"]
    .apply(agrupar_castells_invalids)
)

y_pred = (
    df["castell_pred"]
    .apply(agrupar_castells_invalids)
)

#Amplada i açada global

# Amplada real
y_true_amplada = (
    y_true
    .apply(extract_amplada)
    .apply(agrupar_amplades_invalides)
)

# Amplada global final
y_pred_amplada_global = (
    df["castell_pred"]
    .apply(extract_amplada)
    .apply(agrupar_amplades_invalides)
)

# Amplada clustering
y_pred_clustering = (
    df["amplada_clustering"]
    .fillna(-1)
    .astype(int)
    .astype(str)
)

# Amplada classifier
y_pred_classifier = (
    df["amplada_classificador"]
    .fillna(-1)
    .astype(int)
    .astype(str)
    .apply(agrupar_amplades_invalides)
)

# Alçada real
y_true_alcada = (
    df["castell_real"]
    .apply(extract_alcada)
)

# Alçada predita
y_pred_alcada = (
    df["castell_pred"]
    .apply(extract_alcada)
)

# qualsevol valor no vàlid -> Invàlida
y_pred_alcada = y_pred_alcada.apply(
    lambda x: x if x in ["6", "7", "8", "9"] else "Invàlida"
)

#accuracies

accuracy_global = accuracy_score(
    y_true,
    y_pred
)

accuracy_amplada_global = accuracy_score(
    y_true_amplada,
    y_pred_amplada_global
)

accuracy_clustering = accuracy_score(
    y_true_amplada,
    y_pred_clustering
)

accuracy_classifier = accuracy_score(
    y_true_amplada,
    y_pred_classifier
)

accuracy_alcada = accuracy_score(
    y_true_alcada,
    y_pred_alcada
)

print(f"\nAccuracy global: {accuracy_global*100:.2f}%")
print(f"Accuracy amplada global: {accuracy_amplada_global*100:.2f}%")
print(f"Accuracy clustering: {accuracy_clustering*100:.2f}%")
print(f"Accuracy classifier: {accuracy_classifier*100:.2f}%")
print(f"Accuracy alçada: {accuracy_alcada*100:.2f}%")


with open(
    os.path.join(output_dir, "accuracy.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write(f"Accuracy global: {accuracy_global*100:.2f}%\n")
    f.write(f"Accuracy amplada global: {accuracy_amplada_global*100:.2f}%\n")
    f.write(f"Accuracy clustering: {accuracy_clustering*100:.2f}%\n")
    f.write(f"Accuracy classifier: {accuracy_classifier*100:.2f}%\n")
    f.write(f"Accuracy alçada: {accuracy_alcada*100:.2f}%\n")

#classification

report = classification_report(
    y_true,
    y_pred
)

print("\nClassification Report:\n")
print(report)

with open(
    os.path.join(output_dir, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

#confusion matrix globals

classes_globals = [
    "2d6", "2d7", "2d7f", "2d8", "2d8f",
    "3d6", "3d7", "3d8", "3d8f", "3d9f",
    "4d6", "4d7", "4d8", "4d9f",
    "5d6", "5d7", "5d8",
    "7d6", "7d7", "7d8"
]


plot_confusion_matrix(
    y_true,
    y_pred,
    "Confusion Matrix Global",
    "confusion_matrix_global.png",
    rows=classes_globals,
    cols=classes_globals + ["Invàlid"]
)

#amplada global

plot_confusion_matrix_amplada(
    y_true_amplada,
    y_pred_amplada_global,
    "Confusion Matrix Amplada Global",
    "confusion_matrix_amplada_global.png",
    rows=["2","3","4","5","7","Invalida"],
    cols=["2","3","4","5","7","Invalida"]
)

#clustering global
cols_clustering = sorted(
    list(
        set(
            y_pred_clustering.astype(str)
        )
    ),
    key=str
)


plot_confusion_matrix_amplada(
    y_true_amplada,
    y_pred_clustering,
    "Confusion Matrix Clustering Global",
    "confusion_matrix_clustering_global.png",
    rows=["2","3","4","5","7"],
    cols=["2","3","4","5","7","Invàlida"]
)

#classifier global
plot_confusion_matrix_amplada(
    y_true_amplada,
    y_pred_classifier,
    "Confusion Matrix Classifier Global",
    "confusion_matrix_classifier_global.png"
)

#alçaga global
plot_confusion_matrix(
    y_true_alcada,
    y_pred_alcada,
    "Confusion Matrix Alçada Global",
    "confusion_matrix_alcada_global.png",
    rows=["6","7","8","9"],
    cols=["6","7","8","9"]
)

#comparació pisos troncs

for pisos in [2, 3, 4]:

    # Filtrar dades
    classes_per_pisos = {
        2: ["2d6", "2d7f", "3d6", "4d6", "5d6", "7d6"],
        3: ["2d7", "2d8f", "3d7", "3d8f", "4d7", "5d7", "7d7"],
        4: ["2d8", "3d8", "3d9f", "4d8", "4d9f", "5d8", "7d8"]
    }

    df_subset = df[
        df["castell_real"].isin(classes_per_pisos[pisos])
    ]

    # Saltar si no hi ha dades
    if len(df_subset) == 0:
        continue

    # Ground truth
    y_true_subset = (
    df_subset["castell_real"]
    .apply(extract_amplada)
    .apply(agrupar_amplades_invalides)
)
    #clustering

    y_pred_clustering_subset = (
        df_subset["amplada_clustering"]
        .fillna(-1)
        .astype(int)
        .astype(str)
        .apply(agrupar_amplades_invalides)
    )

    acc_clustering_subset = accuracy_score(
        y_true_subset,
        y_pred_clustering_subset
    )

    print(
        f"\nAccuracy clustering "
        f"pisos_tronc={pisos}: "
        f"{acc_clustering_subset*100:.2f}%"
    )


    cm = pd.crosstab(
    y_true_subset,
    y_pred_clustering_subset
    )

    print(
        "Accuracy des de CM:",
        cm.values.diagonal().sum() / cm.values.sum()
    )

    plot_confusion_matrix_amplada(
    y_true_subset,
    y_pred_clustering_subset,
    f"Confusion Matrix Mòdul Detecció {pisos} Pisos",
    f"confusion_matrix_clustering_{pisos}.png",
    rows=["2","3","4","5","7"],
    cols=["2","3","4","5","7","Invàlida"]
    )

    #classifier

    y_pred_classifier_subset = (
    df_subset["amplada_classificador"]
    .fillna(-1)
    .astype(int)
    .astype(str)
    .apply(agrupar_amplades_invalides)
)

    acc_classifier_subset = accuracy_score(
        y_true_subset,
        y_pred_classifier_subset
    )

    print(
        f"Accuracy classifier "
        f"pisos_tronc={pisos}: "
        f"{acc_classifier_subset*100:.2f}%"
    )

    plot_confusion_matrix_amplada(
    y_true_subset,
    y_pred_classifier_subset,
    f"Confusion Matrix Mòdul Classificador {pisos} Pisos",
    f"confusion_matrix_classifier_{pisos}.png",
    rows=["2","3","4","5","7"],
    cols=["2","3","4","5","7","Invalida"]
    )

print("\nConfusion matrices guardades correctament!")

for pisos in [2, 3, 4]:

    # Filtrar dades
    classes_per_pisos = {
    2: ["2d6", "2d7f", "3d6", "4d6", "5d6", "7d6"],
    3: ["2d7", "2d8f", "3d7", "3d8f", "4d7", "5d7", "7d7"],
    4: ["2d8", "3d8", "3d9f", "4d8", "4d9f", "5d8", "7d8"]
}

    df_subset = df[
        df["castell_real"].isin(classes_per_pisos[pisos])
    ]

    # Saltar si no hi ha dades
    if len(df_subset) == 0:
        continue

    y_true_subset = (
        df_subset["castell_real"]
        .apply(agrupar_castells_invalids)
    )

    y_pred_subset = (
        df_subset["castell_pred"]
        .apply(agrupar_castells_invalids)
    )

    # Accuracy
    acc_subset = accuracy_score(
        y_true_subset,
        y_pred_subset
    )

    print(
        f"\nAccuracy híbrid "
        f"pisos_tronc={pisos}: "
        f"{acc_subset*100:.2f}%"
    )

    cols_subset = sorted(
        [
            c for c in set(y_pred_subset)
            if c != "Invàlid"
        ]
    )

    rows_subset = sorted(
        [
            c for c in set(y_true_subset)
            if c != "Invàlid"
        ]
    )

    plot_confusion_matrix(
        y_true_subset,
        y_pred_subset,
        f"Confusion Matrix Sistema Híbrid {pisos} Pisos",
        f"confusion_matrix_hibrid_{pisos}.png",
        rows=rows_subset,
        cols=cols_subset
    )

    y_true_amplada_subset = (
    df_subset["castell_real"]
    .apply(extract_amplada)
    .apply(agrupar_amplades_invalides)
    )

    #amplada híbrid

    y_true_amplada_subset = pd.Series(
        df_subset["castell_real"]
        .apply(extract_amplada)
        .apply(agrupar_amplades_invalides)
    ).reset_index(drop=True)

    y_pred_hibrid_subset = []

    for _, row in df_subset.iterrows():

        amplada_det = int(row["amplada_clustering"])

        if pd.isna(row["amplada_classificador"]):
            amplada_cls = -1
        else:
            amplada_cls = int(row["amplada_classificador"])

        amplada_hibrida = max(amplada_det, amplada_cls)

        y_pred_hibrid_subset.append(
            agrupar_amplades_invalides(
                str(amplada_hibrida)
            )
        )

    y_pred_hibrid_subset = pd.Series(
    y_pred_hibrid_subset
    ).reset_index(drop=True)

    acc_hibrid_subset = accuracy_score(
    y_true_amplada_subset,
    y_pred_hibrid_subset
    )

    print(
    f"Accuracy amplada híbrid "
    f"pisos_tronc={pisos}: "
    f"{acc_hibrid_subset*100:.2f}%"
    )

    plot_confusion_matrix_amplada(
        y_true_amplada_subset,
        y_pred_hibrid_subset,
        f"Confusion Matrix Sistema Combinat {pisos} Pisos",
        f"confusion_matrix_Sistema_Combinat_{pisos}.png",
        rows=["2","3","4","5","7"],
        cols=["2","3","4","5","7","Invàlida"]
    )