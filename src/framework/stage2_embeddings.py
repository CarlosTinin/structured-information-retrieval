from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .io_utils import ensure_parent_dir, read_csv_smart, save_json

try:
    from nltk.corpus import stopwords as _nltk_stopwords
    _PT_STOPWORDS = set(_nltk_stopwords.words("portuguese"))
except Exception:
    # Fallback if NLTK data not available
    _PT_STOPWORDS = {
        "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
        "as", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele",
        "deles", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em",
        "entre", "era", "essa", "essas", "esse", "esses", "esta", "estas", "este",
        "estes", "eu", "foi", "for", "foram", "há", "isso", "isto", "já", "lhe",
        "lhes", "mais", "mas", "me", "mesmo", "meu", "minha", "muito", "na",
        "nas", "não", "nem", "no", "nos", "nós", "num", "numa", "o", "os", "ou",
        "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que",
        "quem", "se", "sem", "ser", "seu", "sua", "são", "só", "também", "te",
        "tem", "tinha", "tu", "tua", "um", "uma", "uns", "você", "vos", "à", "às",
        "é",
    }


def strip_punctuation_from_text(text: str) -> str:
    """Remove all punctuation and underscores from text, keeping only alphanumeric chars and whitespace."""
    text = re.sub(r"[^\w\s]|_", " ", text, flags=re.UNICODE)
    return re.sub(r"\s{2,}", " ", text).strip()


_ROMAN_NUMERALS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
    "xxi", "xxii", "xxiii", "xxiv", "xxv",
}

# Manual blocklist: personal names, Brazilian state abbreviations, procedural
# boilerplate, and other tokens that carry no discriminative semantic signal.
_MANUAL_BLOCKLIST = {
    # State abbreviations
    "mg", "pa", "sp", "rj", "ba", "go", "mt", "ms", "pr", "rs", "sc",
    "ce", "pe", "ma", "pi", "rn", "pb", "se", "al", "to", "ac", "am",
    "ap", "ro", "rr", "es", "df",
    # Personal names / nicknames observed in SHAP attributions
    "carneiro", "dirceu", "böecker", "boecker", "dinho", "bruno", "laécio",
    "jonatas", "serafim", "correa", "miguel", "ronaldo", "vantuir",
    "marilene", "rai", "wellynton",
    # Procedural boilerplate tokens
    "reqdo", "aduz", "reqte", "pic", "rhc", "registre", "arquivem",
    "publique", "processo", "dpu", "intime", "submetê", "caput", "los",
    "fls", "classe", "trata", "procuradoria", "narra", "considerando",
    "pje", "acr", "arts", "inq", "intimem", "fundamento", "stf", "conv",
    "materialidade", "julgá", "turma", "sujeito", "região", "trf",
    "cumpra", "decido", "sumária", "certifique", "impõe", "cpp",
    "penal", "contando", "passo", "fixo", "ante", "ação",
    "sentença", "incisos", "recebida", "inc", "configuram", "enumera",
    # City names
    "tucuruí", "uruará", "altamira", "teresina", "marabá", "santarém",
    # Misc non-discriminative
    "após", "sobre", "art", "gato", "cachorro", "basicamente", "alçando",
    "onde", "patológico", "senilidade", "deliberadamente", "preenchendo",
    "água", "maior", "data", "necessariamente", "fica", "entorno",
    "gado", "larvicida", "pimp", "priva", "seus", "fogão",
    "notadamente", "verificando", "submeter", "termo", "finalmente",
    "quanto", "juiz", "informante", "contraditório", "ora", "outros",
    "motivo", "vítimas", "exposto", "albis", "machados", "remuneração",
    "fumaça", "choça", "escola", "maisgoais", "denúncia", "empregados",
    "pertinente", "trabalho", "aviltamento", "todos", "aliás",
    "iluminação", "percepção", "possível", "www", "https",
    "oportunamente", "lazer", "violência", "jornadas", "persistente",
    "sábado", "vilas", "permitindo", "efetivamente", "seria", "vinte",
    "artigos", "inciso", "gislene", "sinop", "reu", "fibrocimento",
    "descreve", "cuida", "cidade", "empreiteiro", "trabalhadores",
    "fiscalização", "utensílios", "trabalhador", "corréu",
    "desembargador", "autoria", "livre", "modo", "trabalhistas",
    "saber", "comunidade", "pessoa", "tipo", "improcedente", "engodo",
    "fraude", "segurança", "espécie", "nenhuma", "infere", "seguir",
    "jairo", "alex", "stj", "construtec", "sétima", "oitava", "nona",
    "epi", "higiene", "súmula", "público", "análoga", "aliciar",
    "analisando", "noite", "social", "cita", "carvalho",
    "miserabilidade", "mte", "diones", "acolhendo", "instrução",
    "ouvido", "citado", "prática", "comunicações", "encontra",
    "trabalhos", "exigindo", "redução", "respectivamente",
    "relatos", "passando", "pousada", "mpf", "gráfico", "josué",
    "unânimes", "previdênciarios", "morais", "pessoais", "destacando",
    "acrescenta", "tratando", "circunstâncias", "escravo", "colheita",
    "psicológica", "madeira", "instalada", "pede", "categoria",
    "jurídica", "saúde", "genericamente", "aprisionamento", "flávio",
    "rafael", "primeira", "terceira", "inaplicável", "brasileira",
    "razoabilidade", "insalubres", "coação", "patrão",
    "constrangimentos", "configura", "econômicos", "restrição",
    "intensa", "vídeo", "mediante", "imprescindível", "polícia",
    "respeitosa", "oab", "cláudio", "vara", "resposta", "ademais",
    "noiado", "atingindo", "moderna", "observa", "denunciado",
    "atribuindo", "requereu", "procedendo", "tendo", "anpp",
    "comprometer", "codefat", "etc", "investigado", "ciente",
    "estando", "excluam", "chamado", "juízo", "trabalhar", "minutos",
    "alojamento", "locomoção", "treze", "interrogado", "normas",
    "remota", "submetido", "quer", "correspondente", "lei", "reunião",
    "escolher", "fazer", "ressaltando", "cozinha", "vista", "quintal",
    "relata", "desmembramento", "ofício",
    "eulálio", "pág", "gov", "resp", "sessenta", "arenópolis",
    "duração", "contratação", "apenas", "atas", "permaneceram",
    "vir", "escravidão", "cada", "estável", "rememore", "econômico",
    "caixa", "voluntária", "operação", "prestar", "relatório",
    "seguro", "agência", "datado", "decretar", "frustrada", "lima",
    "coger", "tona", "sujeitando", "mulheres", "emprego", "fraudá",
    "tratada", "crime", "relator", "tratamento", "situação", "cnj",
    "dispositivo", "fundamentação", "ensejará", "preencher",
    "silva", "págs", "cac", "goiânia", "ctps", "esgoto", "alimentos",
    "refeição", "pagando", "afastou", "volume", "extremos", "bahia",
    "aplicam", "aplica", "orgao", "dezenove", "keyla",
    "iracy", "sifuentes", "min", "tios", "nome", "pagá", "energias",
    "pag", "ocorre", "vale", "fumacê", "pais", "boa", "expedido",
    "meses", "consciente", "prestado", "trabalhista", "comida",
    "vilela", "houve", "integralmente", "constatou",
    "feliciano", "melo", "francisco", "dje", "rse", "frango", "arroz",
    "gefm", "produção", "retirar", "enrolava", "reais", "falsificar",
    "reconheço", "conducente", "inepta", "denunciada", "instituição",
    "aplicação", "cinquenta",
    "bastando", "independente", "passou", "res", "aproximadamente",
    "cerceamento", "lenha", "meor", "revelia", "necessário", "veja",
    "salários", "conterá", "valor", "org", "ficar", "copa", "quartos",
    "notícias", "umas", "bolsa", "durante", "seguinte", "vontade",
    "empregador", "lona", "motoserras", "tratores", "bacbal",
    "atualmente", "murado", "caminhão",
    "maisgoias", "bacabal", "quo", "quarto", "sendo",
    "digno", "gratuitamente", "bem", "imperiosa", "resgatos",
    "principalmente", "causa", "politicos",
    "caxias", "respeito", "alimentação", "mera", "outro", "outra",
    "criaco", "feira", "insetos",
    # Written-out numbers
    "um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito",
    "nove", "dez", "trinta", "quarta", "duzentos", "cem", "setenta",
    "dezesseis",
    # Conjunctions
    "porém", "contudo", "embora", "alternativamente", "todavia", "entretanto",
    "conquanto", "posto", "malgrado", "ainda", "portanto", "logo", "pois",
    "assim", "então", "porque", "porquanto", "caso", "desde", "conforme",
    "segundo", "consoante", "enquanto", "quando", "embora", "conquanto",
}


def strip_stopwords_from_text(text: str) -> str:
    """Remove Portuguese stopwords, short tokens (<=2 chars), Roman numerals,
    alphanumeric mixed tokens, and manually blocklisted names/abbreviations."""
    tokens = text.split()
    filtered = []
    for t in tokens:
        t_lower = t.lower()
        # Remove stopwords
        if t_lower in _PT_STOPWORDS:
            continue
        # Remove tokens with 2 or fewer characters
        if len(t) <= 2:
            continue
        # Remove Roman numerals
        if t_lower in _ROMAN_NUMERALS:
            continue
        # Remove manual blocklist (names, state abbreviations)
        if t_lower in _MANUAL_BLOCKLIST:
            continue
        # Remove tokens that mix letters and digits (e.g., 'ma124', 'mg33693')
        has_alpha = any(ch.isalpha() for ch in t)
        has_digit = any(ch.isdigit() for ch in t)
        if has_alpha and has_digit:
            continue
        filtered.append(t)
    return " ".join(filtered)


def strip_numbers_from_text(text: str) -> str:
    """Remove purely numeric tokens from text."""
    tokens = text.split()
    filtered = [t for t in tokens if not re.fullmatch(r"\d+", t)]
    return " ".join(filtered)

try:
    import xgboost as xgb
except Exception:
    xgb = None


def load_dataset(input_csv: str, text_column: str, label_column: str) -> pd.DataFrame:
    df = read_csv_smart(input_csv)
    for col in (text_column, label_column):
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória não encontrada: {col}")

    df = df[[text_column, label_column]].copy()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna()
    df = df[df["text"].str.len() > 10]
    df = df[df["label"].str.len() > 0]
    return df


def get_models(seed: int) -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "SVM (Linear)": SVC(kernel="linear", random_state=seed, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            random_state=seed,
            min_samples_split=7,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    else:
        print("[Aviso] XGBoost indisponível no ambiente (libomp ausente). Modelo será ignorado.")

    return models


def run_stage2_embeddings(
    input_csv: str,
    output_root: str = "output",
    text_column: str = "texto_normalizado",
    label_column: str = "decisao",
    model_name: str = "dominguesm/legal-bert-base-cased-ptbr",
    k_folds: int = 3,
    seed: int = 42,
    target_labels: tuple[str, ...] = ("condenação", "extinto", "absolvição"),
    strip_punctuation: bool = False,
    strip_stopwords: bool = False,
    strip_numbers: bool = False,
) -> dict:
    np.random.seed(seed)

    df = load_dataset(input_csv, text_column=text_column, label_column=label_column)

    if target_labels:
        target_set = {x.strip().lower() for x in target_labels}
        before = len(df)
        df = df[df["label"].isin(target_set)].copy()
        print(f"Filtrando classes-alvo: {sorted(target_set)} | {before} -> {len(df)} documentos")

    if strip_punctuation:
        print("[Ablation] Removendo pontuação dos textos antes da geração de embeddings...")
        df["text"] = df["text"].apply(strip_punctuation_from_text)
        df = df[df["text"].str.len() > 10]  # re-filter after stripping

    if strip_stopwords:
        print("[Ablation] Removendo stopwords dos textos antes da geração de embeddings...")
        df["text"] = df["text"].apply(strip_stopwords_from_text)
        df = df[df["text"].str.len() > 10]

    if strip_numbers:
        print("[Ablation] Removendo tokens numéricos dos textos antes da geração de embeddings...")
        df["text"] = df["text"].apply(strip_numbers_from_text)
        df = df[df["text"].str.len() > 10]

    class_counts = df["label"].value_counts()
    if class_counts.empty:
        raise ValueError("Nenhum documento disponível após filtragem de classes-alvo.")

    min_class_size = int(class_counts.min())
    if min_class_size < 2:
        raise ValueError(
            "Pelo menos uma classe tem menos de 2 exemplos após filtragem. "
            "Não é possível executar validação cruzada estratificada."
        )

    if k_folds > min_class_size:
        print(f"[Aviso] Ajustando k_folds de {k_folds} para {min_class_size} (classe minoritária).")
        k_folds = min_class_size

    encoder = LabelEncoder()
    df["label_encoded"] = encoder.fit_transform(df["label"])

    embedder = SentenceTransformer(model_name)
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    by_model = {name: [] for name in get_models(seed)}
    aggregate = {name: {"preds": [], "trues": []} for name in get_models(seed)}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(df, df["label_encoded"]), start=1):
        print(f"\n--- Fold {fold}/{k_folds} ---")
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        x_train = embedder.encode(train_df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True)
        x_test = embedder.encode(test_df["text"].tolist(), convert_to_numpy=True, show_progress_bar=True)
        y_train = train_df["label_encoded"].values
        y_test = test_df["label_encoded"].values

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        for name, model in get_models(seed).items():
            try:
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                metrics = {
                    "fold": fold,
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                }
                by_model[name].append(metrics)
                aggregate[name]["preds"].extend(np.asarray(y_pred).tolist())
                aggregate[name]["trues"].extend(np.asarray(y_test).tolist())
            except Exception as exc:
                print(f"[Aviso] Modelo {name} falhou no fold {fold}: {exc}")

    report = {
        "embedding_model": model_name,
        "strip_punctuation": strip_punctuation,
        "strip_stopwords": strip_stopwords,
        "strip_numbers": strip_numbers,
        "classes": encoder.classes_.tolist(),
        "models": {},
    }

    for name, metrics in by_model.items():
        if not metrics:
            continue
        f1_values = [m["f1"] for m in metrics]
        report["models"][name] = {
            "fold_metrics": metrics,
            "mean_f1": float(np.mean(f1_values)),
            "std_f1": float(np.std(f1_values)),
            "classification_report": classification_report(
                aggregate[name]["trues"],
                aggregate[name]["preds"],
                target_names=encoder.classes_,
                output_dict=True,
                zero_division=0,
            ),
        }

    # Resumo para terminal e tabela
    rows = []
    for name, metrics in by_model.items():
        if not metrics:
            continue
        rows.append(
            {
                "Modelo": name,
                "Accuracy": float(np.mean([m["accuracy"] for m in metrics])),
                "Precision": float(np.mean([m["precision"] for m in metrics])),
                "Recall": float(np.mean([m["recall"] for m in metrics])),
                "F1": float(np.mean([m["f1"] for m in metrics])),
            }
        )
    if not rows:
        raise RuntimeError("Nenhum modelo concluiu avaliação com sucesso.")
    metrics_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    print("\nResumo de desempenho (média dos folds):")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Exportar tabela LaTeX
    suffix = ""
    if strip_punctuation:
        suffix += "_no_punct"
    if strip_stopwords:
        suffix += "_no_stop"
    if strip_numbers:
        suffix += "_no_num"
    table_path = Path(output_root) / "tables" / f"table{suffix}.tex"
    ensure_parent_dir(table_path)
    latex_df = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        latex_df[col] = latex_df[col].map(lambda x: f"{x:.4f}")
    latex_table = latex_df.to_latex(index=False, escape=True)
    table_path.write_text(latex_table, encoding="utf-8")

    # Exportar matrizes de confusão por modelo
    images_dir = Path(output_root) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for name in by_model.keys():
        if not by_model[name]:
            continue
        y_true = aggregate[name]["trues"]
        y_pred = aggregate[name]["preds"]
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_,
            cbar=False,
        )
        plt.title(f"Matriz de Confusão - {name}")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.tight_layout()

        model_slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        image_path = images_dir / f"matriz_confusao_{model_slug}{suffix}.png"
        plt.savefig(image_path, dpi=200)
        plt.close()

    output_path = Path(output_root) / f"stage2_embeddings_results{suffix}.json"
    ensure_parent_dir(output_path)
    save_json(report, output_path)
    print(f"\nResultados salvos em: {output_path}")
    print(f"Tabela LaTeX salva em: {table_path}")
    print(f"Imagens salvas em: {images_dir}")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Etapa 2 (embeddings) - BERT embeddings + classificadores")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--text-column", default="texto_normalizado")
    parser.add_argument("--label-column", default="decisao")
    parser.add_argument("--model-name", default="dominguesm/legal-bert-base-cased-ptbr")
    parser.add_argument("--k-folds", type=int, default=3)
    parser.add_argument("--target-labels", default="condenação,extinto,absolvição")
    parser.add_argument("--strip-punctuation", action="store_true", default=False,
                        help="Remove punctuation before embedding (ablation study)")
    parser.add_argument("--strip-stopwords", action="store_true", default=False,
                        help="Remove Portuguese stopwords before embedding (ablation study)")
    parser.add_argument("--strip-numbers", action="store_true", default=False,
                        help="Remove purely numeric tokens before embedding (ablation study)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    target_labels = tuple(x.strip() for x in args.target_labels.split(",") if x.strip())
    run_stage2_embeddings(
        input_csv=args.input,
        output_root=args.output_root,
        text_column=args.text_column,
        label_column=args.label_column,
        model_name=args.model_name,
        k_folds=args.k_folds,
        target_labels=target_labels,
        strip_punctuation=args.strip_punctuation,
        strip_stopwords=args.strip_stopwords,
        strip_numbers=args.strip_numbers,
    )


if __name__ == "__main__":
    main()
