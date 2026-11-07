import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent  # папка, где лежит app.py, а не текущая директория
MODEL_PATH = APP_DIR / "model.joblib"
CONFIG_PATH = APP_DIR / "config.json"
TEST_PATH = APP_DIR / "test.csv"

# Почему для модели cache_resource, а для таблиц cache_data (задание 4.5):
# - st.cache_resource хранит ОДИН объект на весь сервер и отдаёт всем сессиям ссылку на него.
#   Модель мы только читаем (predict_proba), копировать её не нужно, а загружать заново дорого.
# - st.cache_data хранит результат в сериализованном виде и каждому вызову отдаёт КОПИЮ.
#   Для таблиц и массивов это правильно: если кто-то изменит DataFrame (например, добавит
#   столбец), кэш не испортится. Для модели копия значила бы сборку объекта из байтов
#   при каждом перезапуске скрипта, то есть почти то же самое, что читать файл заново.
# - Без декоратора joblib.load выполнялся бы при КАЖДОМ действии пользователя
#   (сдвинули слайдер -> скрипт сверху вниз -> модель снова читается с диска): медленно и лишняя память.


@st.cache_resource  # один объект модели на весь сервер
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data  # таблица читается один раз, каждый вызов получает копию
def load_test():
    return pd.read_csv(TEST_PATH)


@st.cache_data
def load_threshold():
    return json.loads(CONFIG_PATH.read_text())["threshold"]


@st.cache_data
def test_proba(_model, X):
    # _model с подчёркиванием: Streamlit не хеширует модель, ключ кэша - только X
    return _model.predict_proba(X)[:, 1]


for path in [MODEL_PATH, CONFIG_PATH, TEST_PATH]:
    if not path.exists():
        st.error(f"Нет файла {path.name}. Сначала выполните ячейку сохранения модели в ноутбуке.")
        st.stop()

model = load_model()
test = load_test()
features = list(model.feature_names_in_)  # колонки, на которых модель училась
X_test = test[features]
y_test = test["target"].to_numpy()
proba_test = test_proba(model, X_test)  # считается один раз, от порога не зависит

st.title("Хорошее ли вино?")
st.caption("Логистическая регрессия на данных Wine Quality: отбираем вина в подборку для подписчиков.")

# порог в session_state по key: его видят все вкладки, в том числе пакетное предсказание
if "threshold" not in st.session_state:
    st.session_state.threshold = float(load_threshold())
threshold = st.sidebar.slider("Порог «хорошее вино»", 0.0, 1.0, step=0.01, key="threshold")
st.sidebar.caption(f"Порог по умолчанию из config.json: {load_threshold():.2f}")

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ["Одно вино", "Модельные метрики", "Файл с винами", "Что выучила модель"])

# ---------- 4.1 одно вино ----------
with tab_one:
    st.subheader("Оценка одного вина")
    stats = X_test.describe()  # min, max, медиана (50%) по test.csv
    values = {}
    with st.form("wine"):
        left, right = st.columns(2)
        num_features = [f for f in features if f != "is_red"]
        for i, col in enumerate(num_features):
            with (left if i % 2 == 0 else right):
                values[col] = st.number_input(
                    col,
                    min_value=float(stats.loc["min", col]),
                    max_value=float(stats.loc["max", col]),
                    value=float(stats.loc["50%", col]),
                    format="%.4f" if stats.loc["max", col] < 2 else "%.2f",
                )
        color = st.radio("Цвет вина", ["белое", "красное"], horizontal=True)
        values["is_red"] = 1 if color == "красное" else 0
        submitted = st.form_submit_button("Оценить")

    if submitted:
        wine = pd.DataFrame([values])[features]  # тот же порядок колонок, что при обучении
        proba = model.predict_proba(wine)[0, 1]
        st.metric("Вероятность, что вино хорошее", f"{proba:.1%}")
        st.progress(float(proba))
        if proba >= threshold:
            st.success(f"Берём в подборку (вероятность ≥ порога {threshold:.2f}).")
        else:
            st.error(f"Не берём в подборку (вероятность < порога {threshold:.2f}).")

# ---------- 4.2 метрики на test ----------
with tab_metrics:
    st.subheader("Качество модели на тестовой выборке")
    pred = (proba_test >= threshold).astype(int)  # от порога зависит только это сравнение

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{metrics.precision_score(y_test, pred, zero_division=0):.3f}")
    c2.metric("Recall", f"{metrics.recall_score(y_test, pred, zero_division=0):.3f}")
    c3.metric("F1", f"{metrics.f1_score(y_test, pred, zero_division=0):.3f}")

    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=["не хорошее", "хорошее"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Матрица ошибок, порог {threshold:.2f}")
    st.pyplot(fig)

    # точка текущего порога на кривых
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr_point = fp / (fp + tn)
    prec_point = tp / (tp + fp) if tp + fp > 0 else 1.0

    fpr, tpr_curve, _ = metrics.roc_curve(y_test, proba_test)
    prec, rec, _ = metrics.precision_recall_curve(y_test, proba_test)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr_curve, label=f"ROC-AUC = {metrics.roc_auc_score(y_test, proba_test):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].scatter(fpr_point, tpr, color="red", zorder=3, label=f"порог {threshold:.2f}")
    axes[0].set(xlabel="FPR", ylabel="TPR (recall)", title="ROC-кривая")
    axes[1].plot(rec, prec, label=f"PR-AUC = {metrics.average_precision_score(y_test, proba_test):.3f}")
    axes[1].axhline(y_test.mean(), color="k", linestyle="--", linewidth=0.8, label="случайная модель")
    axes[1].scatter(tpr, prec_point, color="red", zorder=3, label=f"порог {threshold:.2f}")
    axes[1].set(xlabel="recall", ylabel="precision", title="PR-кривая")
    axes[0].legend()
    axes[1].legend()
    st.pyplot(fig)

    st.info(
        f"В подборку при этом пороге попадает {pred.sum()} вин из {len(pred)}, "
        f"из них хороших {tp}, а пропущено хороших {fn}.\n\n"
        "**Для сомелье:** если **увеличить** порог, модель берёт только вина, в которых уверена: "
        "подборка становится меньше, но в ней меньше посредственных вин (precision растёт), "
        "зато больше хороших вин пропускается (recall падает). "
        "Если **уменьшить** порог, подборка становится шире и пропускается меньше хороших вин, "
        "но в неё чаще попадают посредственные."
    )

# ---------- 4.3 пакетное предсказание ----------
with tab_batch:
    st.subheader("Прогноз по файлу")
    st.caption(f"Используется порог из сайдбара: {threshold:.2f}")
    uploaded = st.file_uploader("CSV с признаками вин (без target)", type="csv")  # пока файла нет – None
    if uploaded is None:
        st.info(f"Загрузите CSV с колонками: {', '.join(features)}")
    else:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as e:  # st.stop() тут нельзя: не нарисуется следующая вкладка
            st.error(f"Не получилось прочитать CSV: {e}")
            batch = None
        missing = [] if batch is None else [column for column in features if column not in batch.columns]
        if batch is None:
            pass
        elif missing:
            st.error(f"В файле нет колонок: {', '.join(missing)}")
        else:
            try:
                # берём только колонки модели: в CSV могут быть и другие
                proba = model.predict_proba(batch[features])[:, 1]
            except Exception as e:  # например, текст вместо чисел или пропуски
                st.error(f"Модель не смогла посчитать прогноз: {e}")
            else:
                batch = batch.copy()
                batch["good_proba"] = proba
                batch["in_selection"] = (proba >= threshold).astype(int)  # 1 – берём в подборку
                st.metric("Попадут в подборку", int(batch["in_selection"].sum()))
                st.dataframe(batch.sort_values("good_proba", ascending=False), hide_index=True)
                st.download_button(
                    "Скачать прогнозы",
                    data=batch.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

# ---------- 4.4 коэффициенты ----------
with tab_coef:
    st.subheader("Коэффициенты логистической регрессии")
    st.caption("Признаки стандартизованы, поэтому веса можно сравнивать между собой: "
               "вес – изменение логита при росте признака на одно стандартное отклонение.")
    coefs = pd.Series(model[-1].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(coefs.index, coefs.values, color=["tab:red" if v < 0 else "tab:green" for v in coefs.values])
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("коэффициент")
    ax.set_title("Зелёные – «за» хорошее вино, красные – «против»")
    st.pyplot(fig)

    top_pos = coefs[coefs > 0].sort_values(ascending=False).head(3)
    top_neg = coefs[coefs < 0].head(3)
    st.write("Сильнее всего **за** хорошее вино:", ", ".join(f"`{k}` ({v:+.2f})" for k, v in top_pos.items()))
    st.write("Сильнее всего **против**:", ", ".join(f"`{k}` ({v:+.2f})" for k, v in top_neg.items()))
