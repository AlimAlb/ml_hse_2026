import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def feature_name(column):
    return column.replace('_', ' ')


def show_batch(uploaded, model, features, threshold):
    try:
        batch = pd.read_csv(uploaded)
        missing = [column for column in features if column not in batch.columns]
        if missing:
            st.error('Нет столбцов — ' + ', '.join(missing))
            return

        X_batch = batch[features].astype(float)
        if not X_batch['is_red'].isin([0, 1]).all():
            st.error('В is_red должны быть только 0 и 1.')
            return
        batch['probability'] = model.predict_proba(X_batch)[:, 1]
    except (ValueError, UnicodeDecodeError):
        st.error('Проверьте CSV. Признаки должны быть заполнены числами.')
        return

    batch['recommend'] = (batch['probability'] >= threshold).astype(int)
    st.dataframe(batch, hide_index=True)
    st.download_button(
        'Скачать прогнозы', batch.to_csv(index=False).encode('utf-8'),
        file_name='predictions.csv', mime='text/csv', on_click='ignore',
    )


st.set_page_config(page_title='Подборка вин', layout='wide')
st.title('Подборка вин')
st.caption('Модель оценивает вероятность того, что вино получит 7 баллов или выше.')

model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target']
proba = test_proba(model, X_test)

config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
threshold = st.sidebar.slider(
    'Порог рекомендации', 0.0, 1.0, float(config['threshold']),
    step=0.001, format='%.6f',
)
st.sidebar.caption('Этот порог применяется к одному вину, всему CSV и тестовой выборке.')

single_tab, metrics_tab, batch_tab, coef_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель',
])

with single_tab:
    wine_type = st.radio('Цвет вина', ['Белое', 'Красное'], horizontal=True)
    values = {'is_red': int(wine_type == 'Красное')}
    columns = st.columns(3)

    numeric_features = [column for column in features if column != 'is_red']
    for i, column in enumerate(numeric_features):
        observed = X_test[column]
        lower = float(observed.min())
        upper = float(observed.max())
        values[column] = columns[i % len(columns)].number_input(
            feature_name(column), min_value=lower, max_value=upper,
            value=float(observed.median()), step=max((upper - lower) / 100, 1e-6),
            format='%.6f',
        )

    wine = pd.DataFrame([values], columns=features)
    probability = model.predict_proba(wine)[0, 1]
    st.metric('Вероятность хорошего вина', f'{probability:.2%}')
    if probability >= threshold:
        st.success('Добавить в подборку.')
    else:
        st.info('Пропустить при текущем пороге.')

with metrics_tab:
    predictions = proba >= threshold
    confusion = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)

    columns = st.columns(3)
    columns[0].metric('Precision', f'{precision:.1%}')
    columns[1].metric('Recall', f'{recall:.1%}')
    columns[2].metric('F1', f'{f1:.1%}')

    fig, (ax_cm, ax_roc, ax_pr) = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    metrics.ConfusionMatrixDisplay(confusion, display_labels=['Пропустить', 'Добавить']).plot(
        ax=ax_cm, cmap='Blues', colorbar=False,
    )
    ax_cm.set(title='Матрица ошибок', xlabel='Решение модели', ylabel='Фактический класс')

    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    roc_auc = metrics.roc_auc_score(y_test, proba)
    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax_roc.plot([0, 1], [0, 1], '--', color='gray')
    ax_roc.scatter(fp / (fp + tn), recall, color='tab:red', label='Текущий порог', zorder=3)
    ax_roc.set(title='ROC-кривая', xlabel='FPR', ylabel='Recall')
    ax_roc.legend()

    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, proba)
    pr_auc = metrics.average_precision_score(y_test, proba)
    ax_pr.step(pr_recall, pr_precision, where='post',
               label=f'AP = {pr_auc:.3f}')
    ax_pr.axhline(y_test.mean(), linestyle='--', color='gray')
    ax_pr.scatter(recall, precision, color='tab:red', label='Текущий порог', zorder=3)
    ax_pr.set(title='PR-кривая', xlabel='Recall', ylabel='Precision')
    ax_pr.legend()

    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        'При повышении порога в подборку попадает меньше вин. Среди пропущенных могут '
        'оказаться и хорошие. Если снизить порог, подборка станет больше, но в неё '
        'может попасть больше посредственных вин.'
    )

with batch_tab:
    st.caption('Загрузите CSV с 12 признаками. Решение 1 означает «добавить в подборку».')
    uploaded = st.file_uploader('CSV с признаками вин', type='csv')
    if uploaded is not None:
        show_batch(uploaded, model, features, threshold)

with coef_tab:
    coefficients = pd.Series(
        model.named_steps['logisticregression'].coef_[0], index=features,
    ).sort_values()
    colors = ['tab:blue' if coefficient > 0 else 'tab:orange' for coefficient in coefficients]

    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    ax.barh([feature_name(column) for column in coefficients.index], coefficients, color=colors)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент после стандартизации')
    st.pyplot(fig)
    plt.close(fig)
    positive = feature_name(coefficients.idxmax())
    negative = feature_name(coefficients.idxmin())
    st.write(f'Сильнее всего за рекомендацию — **{positive}**.')
    st.write(f'Сильнее всего против — **{negative}**.')
