import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_iqr_boxplot(
    df, column, target=None, figsize=(10, 5), palette="Set2"
):
    """Универсальная функция для построения Boxplot с автоматическим расчетом IQR и выбросов.

    Parameters:
    df (pd.DataFrame): Исходный датасет
    column (str): Название числовой колонки для анализа
    target (str, optional): Название целевой колонки (например, 'is_rejected') для сравнения групп
    figsize (tuple): Размер графика (ширина, высота)
    palette (str): Цветовая палитра для seaborn
    """
    # 1. Расчёт IQR и границ выбросов для общей выборки
    series = df[column].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outliers_pct = (len(outliers) / len(series)) * 100

    # 2. Настройка стиля и построение графика
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    # Передаём параметры для подсветки выбросов красными точками
    flierprops = dict(
        marker="o",
        markerfacecolor="#e74c3c",
        markersize=5,
        linestyle="none",
        alpha=0.6,
    )

    if target:
        ax = sns.boxplot(
            data=df,
            x=column,
            y=target,
            hue=target,  # Убирает UserWarning в Seaborn 0.13+
            legend=False,
            palette=palette,
            flierprops=flierprops,
            orient="h",
        )
        title_text = f"Boxplot: '{column}' в разрезе '{target}'"
    else:
        ax = sns.boxplot(
            data=df, x=column, color="#3498db", flierprops=flierprops
        )
        title_text = f"Boxplot и IQR-анализ для '{column}'"

    # 3. Добавление инфо-блока со статистикой IQR на график
    stats_text = (
        f"📊 **IQR Статистика ({column}):**\n"
        f"• Q1 (25%): {q1:,.2f}\n"
        f"• Медиана (50%): {series.median():,.2f}\n"
        f"• Q3 (75%): {q3:,.2f}\n"
        f"• IQR: {iqr:,.2f}\n"
        f"• Нижняя граница: {lower_bound:,.2f}\n"
        f"• Верхняя граница: {upper_bound:,.2f}\n"
        f"🚨 Выбросы: {len(outliers):,} шт. ({outliers_pct:.2f}%)"
    )

    # Размещаем плашку со статистикой в верхнем правом углу
    plt.gca().text(
        0.98,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            alpha=0.9,
            edgecolor="#b2bec3",
        ),
    )

    # 4. Оформление заголовков и осей
    plt.title(title_text, fontsize=14, pad=15, fontweight="bold")
    plt.xlabel(column, fontsize=11)
    if target:
        plt.ylabel(target, fontsize=11)

    plt.tight_layout()
    plt.show()







def show_target_by_category(df, cat_col, target_col="Is Rejected", digits=1):
    """Выводит процентное соотношение целевой переменной внутри одной выбранной категории."""
    print(f"🔹 Признак: '{cat_col}' (соотношение '{target_col}' в %):\n")

    ct = (
        pd.crosstab(df[cat_col], df[target_col], normalize="index") * 100
    ).round(digits)

    return ct





# функция для визуализации распределения признака в разрезе целевой
def plot_histogram(
    df, column, hue=None, kde=True, bins=30, multiple="layer", rotation=0
):
    """
    Параметры:
    - df: DataFrame
    - column: название исследуемой колонки
    - hue: целевая переменная для цветового разделения групп
    - kde: отображать ли линию плотности распределения
    - bins: количество интервалов
    - multiple: стиль наложения групп при указании hue
    """
    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x=column,
        hue=hue,
        kde=kde,
        bins=bins,
        multiple=multiple,
        palette="Set1" if hue else None,
    )

    title = f"Распределение признака: {column}"
    if hue:
        title += f" (в разрезе {hue})"

    plt.title(title, fontsize=14, pad=12)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Количество наблюдений", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    if rotation != 0:
        plt.xticks(rotation=rotation, ha="right")

    plt.tight_layout()
    plt.show()





def full_stats(df):
    def calc_column_stats(s):
        # Игнорируем нечисловые колонки
        if not pd.api.types.is_numeric_dtype(s):
            return None
        
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        
        return pd.Series({
            'count': s.count(),
            'unique': s.nunique(),
            'mean': s.mean(),
            'median': s.median(),
            'std': s.std(),
            'min': s.min(),
            'max': s.max(),
            'Q1': q1,
            'Q3': q3,
            'IQR': q3 - q1,
            'P1': s.quantile(0.01),
            'P5': s.quantile(0.05),
            'P95': s.quantile(0.95),
            'P99': s.quantile(0.99),
            'skewness': s.skew(),
            'zero %': (s == 0).mean() * 100
        })

    return df.apply(calc_column_stats).dropna(axis=1, how='all')




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_target_rate_by_bins(
    df, feature, target="Is Rejected", bins=10, qcut=True
):
    """
    Разбивает числовой признак на квантили (bins) и показывает Target Rate (вероятность 1) в каждой группе.
    Отлично подходит для проверки монотонности (Monotonicity).
    """
    temp_df = df[[feature, target]].copy()

    # Разбиваем на корзины
    if qcut:
        # qcut бьет на равные по количеству людей группы (децили)
        temp_df["bin"] = pd.qcut(
            temp_df[feature], q=bins, duplicates="drop"
        ).astype(str)
    else:
        # cut бьет на равные математические отрезки
        temp_df["bin"] = pd.cut(temp_df[feature], bins=bins).astype(str)

    # Считаем Default Rate (Target Rate) и количество людей в корзине
    grouped = (
        temp_df.groupby("bin", observed=True)[target]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Сортируем корзины по возрастанию значений
    grouped["bin_sort"] = grouped["bin"].apply(
        lambda x: float(x.split(",")[0].replace("(", "").replace("[", ""))
    )
    grouped = grouped.sort_values("bin_sort")

    # Превращаем mean в проценты
    grouped["target_rate_pct"] = grouped["mean"] * 100

    # Строим график
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Столбцы - количество наблюдений (Count)
    sns.barplot(
        data=grouped,
        x="bin",
        y="count",
        color="lightgray",
        alpha=0.6,
        ax=ax1,
        label="Количество заявок",
    )
    ax1.set_ylabel("Количество заявок", color="gray", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(False)

    # Линия - Target Rate (Процент отказов)
    ax2 = ax1.twinx()
    sns.lineplot(
        data=grouped,
        x="bin",
        y="target_rate_pct",
        color="#e74c3c",
        marker="o",
        linewidth=3,
        markersize=10,
        ax=ax2,
        label="% Отказов (Target Rate)",
        sort=False,
    )

    # Добавляем подписи процентов на точки
    for i, row in enumerate(grouped.itertuples()):
        ax2.text(
            i,
            row.target_rate_pct + 1,
            f"{row.target_rate_pct:.1f}%",
            color="#c0392b",
            fontweight="bold",
            ha="center",
        )

    ax2.set_ylabel("Доля отказов (%)", color="#c0392b", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.title(
        f"Анализ монотонности риска: {feature} vs {target}",
        fontsize=15,
        pad=15,
        fontweight="bold",
    )

    # Объединяем легенды
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True
    )

    plt.tight_layout()
    plt.show()










import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_target_correlation(df, target_col="Is Rejected", figsize=(10, 6)):
    """
    Строит график корреляций (Pearson vs Spearman) всех числовых признаков с целевой переменной.
    
    Parameters:
    df (pd.DataFrame): Датафрейм
    target_col (str): Название целевой переменной
    figsize (tuple): Размер графика
    """
    # 1. Отбираем только числовые колонки
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    if target_col in num_cols:
        num_cols.remove(target_col)

    # 2. Считаем корреляции только для таргета
    pearson_corr = df[num_cols + [target_col]].corr(method="pearson")[target_col].drop(target_col)
    spearman_corr = df[num_cols + [target_col]].corr(method="spearman")[target_col].drop(target_col)

    # 3. Собираем в один DataFrame
    corr_df = pd.DataFrame({"Pearson": pearson_corr, "Spearman": spearman_corr}).reset_index()
    corr_df = corr_df.rename(columns={"index": "Feature"})

    # 4. Сортируем по силе связи (по модулю Spearman, так как он важнее для деревьев)
    corr_df["abs_spearman"] = corr_df["Spearman"].abs()
    corr_df = corr_df.sort_values(by="abs_spearman", ascending=False)
    sorted_features = corr_df["Feature"].tolist()

    # Готовим данные для seaborn (переводим колонки в строки)
    corr_melted = corr_df.melt(
        id_vars=["Feature"], 
        value_vars=["Pearson", "Spearman"], 
        var_name="Method", 
        value_name="Correlation"
    )

    # 5. Строим график
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=corr_melted, 
        x="Correlation", 
        y="Feature", 
        hue="Method", 
        order=sorted_features, 
        palette=["#3498db", "#e74c3c"]
    )

    # Оформление
    plt.title(f"Корреляция признаков с целевой переменной ({target_col})", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Коэффициент корреляции", fontsize=12)
    plt.ylabel("Признак", fontsize=12)
    plt.axvline(x=0, color='black', linewidth=1) # Линия нуля

    # Добавляем значения прямо на бары
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

    plt.legend(title="Метод", loc="lower right", frameon=True)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


def plot_mutual_information(
    df, target_col="Is Rejected", cat_cols=None, figsize=(10, 6)
):
    """Считает Mutual Information (Взаимную информацию) для всех 13 признаков

    и выводит их в едином рейтинге.
    """
    df_temp = df.copy()

    # 1. Быстро кодируем категориальные признаки в числа для расчёта MI
    if cat_cols is None:
        cat_cols = df_temp.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    for col in cat_cols:
        df_temp[col] = df_temp[col].astype("category").cat.codes

    X = df_temp.drop(columns=[target_col])
    y = df_temp[target_col]

    # Определяем индексы категориальных колонок для корректного расчета
    discrete_features = [X.columns.get_loc(col) for col in cat_cols]

    # 2. Считаем Mutual Information
    mi_scores = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=42
    )
    mi_df = (
        pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})
        .sort_values(by="MI_Score", ascending=False)
        .reset_index(drop=True)
    )

    # 3. Визуализация
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=mi_df, x="MI_Score", y="Feature", palette="viridis"
    )

    plt.title(
        "Единый рейтинг признаков: Mutual Information (Взаимная информация)",
        fontsize=14,
        pad=15,
        fontweight="bold",
    )
    plt.xlabel("Mutual Information Score", fontsize=12)
    plt.ylabel("Признак", fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.show()

    return mi_df





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_feature_correlation_heatmap(
    df, target_col="Is Rejected", figsize=(10, 8)
):
    """Строит треугольную матрицу корреляций между числовыми признаками

    (без целевой переменной) для выявления мультиколлинеарности.
    """
    # 1. Отбираем числовые признаки и исключаем таргет
    num_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    # 2. Считаем матрицу корреляций Пирсона
    corr_matrix = df[num_cols].corr(method="pearson")

    # 3. Маска для скрытия дублирующей верхней половины матрицы
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 4. Построение графика
    plt.figure(figsize=figsize)
    sns.set_style("white")

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title(
        "Матрица мультиколлинеарности числовых признаков (Feature-vs-Feature)",
        fontsize=14,
        pad=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()