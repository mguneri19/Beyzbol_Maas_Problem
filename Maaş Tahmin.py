###################################################
# PROJECT: SALARY PREDICTION WITH MACHINE LEARNING
###################################################

# İş Problemi

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
# 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.


# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör



############################################
# Gerekli Kütüphane ve Fonksiyonlar
############################################

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.errors import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#############################################
# 1. Genel Resim
#############################################

df = pd.read_csv("Data/hitters.csv")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include=[np.number]).quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
#############################################

import os

def cat_summary(dataframe, col_name, plot=False, save_dir="cat_plots"):
    value_counts = dataframe[col_name].value_counts()
    ratios = 100 * value_counts / len(dataframe)

    summary_df = pd.DataFrame({col_name: value_counts, "Ratio (%)": ratios.round(2)})
    print(summary_df)
    print("##########################################")

    if plot:
        # Grafik klasörünü oluştur
        os.makedirs(save_dir, exist_ok=True)

        # Grafik oluştur
        plt.figure(figsize=(6, 4))
        sns.countplot(data=dataframe, x=col_name, palette="pastel")
        plt.title(f"Count Plot for {col_name}")
        plt.xlabel(col_name)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Kaydet
        filename = os.path.join(save_dir, f"{col_name}_countplot.png")
        plt.savefig(filename)
        plt.close()

for col in cat_cols:
    cat_summary(df, col, plot=True)

#############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
#############################################

import os

def num_summary(dataframe, numerical_col, plot=False, save_dir="num_plots"):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        # Klasör yoksa oluştur
        os.makedirs(save_dir, exist_ok=True)

        # Grafik oluştur
        plt.figure(figsize=(6, 4))
        dataframe[numerical_col].hist(bins=20, edgecolor="black", color="skyblue")
        plt.xlabel(numerical_col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {numerical_col}")
        plt.tight_layout()

        # Kaydet
        filename = os.path.join(save_dir, f"{numerical_col}_hist.png")
        plt.savefig(filename)
        plt.close()

for col in num_cols:
    num_summary(df, col, plot=True)


#############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
#############################################

def target_summary_with_cat(dataframe, target, categorical_col):

    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)


#############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
#############################################

def high_correlated_cols(dataframe, plot=False, corr_th=0.90, save=False, save_path="corr_heatmap.png"):
    # Sadece sayısal sütunlar
    num_df = dataframe.select_dtypes(include=[np.number])

    # Korelasyon matrisi (NaN'ler pairwise otomatik yok sayılır)
    corr = num_df.corr()

    # Üst üçgen maske ve drop list
    cor_matrix = corr.abs()
    upper_triangle = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle.columns if any(upper_triangle[col] > corr_th)]

    if plot:
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="RdBu", center=0, square=False)
        plt.title("Correlation Heatmap (numeric-only)")
        plt.tight_layout()
        if save:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path)
        plt.show()

    return drop_list

high_correlated_cols(df, plot=True) #['Hits', 'Runs', 'CAtBat', 'CHits', 'CRuns', 'CRBI', 'CWalks']

# Isı haritasını kaydet (ör. plots/corr.png)
to_drop = high_correlated_cols(df, plot=True, corr_th=0.9, save=True, save_path="plots/corr.png") #çıkarılacak sütunlar


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)


#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))


def plot_outlier_boxplots(dataframe, q1=0.10, q3=0.90, save_dir="outlier_plots"):
    outlier_cols = []

    # Klasör oluştur
    os.makedirs(save_dir, exist_ok=True)

    # Outlier kontrolü
    for col in dataframe.select_dtypes(include=["float64", "int64"]).columns:
        low, up = outlier_thresholds(dataframe, col, q1=q1, q3=q3)
        if dataframe[(dataframe[col] < low) | (dataframe[col] > up)].any(axis=None):
            outlier_cols.append(col)

    print(f"Outlier bulunan değişkenler: {outlier_cols}")

    # Boxplot çizimleri
    for col in outlier_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=dataframe[col], color="skyblue")
        plt.title(f"Boxplot - {col}")
        plt.tight_layout()
        filename = os.path.join(save_dir, f"{col}_boxplot.png")
        plt.savefig(filename)  # PNG olarak kaydet
        plt.close()


#Outlier olanları görselleştirelim
plot_outlier_boxplots(df, q1=0.10, q3=0.90) #3 tane çıktı. CHmRun ile CWalks üst limitte aykırılar çok fazla


# Kayıt klasörünü oluştur.
save_dir = "special_outlier_plots"
os.makedirs(save_dir, exist_ok=True)

for col in ["CHmRun", "CWalks"]:
    # Limitleri hesapla
    low, up = outlier_thresholds(df, col, q1=0.01, q3=0.99)
    print(f"{col} -> Lower Limit: {low}, Upper Limit: {up}")

    # Boxplot çiz ve kaydet
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color="skyblue")
    plt.title(f"Boxplot - {col} (q1=0.01, q3=0.99)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{col}_boxplot.png"))
    plt.show()
    plt.close()
#0.90 ile 0.99 arasında hiç fark olmadı.

#alt ve üst limitleri baskılayalım
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


#############################################
# 2. Missing Values (Eksik Değerler)
#############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df) #Sadece Salary-hedef değişkende 59 tane %18'i boş

df.dropna(inplace=True) #boşları siliyoruz


#############################################
# 3. Feature Extraction (Özellik Çıkarımı)
#############################################

new_num_cols=[col for col in num_cols if col!="Salary"]

df[new_num_cols]=df[new_num_cols]+0.0000000001 #log dönüşüm hatasını engellemek için

eps = 1e-10  # Sıfıra bölmeyi önlemek için

# A. Sezon oranları
df["NEW_BA"] = df["Hits"] / (df["AtBat"] + eps)
df["NEW_HR_rate"] = df["HmRun"] / (df["AtBat"] + eps)
df["NEW_RBI_rate"] = df["RBI"] / (df["AtBat"] + eps)
df["NEW_BB_rate"] = df["Walks"] / (df["AtBat"] + df["Walks"] + eps)
df["NEW_RunProd_rate"] = (df["RBI"] + df["Runs"]) / (df["AtBat"] + eps)

# B. Kariyer oranları
df["NEW_C_BA"] = df["CHits"] / (df["CAtBat"] + eps)
df["NEW_C_HR_rate"] = df["CHmRun"] / (df["CAtBat"] + eps)
df["NEW_C_RBI_rate"] = df["CRBI"] / (df["CAtBat"] + eps)
df["NEW_C_BB_rate"] = df["CWalks"] / (df["CAtBat"] + df["CWalks"] + eps)

# C. Form farkları (sezon - kariyer)
df["NEW_BA_diff"] = df["NEW_BA"] - df["NEW_C_BA"]
df["NEW_HR_rate_diff"] = df["NEW_HR_rate"] - df["NEW_C_HR_rate"]
df["NEW_RBI_rate_diff"] = df["NEW_RBI_rate"] - df["NEW_C_RBI_rate"]
df["NEW_BB_rate_diff"] = df["NEW_BB_rate"] - df["NEW_C_BB_rate"]

# D. Kişi-başı / yıl başı üretimler
df["NEW_Hits_per_year"] = df["CHits"] / (df["Years"] + eps)
df["NEW_HR_per_year"] = df["CHmRun"] / (df["Years"] + eps)
df["NEW_RBI_per_year"] = df["CRBI"] / (df["Years"] + eps)
df["NEW_Runs_per_year"] = df["CRuns"] / (df["Years"] + eps)
df["NEW_PA_per_year"] = (df["CAtBat"] + df["CWalks"]) / (df["Years"] + eps)

# E. Güç / temas metrikleri
df["NEW_Power_per_hit"] = df["HmRun"] / (df["Hits"] + eps)
df["NEW_RBI_per_hit"] = df["RBI"] / (df["Hits"] + eps)

# F. Savunma verimi
df["NEW_Fielding_pct"] = (df["PutOuts"] + df["Assists"]) / (df["PutOuts"] + df["Assists"] + df["Errors"] + eps)
df["NEW_PO_per_year"] = df["PutOuts"] / (df["Years"] + eps)
df["NEW_A_per_year"] = df["Assists"] / (df["Years"] + eps)
df["NEW_Err_rate"] = df["Errors"] / (df["PutOuts"] + df["Assists"] + df["Errors"] + eps)

# G. Bu sezon - kariyer/yıl farkları
df["NEW_AtBat_diff"] = df["AtBat"] - (df["CAtBat"] / (df["Years"] + eps))
df["NEW_Hits_diff"] = df["Hits"] - (df["CHits"] / (df["Years"] + eps))
df["NEW_HR_diff"] = df["HmRun"] - (df["CHmRun"] / (df["Years"] + eps))
df["NEW_RBI_diff"] = df["RBI"] - (df["CRBI"] / (df["Years"] + eps))
df["NEW_Runs_diff"] = df["Runs"] - (df["CRuns"] / (df["Years"] + eps))
df["NEW_Walks_diff"] = df["Walks"] - (df["CWalks"] / (df["Years"] + eps))

# H. Etkileşimler
df["NEW_RW_interact"] = df["RBI"] * df["Walks"]
df["NEW_HRW_interact"] = df["HmRun"] * df["Walks"]

df.head()
len(df.columns) #52 değişken

#############################################
# 4. One-Hot Encoding
#############################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = (pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first))
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head() #kategorikler true-false

df = df.astype({col: int for col in df.select_dtypes(include=["bool"]).columns})
df.head() #kategorikleri 0-1 yaptık

#############################################
# 5. Feature Scaling (Özellik Ölçeklendirme)
#############################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 263
Variables: 52
cat_cols: 3
num_cols: 49
cat_but_car: 0
num_but_cat: 3
"""

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()


#############################################
# Base Models
#############################################

y = df["Salary"] #hedef değişken
X = df.drop(["Salary"], axis=1)

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
"""
RMSE: 331.4733 (LR) 
RMSE: 326.1037 (Ridge) 
RMSE: 335.6167 (Lasso) 
RMSE: 315.6034 (ElasticNet) 
RMSE: 311.8995 (KNN) 
RMSE: 328.6695 (CART) 
RMSE: 244.0288 (RF)  -- the second best
RMSE: 444.9893 (SVR) 
RMSE: 234.9128 (GBM)  -- the best one
RMSE: 264.4702 (XGBoost) 
RMSE: 264.8359 (LightGBM) 
RMSE: 248.4288 (CatBoost) -- the third one


"""


################################################


# Skor (neg-MSE) -> RMSE'ye çevireceğiz
scoring = "neg_mean_squared_error"
cv = 5
random_state = 17

# En iyi 3 Modele göre yapalım
gbm = GradientBoostingRegressor(random_state=random_state)
rf  = RandomForestRegressor(random_state=random_state, n_jobs=-1)
cat = CatBoostRegressor(random_state=random_state, verbose=False, loss_function="RMSE")

# Parametre ızgaraları (makul ve hızlı)
gbm_params = {
    "n_estimators": [300, 500, 800, 1000],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [1.0, 0.7],
    "min_samples_leaf": [1, 5, 10],
}

rf_params = {
    "n_estimators": [300, 500, 800],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2", 0.5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

cat_params = {
    "iterations": [500, 1000, 1500],
    "learning_rate": [0.03, 0.1],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [3, 10]
}

searches = [
    ("GBM", gbm, gbm_params),
    ("RF",  rf,  rf_params),
    ("CatBoost", cat, cat_params),
]

best_models = {}
results = []

for name, model, params in searches:
    print(f"\n### {name} GridSearch başlıyor...")
    gs = GridSearchCV(model, params, cv=cv, n_jobs=-1, scoring=scoring, verbose=1)
    gs.fit(X, y)

    best_rmse = np.sqrt(-gs.best_score_)
    print(f"{name} BEST RMSE (CV ort): {best_rmse:.4f}")
    print(f"{name} BEST PARAMS: {gs.best_params_}")

    # İsteğe bağlı: best estimator ile tekrar 10-fold RMSE (daha güvenilir karşılaştırma)
    best_est = gs.best_estimator_
    cv_rmse = np.mean(np.sqrt(-cross_val_score(best_est, X, y, cv=10, scoring=scoring)))
    print(f"{name} RMSE (10-fold yeniden): {cv_rmse:.4f}")

    best_models[name] = best_est
    results.append((name, best_rmse, cv_rmse, gs.best_params_))



# Sonuçları performansa göre sırala ve göster
results_sorted = sorted(results, key=lambda t: t[2])  # 10-fold RMSE'ye göre
print("\n=== En iyi 10-fold RMSE'ye göre sıralı sonuçlar ===")
for name, best_rmse, cv_rmse, params in results_sorted:
    print(f"{name}: BEST_CV_RMSE={best_rmse:.4f} | 10FOLD_RMSE={cv_rmse:.4f} | {params}")


"""
=== En iyi 10-fold RMSE'ye göre sıralı sonuçlar ===

GBM: BEST_CV_RMSE=254.6755 | 10FOLD_RMSE=232.5910 | {'learning_rate': 0.01, 'max_depth': 3, 'min_samples_leaf': 1, 'n_estimators': 800, 'subsample': 0.7}
RF: BEST_CV_RMSE=261.8727 | 10FOLD_RMSE=244.3421 | {'max_depth': None, 'max_features': 0.5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
CatBoost: BEST_CV_RMSE=263.7600 | 10FOLD_RMSE=245.5143 | {'depth': 4, 'iterations': 500, 'l2_leaf_reg': 3, 'learning_rate': 0.03}
"""

#GBM modeliyle ilerliyoruz
best_params = {
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "n_estimators": 800,
    "subsample": 0.7,
    "random_state": 17
}

final_gbm = GradientBoostingRegressor(**best_params)
final_gbm.fit(X, y) #model eğitim

# 10-fold RMSE’yi tekrar görmek istersen:
cv_rmse = np.mean(np.sqrt(-cross_val_score(final_gbm, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"Final GBM 10-fold RMSE: {cv_rmse:.4f}")

#Final GBM 10-fold RMSE: 232.5910

###########################################
# Feature Importances           ###########
###########################################


# Feature importance değerlerini al
importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": final_gbm.feature_importances_
}).sort_values(by="Importance", ascending=False)

# İlk 15 özellik
top_n = 15
top_importances = importances.head(top_n)

# Görselleştir ve kaydet
plt.figure(figsize=(10, 6))
plt.barh(top_importances["Feature"], top_importances["Importance"], color="skyblue")
plt.gca().invert_yaxis()  # En önemliler üstte olsun
plt.title(f"GBM - Top {top_n} Feature Importance", fontsize=14)
plt.xlabel("Importance")
plt.tight_layout()

# Kaydet
save_dir = "feature_importance_plots"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, f"gbm_feature_importance_top{top_n}.png"))
plt.close()

# İlk 15 özelliği yazdır
print(top_importances)
print(f"Görsel kaydedildi: {os.path.join(save_dir, f'gbm_feature_importance_top{top_n}.png')}")

"""
             Feature     Importance
        NEW_RW_interact    0.151
                CRBI       0.145
                CHits       0.141
"""


#######################

#Özellik mühendisliğinde çıkan ilk 3'e göre yeni türetilen featurelar

eps = 1e-10

# NEW_RW_interact türevleri
df["EXTRA_NEW_RW_interact_per_year"] = df["NEW_RW_interact"] / (df["Years"] + eps)
df["EXTRA_NEW_RW_interact_per_PA"]   = df["NEW_RW_interact"] / (df["AtBat"] + df["Walks"] + eps)
df["EXTRA_NEW_RW_over_CRBI"]         = df["NEW_RW_interact"] / (df["CRBI"] + eps)
df["EXTRA_NEW_RW_over_CHits"]        = df["NEW_RW_interact"] / (df["CHits"] + eps)

# CRBI ↔ CHits oranları
df["EXTRA_NEW_CRBI_over_CHits"]      = df["CRBI"] / (df["CHits"] + eps)
df["EXTRA_NEW_CHits_over_CRBI"]      = df["CHits"] / (df["CRBI"] + eps)

# Etkileşimler
df["EXTRA_NEW_RWxCRBI"]              = df["NEW_RW_interact"] * df["CRBI"]
df["EXTRA_NEW_RWxCHits"]             = df["NEW_RW_interact"] * df["CHits"]
df["EXTRA_NEW_CRBIxCHits"]           = df["CRBI"] * df["CHits"]

df.head()

y = df["Salary"] #hedef değişken
X = df.drop(["Salary"], axis=1)


final_gbm = GradientBoostingRegressor(**best_params)
final_gbm.fit(X, y) #model eğitim

# 10-fold RMSE’yi tekrar görmek istersen:
cv_rmse = np.mean(np.sqrt(-cross_val_score(final_gbm, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"Final GBM 10-fold RMSE: {cv_rmse:.4f}")

#Final GBM 10-fold RMSE: 230.0040 -- 232'den 230'a düştü