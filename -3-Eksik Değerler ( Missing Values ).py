# Missing Values (Eksik Değerler (NA))

# Silme
# Deger atama yontemleri (Basit atama yontemleri; ort. atama, medyan atama gibi...)
# Tahmine dayali yontemler (Makine ogrenmesi ya da istatiksel cikarimlar sonucu)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler                 # Standartlastirma, donusturme metodlari

pd.set_option('display.max_columns', None)                                                                 # Butun sutunlari goster.
pd.set_option('display.max_rows', None)                                                                    # Butun satirlari goster.
pd.set_option('display.float_format', lambda x: '%.3f' % x)                                                # Virgulden sonra üc basamak goster.
pd.set_option('display.width', 500)                                                                        # Sutunlara 500 siniri koy.

def grab_col_names(dataframe, cat_th=10, car_th=20): # "cat_th=10, car_th=20" ; Bunlar projeden projeye degisir.
# Bir degisken sayisal olsa dahi 10'dan az sinifa sahip ise bu degisken benim icin kategoriktir; "cat_th=10"
# Yorumu; İlgili degiskendeki essiz deger sayisi, belirledigim threshold (esik) sayisindan kucukse ve degiskenin tipi de object degilse, numerik olarak saklanan kategorik degiskenleri yakala... :)

 # "car_th=20" : kardinal threshold degerim; Kategorik degiskenimin 20'den fazla (essiz) sinifi varsa kategorik gibi gozukup kardinaldir (olculebilirligi yoktur; gozlem sayisi kadar sinifi varsa zaten yorumlayamam.). :)
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal ( Kategorik gorunumlu olup bilgi tasimayan; olcum degeri olmayan) değişkenlerin isimlerini verir.
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
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"] # kategorik degiskenleri sectik
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"] # kategorik olan ama numerik gozuken degiskenleri de sectik

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"] # kategorik gorunen ama kardinal olan degiskenler

    cat_cols = cat_cols + num_but_cat # cat_cols listemizi bastan olusturuyoruz. Numerik gorunumlu kategorikleri de ekledik.
    cat_cols = [col for col in cat_cols if col not in cat_but_car] # cat_cols'un icinde kardinalitesi yuksek olanları da cikardim.
# Sonuc olarak; cat_cols: Kategorik olanlar + Numerik gozukup kategorik olanlar - Kategorik gorunen ama kardinal olanlar


    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] # Tipi object olmayanlar; int veya float olanlar gelecek
    num_cols = [col for col in num_cols if col not in num_but_cat] # Numerik olup kategorik olarak gozukenleri cikardim.
# num_cols = Tipi object olmayanlar(int - float) - Numerik gozuken ama kategorik olan

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car



# Eksik Degerlerin Yakalanmasi;

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()


# Veri setinde eksik gozlem var mi yok mu sorgusu;
df.isnull().values.any()

# Hangi degiskende kac tane eksik deger var sorgusu;
df.isnull()                                                                 # True'lar 1, False'lar 0 doner.
df.isnull().sum()                                                           # Kac tane sorusuna cevap verir.

# Degiskenlerdeki tam deger sayisi;
df.notnull().sum()

# Veri setindeki toplam eksik deger sayisi;
df.isnull().sum().sum()                                                     # Kendisinde en az bir tane eksik hucre olan satir sayisidir.

# En az bir tane eksik degere sahip olan gozlem birimleri
df[df.isnull().any(axis=1)]                                                 # Sutunlara gore eksik deger olan gozlem birimleri (NaN)

# Tam olan gozlem birimleri
df[df.notnull().all(axis=1)]                                                # Sutunlara gore tam olan gozlem degerleri

# Veri setindeki eksik deger olan degiskenleri azalan sekilde sıralamak istersek;
df.isnull().sum().sort_values(ascending=False)


# Bu eksikligin tam veri setine gore oranini gormek istersem;
# "df.isnull().sum()": Veri setindeki eksikligin frekansi,
# "df.shape[0]": Veri setinin toplam gozlem sayisi,
# Yuzdelik olarak gorebilmek icin 100 ile carptik.
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)         # Azalan sekilde eksiklik oranlari gelmis oldu.

# Sadece eksik degere sahip degiskenlerin isimlerini yakalamak istersem (list comphrension ile);
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

# 1) Eksiklik frekansı,
# 2) Eksiligin yuzdelikleri,
# 3) Eksiklige sahip olan degiskenlerin secilmesi,
# Bunlari tek bir fonksiyonda ulasilabilir hale getirmek istersek;

def missing_values_table(dataframe, na_name=False): # Argumanlarimizdan biti dataframe digeri bir ozellik (na_name=False)
# na_name: eksik degerlerin barindigi degiskenin isimlerini ver.
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]                                # df icerisinden eksiklige sahip degiskenler seciliyor.

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)                                          # Eksiklige sahip degiskenler icin eksik deger sayisi toplamini al, sirala.
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)              # Eksik deger orani
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])                              # Degerlerle oranları concat et-birlestir. "keys=['n_miss', 'ratio']": Sutunlari belirledik.
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)                                              # Eksik degere sahip degiskenlerin degerlerini de gormek istedik. Argumandaki ozelligi True yaptik.



# Eksik Deger Problemini Cozme;

missing_values_table(df)                                                    # Eksik degerlerimiz

# Agac modellerinde aykirikliklar ve eksik degerlerin etkisi yoka yakindir. Gozardi etmek mantiklidir.
# İstisna: Regresyon problemi ile ilgileniyorsak, bagimli degisken sayisal bir degiskense ve orada aykirilik olmasi durumunda, sonuca gitme suresi (optimizasyon islemlerinin suresi) uzayabilir.



# Çözüm 1: Hizlica silmek;

df.dropna().shape                                           # Gozlem sayisinin azaldigini gorduk.
                                                            # 891'den 183'e dustu. Cunku; bir satirda en az bir tane bile eksik deger varsa "dropna" onlari sildi.
                                                            # Veri boyuyu >> oldugunda silmek sorun cikarmayabilir.




# Çözüm 2: Basit Atama Yontemleri ile Doldurmak;

df["Age"].mean()                                            # Yas degiskeninin ortalamasi
df["Age"].fillna(df["Age"].mean())                          # Yastaki eksiklikleri ortalamasi ile doldurduk.
df["Age"].isnull().sum()                                    # Yeniden atama yapmadigimiz icin kaydedilmesi, eksik degerler 177 cikti.

# Eksik degerleri ort. ile doldurduktan sonra eksik deger var mi sorgusunu birlikte yapmak istersek;
df["Age"].fillna(df["Age"].mean()).isnull().sum()


# Yastaki eksiklikleri medyani ile doldurmak;
df["Age"].fillna(df["Age"].median()).isnull().sum()

# Yastaki eksiklikleri herhangi bir sabir deger ile doldurmak;
df["Age"].fillna(0).isnull().sum()


# Veri setindeki tum degiskenler icin eksik degerleri doldurma isleminin hizli yapmak icin;

# apply ile yapmak istersek;

# df.apply(lambda x: x.fillna(x.mean()), axis=0)
# apply'in "satirlara gore" ama "sutun bazinda" gitmesini sectik (axis=0)
# Kod hata verir. Cunku; kategorik degiskenler var ve bunlarin ortalamasini alamaz. :)

# !!! Sayisal degiskenleri ortalaması ile doldurmak icin;
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()                                          # Degiskenin tipi object'en farklıysa bu degiskeni ortalamasi ile doldur.
# Tipi object'en farkli degilse old. gibi kalsin.

# İslemin sonucunu dff diye kaydedelim;
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)                                           # Yas degiskenindeki eksikliklerden kurtulmus olduk.

# Eksik degere gore siralarsak;
dff.isnull().sum().sort_values(ascending=False)

# Kategorik degiskenler icin en mantikli doldurma yontemi; modunu (en cok tekrar eden) almaktır.
df["Embarked"].mode()[0]
# mode()[0] : Modun string karsiligina yani values'una erismek icin
df["Embarked"].fillna(df["Embarked"].mode()[0])
# Embarked'daki eksik degerleri 'S' ile doldurmus olduk.

# Embarked icinde artık eksik deger olup olmadigini gormek icin;
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

# Ozel bir ifade ile doldurmak istersek;
df["Embarked"].fillna("missing")

# Kategorik degiskenler icin otomatik doldurma;
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
# Kosul: Tipi object ve eşsiz deger sayisi (len ile tuttuk) <= 10 olanlar.
# Kosul saglaniyorsa eksiklikleri mod ile doldur, saglanmiyorsa old. gibi birak. :)

# Bunu eksik deger var mi sorgusu ile birlikte yapmak istersek;
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# Embarked'da eksik deger kalmadigini gormus olduk.
# Age'de atama yapmadigimiz icin eksik degerleri hala goruyoruz. Cabin icin islem yapmadik.





# Kategorik Değişken Kiriliminda Deger Atama;

# Cinsiyete gore titanic veri setini groupby'a alıp icerisinden yas degiskenini secip, ortalamasini alirsak;
df.groupby("Sex")["Age"].mean()

df["Age"].mean()                                                        # Yasin ortalamasi 29

# Kadinlardaki ve erkeklerdeki eksikliklere farkli ortalama atamasi yapmak (daha dogru) icin;
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))            # Sorgusunu yaptik.
# Cinsiyete gore veri setini groupby'a al, daha sonra yas degiskenini sec, yas degiskeninin ortalamasini ayni groupby kiriliminda ilgili yerlere yerlestir.

# Daha acik yazmak istersek;
df.groupby("Sex")["Age"].mean()["female"]                               # groupby'a gore ortalama degerimiz

df.loc[(df["Age"].isnull()) & (df["Sex"]=="female")]                    # Yas degiskeninde isnull olanlari ve cinsiyetteki kadinlari sec.

# Ek olarak yas degiskenini de secmek istersem;
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"]

# gorupby'a gore ort. isleminin kadinlara gore olan kismina, kadinlarin yas ortalasmini atamak icin;
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.isnull().sum()

# Erkekler icin de ayni islemleri yapmak istersek;
df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()                                                       # Yas degiskenindeki tum eksik degerleri doldurmus olduk.




# Çözüm 3: (Makine Ogrenmesi Yontemi ile) Tahmine Dayalı Atama ile Eksik Degerleri Doldurma;

# Eksiklige sahip degiskeni bagimli degisken, diger degiskenler bagimsiz degiskenler olarak kabul edip bir modelleme islemi gerceklestirecegiz.
# Modelleme islemine gore de eksik degerlere sahip olan noktalari tahmin etmeye calisacagiz.
# 1) Degiskenleri modelin bekledigi standarta gore yazmaliyiz.
# 2) KNN uzaklık tabanli bir algoritma oldugundan dolayi, degiskenleri standartlastirmamliyiz.


df = load()                                                             # Veri setimizi tekrar okuttuk

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]        # Numerik degiskenler icersinden "PassengerId"i cikardik.

# cat_cols icin label encoder ve one hot encoding islemi yapmak (get_dummies) icin;
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
# "drop_first=True": iki sinifa sahip olan kategorik degiskenlerin ilk sinifini atip, ikinci sinifi tutacak.

dff.head()
# "cat_cols + num_cols": iki listeyi topladik. cat_but_car bilgi tasimadigi icin dahil etmedik.
# Tum degiskenleri vermis olsak bile "get_dummies" sadece kategorik degiskenlere donusum uygulamaktadir.
# Yani yukarida yaptigimiz islem; iki ya daha fazla kategorik sinifli degiskenleri numerik sekilde ifade etmis olduk.



# Degiskenlerin standartlastirilmasi
scaler = MinMaxScaler()                                                 # scaler: degerleri 0-1 arasinda donustur bilgisi tasir.

# scaler'i veri setine uyguluyoruz ve islem sonrası formatı dataframe'e ceviriyoruz;
# dataframe'in isimlerini de dff.columns'dan aliyoruz (columns=dff.columns).
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()
# Makine ogrenmesi teknigini kullanmak icin verimiz uygun hale geldi. :)




# KNN'in uygulanması;

from sklearn.impute import KNNImputer                                   # KNNImputer methodunu cagiriyoruz
                                                                        # KNNImputer methodu: Makine ogrenmesi yontemiyle eksik degerleri tahmine dayali doldurma imkani saglar.
imputer = KNNImputer(n_neighbors=5)                                     # Model nesnemi (impuyter) olusturup, komsuluk degerimi 5 yapiyorum.
                                                                        # Komsuluklardaki dolu olan gozlemlerin ortalamasini alır ve eksik degere atar.
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)     # imputer'ı uyguladiktan sonra formatı dataframe'e cevirdik.
dff.head()

# Doldurdugum yerleri gormek istiyorum;
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)  # Geri donusturme islemi yapiyorum
dff.head()

# Kiyaslama yapmak icin atamalari gorup karsilastirma yapmam lazim;

df["age_imputed_knn"] = dff[["Age"]]                                    # dff icerisindeki "age" degiskenini, df icerisine "age_imputed_knn" diye atiyoruz.
                                                                        # Artık ilk dataframe'mde hem "age" hem de "age_imputed_knn" var.

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]                  # ilk degiskenimde "Age" deki eksik degerleri satirlardan sec ve bu iki degiskeni (["Age", "age_imputed_knn"]) getir.
df.loc[df["Age"].isnull()]                                              # Tum degiskenlerdeki atamalari detayli gormek istersem.

# Eksik degerler yerine tahmin edilen degerleri atamis oldum.



# Recap;

df = load()                                         # Veriyi yukledik.
                                                    # missing table
missing_values_table(df)                            # Eksik veriyi basitce raporladik.

# Sayisal degiskenleri medyan ile doldurma; (ortalama aykiri degerlden etkilendigi icin medyani kullanmak daha mantikli :).)
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()

# Kategorik degiskenleri mode ile doldurma;
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

# Kategorik degisken kiriliminda sayisal degiskenleri doldurmak;
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayali Atama ile Doldurma





# Eksik Veride Gelismis Analizler;


# Eksik Veri Yapisinin Incelenmesi;

msno.bar(df)                                        # Ilgili veri setindeki tam sayilari (degerleri) gostermektedir. Yani tam olan gozlem sayilarini vermektedir.
plt.show(block=True)

msno.matrix(df)                                     # Degiskenlerdeki eksikliklerin bir araya cikip cikmama durumlarini incelemek icin gorsel arac
plt.show(block=True)
                                                    # Degiskenlerdeki eksikliklerin ortusme durumlarına bakılır (beyaz alanlar).

msno.heatmap(df)                                    # Eksiklikler uzerine kurulu isi haritasi
plt.show(block=True)
# Eksik degerlerin rassalligi ile ilgili bilgi; degiskenlerdeki eksiklikler acaba birlikte mi olusuyor gibi durumu incelemek istiyorduk
# Eksikliklerin birlikte cikması ve eksikliklerin belirli bir degiskene bagimli olarak cikmasi senaryolari vardir.
# Sag taraftaki palet +1 ile -1 arasinda ; bir korelasyon ifadesidir. 1'lere yakin olmasi kuvvetli iliskiyi ifade eder.
# Pozitif yondeki kuvvetli iliski; degiskenlerdeki eksikliklerin birlikte ortaya ciktigi anlamini tasir. Birinde eksiklik varken digerinde de vardir (yoksa yoktur).
# Negatif yondeki kuvvetli iliski; birisinde varken digerinde yok, birisinde yokken digerinde var.
# Cabin ile age arasindaki eksiklik korelasyon 0,1; anlamsiz bir korelasyon.
# Embarked ile Cabin arasindaki korelasyon -0,1; anlamsiz bir korelasyon.
# Diyoruz ki;
# Bizim veri setimiz icin korelasyonlar anlamli degil (1'e yakin degil) yani; eksiklikler birlikte olusmamistir yorumu yapabiliriz.




# Eksik Degerlerin Bagimli Degisken ile Iliskisinin Incelenmesi;

# Degiskenlerdeki eksikliklerin acaba bagimli degisken tarafinda bir karsiligi var mi?

missing_values_table(df, True)                                                   # Eksik degere sahip olan degiskenleri cektik. True dedigimiz icin indexler de geldi.
na_cols = missing_values_table(df, True)                                         # Eksik degere sahip olan degiskenleri kaydettik.


# Eksik degerlerle bagimli degiskenler arasindaki durumu incelemek icin fonk.;
def missing_vs_target(dataframe, target, na_columns):                            # target: ilgili veri setimizdeki bagimli degiskenimiz.
    temp_df = dataframe.copy()                                                   # Istege bagli orj. dataframe'i bozmamak icin, dataframe'in kopyasi olusturuluyor.

    for col in na_columns: # na_colums'lar icerisinde gez.
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)       # Degiskene '_NA_FLAG' ekle ve eksiklik gordugun yere 1, digerine 0 yaz.

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns     # temp_df'de bir secim yap, tum satirlari al fakat sutunlarda icerisinde _NA_ ifadesi olanlari sec.

    for col in na_flags:                                                        # Yeni olusturdugun degiskenler icerisinde gez.
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
# Bu degiskenleri groupby'a aldiktan sonra hedef degiskenin (survived) ortalamasini al, degiskene gore groupby alip hedef degiskenin count'unu al.

missing_vs_target(df, "Survived", na_cols)                                      # Survived ile eksikliklere sahip degiskenleri karsilastirdik
# TRGET_MEAN: Survived degiskenim acisindan ortalama
# 0: Dolu, 1: Eksik degerlere sahip olan yerler

# Orn.; Eksik degere sahip age degiskeni icin 714 tane dolu deger ve dolu degerler icin survived-hayatta kalma orani: 0,406
# Eksik degere sahip age degiskeni icin 177 tane bos deger ve bos degerler icin survived-hayatta kalma orani ort.: 0,294

# Gemi calisanlarinin cabin bilgisi olmadigindan cabindeki eksik degerlerin dolu degerlere gore hayatta kalma oranindan daha dusuk olmasını is bilgisiyle yorumlayabiliriz.
# cabin'deki eksik degerleri silmek cok dogru olmayacaktir. Cunku hayatta kalma analizinde etkindir.



# Recap;

df = load()
na_cols = missing_values_table(df, True)
# Sayisal degiskenleri direkt "median" ile doldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# Kategorik değişkenleri "mode" ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# Kategorik degisken kiriliminda sayisal degiskenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayali Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)

