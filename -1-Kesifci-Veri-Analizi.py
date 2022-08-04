# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA) ; Hizli bir sekilde genel fonksiyonlar ile elimize gelen verileri analiz etmek

# 1. Genel Resim
# 2. Kategorik Degisken Analizi (Analysis of Categorical Variables)
# 3. Sayisal Degisken Analizi (Analysis of Numerical Variables)
# 4. Hedef Degisken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


## 1. Genel Resim ; Veri setinin dis ozellikleri ve ic ozelliklerininin genel hatlarini edinmek

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)                  # Butun degiskenleri goster.
pd.set_option('display.width', 500)                         # Degiskenleri alt alta degil yan yana duzgun goster :)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape                        # Satir ve sutun bilgisi
df.info()
df.columns                      # Degisken isimleri
df.index
df.describe().T                 # Sayisal degikenlerin betimsel istatistikleri
df.isnull().values.any()        # Eksik deger var mi?
df.isnull().sum()               # Veri setindeki butun degiskenlerdeki eksik degerlerin sayisini verir.

# Bazi metodlari fonksiyon olarak tanimlayacagiz; (Degisiklikler veri setine nasil yansidi gormek istiyoruz.)
def check_df(dataframe, head=5):    # "head" metodunu burada bicimlendirmek istiyoruz.
    print("##################### Shape #####################")
    print(dataframe.shape)
    check_df()                      # Boyut bilgisi geldi.

    print("##################### Types #####################")
    print(dataframe.dtypes)         # Tip sorgulamasi


    print("##################### Head #####################")
    print(dataframe.head(head))     # Bastan bes veri

    print("##################### Tail #####################")
    print(dataframe.tail(head))     # Sondan bes veri

    print("##################### NA #####################")
    print(dataframe.isnull().sum()) # Veri setindeki eksik degerler ve bunlarin frekanslari

    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)     # Sayisal degiskenlerin dagilim bilgileri
# Ihtiyaclarimiza gore sekillendiriyoruz

check_df(df) # Bu fonksiyonu kullanarak artik veri ile ilgili hizlica bilgi edinebiliriz.

### Başka bir veri seti kullandıgımızda da artık fonksiyonu kolayca kullanabiliyoruz;
df = sns.load_dataset("tips")
check_df(df)


df = sns.load_dataset("flights")
check_df(df)



## 2. Kategorik Değişken Analizi (Analysis of Categorical Variables) ***
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["survived"].value_counts()       # Tek bir degiskeni analiz etmek istedigimizde sinif sayilarina erisebiliriz.
df["sex"].unique()                  # Bir baska degiskenin unique degerleri
df["class"].nunique()               # Degisken icerisinde toplamda kac tane essiz deger var?


# Veri setinde olasi butun kateogrik degiskenleri secmek istiyoruz;
# Tipten hareketle (olasi) kategorik degiskenlerimiz; "category", "object", "bool"
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]


df["sex"].dtypes
str(df["sex"].dteypes)
str(df["sex"].dtypes) in ["object"]

str(df["alone"].dtypes) in ["bool"]



# Tip bilgisi farkli oldugu halde numerik ama kategorik olan degiskenleri (list comprhension ile) yakalamak icin; (Orn. : "survive" kategorik ama 0 ve 1 ile ifade ediliyor)

# int veya float olup essiz sinif sayisi belirli bir degerden kucuk olan degiskenleri yakalayarak yapiyoruz;
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

# Kardinalitesi yuksek (Olculebilirligi acıklanamayacak kadar fazla sinif) degiskenler icin; bu artık kategorik degil :)
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat                                   # cat_cols'ların uzerine butun cat_cotları eklemis olduk. :)

cat_cols = [col for col in cat_cols if col not in cat_but_car]      # cat_but_car olsaydi bu sekilde eleme yapacaktik.

df[cat_cols]                                                        # Belirledigimiz essiz sinif sayisina gore kategorik degisenlerimiz
df[cat_cols].nunique()                                              # Essiz sinif sayilarina baktik.

[col for col in df.columns if col not in cat_cols]                  # Sayisal degiskenleri secmik olduk.



# Sectikten sonra basit bir fonksiyon yaziyoruz;

df["survived"].value_counts()                                       # Hangi siniftan kacar tane var?
100 * df["survived"].value_counts() / len(df)                       # Siniflarin yuzdelik bilgisini yazdirdik ( 100 ile carpıp, degiskenin boyut bilgisine bolersek; ilgili degiskenin siniflarinin oran bilgisine ulasiriz.


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")             # Birden fazla degisken yazdirmak istedigimizde;

cat_summary(df, "sex")                                              # Fonksiyonu cagiriyoruz.


# cat_cols'da gezip icerisindeki tum degiskenlere bu fonksiyonu uygulamak istedigimizde;
for col in cat_cols:
    cat_summary(df, col)                                            # Butun kategorik degiskenler otomatik olarak yazdirildi.




# ileri bir seviyeye tasimak istersek; Grafik eklemek istersek;

# Kategorik degiskenleri sutun grafik ile olusturuyoruz;

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe) # (x sutunu, veri)
        plt.show(block=True)                                        # Birden fazla donguye girdigimizde grafigimizin anlasilir olmasi icin.

cat_summary(df, "sex", plot=True)                                   # Gorseli olusturduk.

# Butun kategorik degiskenlere uygulamak istersek;
for col in cat_cols:
    cat_summary(df, col, plot=True)                                 # adult_male degiskeni bool tipli oldugundan bir yerden sonra hata aldik bunu gorsellestiremedi.

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("sdfsdfsdfsdfsdfsd")
    else:
        cat_summary(df, col, plot=True)
# Tip bilgisi "bool" olanları yazdirdik ama isleme sokmadik, digerlerinin grafigini cizdirmis olduk.


df["adult_male"].astype(int) # ilgili degiskeni secip kullandigimiz fonksiyonumun kabul edecegi bir formata cevirdik.
# True gordugu yerlere:1 , digerlerine : 0 koydu; degiskeni cevirmis olduk.


# Yaptiklarimizi programatik olarak yapmak istersek; Buyuk olcekli islerde mumkun oldugu kadar az ve anlasilir is yapmak önemli !
for col in cat_cols:
    if df[col].dtypes == "bool":                        # Tip "bool" ise sorgulamasini yaptik.
        df[col] = df[col].astype(int)                   # Tipi donusturduk.
        cat_summary(df, col, plot=True)

    else:
        cat_summary(df, col, plot=True)


# Yapiyi karmasiklastirip yapmak istersek;
# if-else sorgusu yapip ve saglanmasi ve saglanmamasi durumuna gore iceride bicimlendirecegiz;
def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df, "adult_male", plot=True)





def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")




## 3. Sayisal Degisken Analizi (Analysis of Numerical Variables)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()


cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]



df[["age","fare"]].describe().T                     # Bunlarin betimsel istatistiklerine erismek istedik.

# Programatik olarak veri setinden sayisal degiskenleri seciyoruz;
num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
# num_cols'da olup cat_cols'da olamayanlar;
num_cols = [col for col in num_cols if col not in cat_cols]

# Bunlar için fonksiyon yazalım; (Olceklebilirlik ve genellendirebilirlik onemli)
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")                              # age degiskeninin ceyreklik degerlerini bicimlendirmis olduk.

# Veri setindeki sayisal tüm degiskelerimiz icin yaparsak;
for col in num_cols:
    num_summary(df, col)

# Grafiklestirmek istersek;
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:                                        # Eger "plot" ozelligi "True" ise;
        dataframe[numerical_col].hist()             # Ilgili bolumu sec; histogramla gorsellestir.
        plt.xlabel(numerical_col)                   # Eksene isim verdik
        plt.title(numerical_col)                    # Title ekledik
        plt.show(block=True)


num_summary(df, "age", plot=True)                   # Gorsel geldi

# Tum degiskenler icin yapmak istersek;
for col in num_cols:
    num_summary(df, col, plot=True)


# Degiskenlerin Yakalanmasi ve Islemlerin Genellestirilmesi;

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()


# docstring

def grab_col_names(dataframe, cat_th=10,  car_th=20): # Bir deg. sayisal olsa dahi eger essiz sinif sayisi < 10 ise kategorik degisken muamelesi yapacagiz. Kategorik deg. eger essiz sinif sayisi 20 den buyukse de kardinal degisken muamelesi yapacagiz.
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal degiskenlerin isimlerini verir.    # ilk fonksiyonun ne gorev yapacagi yazilir.

    Parameters
    ----------
    dataframe: dataframe    # parametre kisminda ilk once tip bilgisi girilir
        degisken isimleri alınmak istenen dataframe'dir. # Ne gorev yapacagi yazilir
    cat_th: int, float    # arguman tiplerinin neler olabilecegi yazilir
        numerik fakat kategorik olan degiskenler için sinif eşik değeri # Gorevi
    car_th: int, float
        kategorik fakat kardinal degiskenler için sinif esik degeri

    Returns # Cikti ne ifade edecek kismi
    -------
    cat_cols: list   # tipi
        Kategorik degisken listesi # gorevi
    num_cols: list
        Numerik degisken listesi
    cat_but_car: list
        Kategorik gorunumlu kardinal degisken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam degisken sayisi
    num_but_cat cat_cols'un icerisinde.

    """
# Boylelikle fonksiyonumuz icin dokumantasyon yazmis olduk.
# Dokumanyasonu gormek icin " help(grab_col_names) " yazip fonksiyonu cagiririz.


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]      # Kategorik degiskenleri olusturdugumuz kesim burada bitti.

# Numarik degiskenleri olusturalım;
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]         # tipi int ve float olanlari sec ama bunlar cat_cols'da olmasin.

# Basit bir raporlama bolumu eklersek;
    print(f"Observations: {dataframe.shape[0]}")                        # Gozlem sayisini shape'in sifirinci indexinden (Shape de iki cıktı var; birincisi: satir sayisi, ikincisi: degisken sayisi)
    print(f"Variables: {dataframe.shape[1]}")                           # ( Observations: Gozlemler, Variables: Degiskenler)
    print(f'cat_cols: {len(cat_cols)}')                                 # Kategorik degiskenlerin boyutu
    print(f'num_cols: {len(num_cols)}')                                 # Numerik degiskenlerin boyutu
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car                              # hesaplanan degerleri tutmak icin "return "yaptik.
#num_but_cols zaten num_cols icerisinde

grab_col_names(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df)                    # Ciktiyi tutmak icin atama yapiyoruz.
# Hem raporu hem degiskenleri almis olduk.

# Ogrendiklerimizi toparlayacak olursak;
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)                                                # cat_cols'larda gezip fonksiyonu uygulamis olduk. Kategorik degiskenler ozetlenmis oldu.


# Numerik degiskenler icin (Num. deg. fonk uyguluyoruz):
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)                                     # Fonksiyonu numarik degiskenler uzerinde gezdirmis olduk.




# BONUS (cat_summary'i plot ozelligi ile bicimlendirilecek sekilde rahatca kullanacagimiz bir yol ele alıyoruz;)

df = sns.load_dataset("titanic")                                       # Veri setini bastan okutuyoruz
df.info()

# bool tipteki degiskenleri donusturuyoruz (int'e ceviriyoruz);
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

# grab fonksiyonumuzu tekrar calistiriyoruz;
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# cat_summary fonk. grafik ozellikli olacak sekilde yeniden tanimliyoruz;
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)                                      # Tum kategorik degiskenler icin grafik


for col in num_cols:
    num_summary(df, col, plot=True)                                      # Tum numerik degiskenler icin grafik




## 4. Hedef Değişken Analizi (Analysis of Target Variable)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

# Survive hedef degiskenini, kategorik ve numerik degiskenler acisinden analiz etmek istiyoruz;

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    ------
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
    num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > car_th and str(df[col].dtypes) in ["category", "object"]]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

# Survive hedef degiskenini, kategorik ve numerik degiskenler acisinden analiz etmek istiyoruz;

df["survived"].value_counts()
cat_summary(df, "survived")



### Hedef Değişkenin Kategorik Degiskenler ile Analizi (Kategorik degiskenler ile caprazlamasi);

# Survive durumunun nasil ortaya ciktigini analiz etmek istiyoruz !
# Bagimli degiskene gore diger degiskenleri goz onunde bulundurarak analizler yapmamiz lazim; degiskenleri caprazlamamiz lazim;

df.groupby("sex")["survived"].mean()        # Cinsiyete gore groupby alip, survive'in ortalamasina baktik.
# Kadinlarin %74 erkeklerin %18 hayatta kaldiği bilgisine eristik. Hedef degiskeni analiz etmis olduk; Kadin olmak hayatta kalma durumunu arttiriyor. :)

# Bu islemi fonksiyon ile tanimlayacak olursak;
def target_summary_with_cat(dataframe, target, categorical_col):    # 3. argumanimiz, kullanacagimiz kategorik degisken.
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

# "dataframe.groupby(categorical_col)[target].mean()" : dataframe'i categorical_col'a gore groupby'a al sonra target'in ortalamasini al.

target_summary_with_cat(df, "survived", "sex")
target_summary_with_cat(df, "survived", "pclass")

# Tek tek yapmak yerine, fonksiyonda butun kategorik degiskenleri hizli hizli gezmek icin;
for col in cat_cols:
    target_summary_with_cat(df, "survived", col)




### Hedef Degiskenin Sayisal Degiskenler ile Analizi;


df.groupby("survived")["age"].mean() # Bagimli degiskenimizi groupby yapip, aggregation bolumune de sayisal degiskenimizi getiririz.
# Hayatta kalip kalmayanlarin yas ortalamasını bulmus olduk.

# yada bu kullanimi tercih edebiliriz;
df.groupby("survived").agg({"age":"mean"})  # Cikti daha duzgun olacaktir.

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df, "survived","age")

# Tum numerik degiskenlere fonksiyonu uygulamak icin;
for col in num_cols:
    target_summary_with_num(df, "survived", col)




## 5. Korelasyon Analizi (Analysis of Correlation)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv") # Meme kanseriyle ilgili csv dosyasi
df.head() # İstemedigimiz degiskenleri gorduk,
df = df.iloc[:, 1:-1] # Problemli degiskenleri disarida birakmak icin bir secim yapildi
df.head()

# Amacimiz elimize bir veri seti geldiginde bunu "isi haritası" araciligiyla korelasyonlarina bakmak.
#  ve daha sonra yuksek korelasyonlu bir degisken setindeki yuksek korelasyonlu degiskenlerden bazilarini disarida birakabilmek.

# Num. degiskenleri sececek basit bir list comph. yapisi olusturuyoruz;
num_cols = [col for col in df.columns if df[col].dtype in [int, float]]     # Veri setindeki sayisal degiskenleri sectik

corr = df[num_cols].corr()

# Korelasyon: Degiskenlerin birbirleriyle iliskisini temsil eden istatistiksel olcumdur. -1 ie +1 arasinda deger alir.
# -1, ve +1 'e yaklastikca iliskinin siddeti kuvvetlenir. İki degisken arasindaki iliski pozitifse; pozitif kroelasyon (bir degiskenin degeri artarsa digerininki de artar),
# değilse; negatif korelasyon (Bir degiskenin degeri artarken digerinin degeri azalir) denir.
# Sifir civarindaki bir korelasyon degeri, korelasyon olmadigi anlamina gelir.
# 0 - 0,50 arasi; dusuk korelasyon.

sns.set(rc={'figure.figsize': (12, 12)})    # seaborn icerisinden grafik boyutu 12*12 olsun diye ayar yapiyoruz.
sns.heatmap(corr, cmap="RdBu")              # seaborn icerisinden heatmap'i cagiriyoruz ve "corr" nesnesini yerlestiriyoruz.
plt.show()
# Maviler: Pozitif korelasyonlar, Kirmizilar: Negatif korelasyonlar



#######################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi

# İhtiyacimiz oldugunda sadece bir analiz araci olarak kullanacagiz; her zaman gerekmez.
# Korelasyon degeri 1 e cok yakin oldugunda neredeyse ayni degisken gibi olur ve bir tanesini silmek isteyebilriz.
#######################



# Korelasyonun pozitif veya negatif olmasiyla ilgilenmiyoruz; hepsini mutlak deger fonksiyonundan "abs()" gecirip pozitif hale getiriyoruz.
# Bir sebebi daha yazacagimiz fonksiyonlarda daha kolay islem yapmak icin.
cor_matrix = df.corr().abs()


#           0         1         2         3
# 0  1.000000  0.117570  0.871754  0.817941
# 1  0.117570  1.000000  0.428440  0.366126
# 2  0.871754  0.428440  1.000000  0.962865
# 3  0.817941  0.366126  0.962865  1.000000

# Bu matriste gereksiz elemanlar var. Gereksiz elemanlari sildigimde elde edecegimiz yapi;

#     0        1         2         3
# 0 NaN  0.11757  0.871754  0.817941
# 1 NaN      NaN  0.428440  0.366126
# 2 NaN      NaN       NaN  0.962865
# 3 NaN      NaN       NaN       NaN

#Kosegen ve altı NaN oldugunda gereksiz elemanlardan kurtulmusuzdur.

# Matriste gordugumuz durumu kendi veri setimizde de gormek istiyoruz;
# np.triu(np.ones(cor_matrix.shape), k=1 : 1'lerden olusan ve olusturdugumuz matrisin boyutunda bir np.array olusturuyoruz.
# (np.ones(cor_matrix.shape), k=1).astype(bool) : Olusturgumuz nparray'i bool'a ceviriyoruz.
# Yukaridaki yapiyi gormek icin, numpy'daki "np.triu" yapisini kullaniyoruz.

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
# Warning'i gormezden gel.
# Kosegen ve altindakilerin NaN oldugunu goruyoruz. Coklama; birbirini tekrar etme durumundan kurtulmus olduk.

# kolonlardan herhangi birisi %90dan buyuk olanları sec; (Matristen eleman seciyoruz yani)
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
drop_list                        # %90 dan buyuk olan korelasyonlar geldi.
cor_matrix[drop_list]            # Yuksek korelasyonlu olanlari sectik.
df.drop(drop_list, axis=1)       # Yuksek korelasyonlu degiskenlerden arindirmis olduk (549 rows * 21 columns)
df.shape                         # Kontrol amacli baktik; 10 tane degisken gitmis.


# Bu islemleri fonksiyonlastirirsak;
def high_correlated_cols(dataframe, plot=False, corr_th=0.90):                                                  # corr_th=0.90 korelasyon degerinin esik degerini belirledik.
# plot=False: ısı haritasını bir opsiyon olarak ekledik ama on tanımlı degerisini false yaptım.
    corr = dataframe.corr()                                                                                     # Korelasyon olusturduk.
    cor_matrix = corr.abs()                                                                                     # Mutlak degerini aldik.
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))              # Kosegen elemanlarina gore bir duzeltme islemi yaptik.
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]     # drop_list ile belirli korelasyon uzerindekileri silmek icin bir liste olusturduk.
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)                                    # df adini fonksiyona girdigimizde drop etmem gereken degiskenlerin listesini bana verecek.
drop_list = high_correlated_cols(df)                        # Kaydettik.
drop_list = high_correlated_cols(df, plot=True)             # Gorsel
df.drop(drop_list, axis=1)                                  # Silme islemi yaptik.
high_correlated_cols(df.drop(drop_list, axis=1), plot=True) # Silinmis halini fonksiyona gonderdik.

# Kosegendeki elemanlar kendileriyle korelasyonları gosterdigi icin yogun ama diger yogunluklar gitti.
