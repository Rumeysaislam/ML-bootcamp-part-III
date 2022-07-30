# FEATURE ENGINEERING & DATA PRE-PROCESSING

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor            # Cok degiskenli aykiri deger yakalamak icin

pd.set_option('display.max_columns', None)                  # Butun sutunlari goster.
pd.set_option('display.max_rows', None)                     # Butun satirlari goster.
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Virgulden sonra üc basamak goster.
pd.set_option('display.width', 500)                         # Sutunlara 500 siniri koy.


############################################################################
# Asagidaki kısım veriyi okuma zamanini olcmek icin yapilmistir: :D :D :D
import time
start_time = time.time()
data = pd.read_csv("datasets/application_train.csv")
print(data)
print("--- %s seconds ---" % (time.time() - start_time))    #; %.2f yaparak virgulden sonraki basamak bilgisini degistirebilirsin.

#############################################################################
# Kucuk olcekli veri setimiz icin fonksiyon tanimlarsak;
def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()
df.head()



# OUTLIERS (Aykiri Degerler)


# Aykiri Degerleri Yakalama

# Grafik Teknigiyle Aykiri Degerler;
###################

sns.boxplot(x=df["Age"])                                    # boxplot: Sayisal degiskenin dagilim bilgilerini verir
plt.show(block=True)

# Sayisal degiskenler icin boxplottan sonra "histogram grafigi" gelir.



# Aykiri Degerler Nasil Yakalanir?

# Ceyrek degerler uzerinden IQR hesabi yapacagiz;
q1 = df["Age"].quantile(0.25)                               # Birinci ceyrek
q3 = df["Age"].quantile(0.75)                               # Ucuncu ceyrek
iqr = q3 - q1                                               # IQR formulu
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr                                        # Alt sinir negatif geldi, yapacagimiz islemlerde bunu gormezden gelecek(Yas negatif olamaz).

# Alt sinirdan kucuk, ust sinirdan buyukleri cagirirsak;
df[(df["Age"] < low) | (df["Age"] > up)]

# Aykiri degerlerin index bilgilerini cagirmak istersek;
df[(df["Age"] < low) | (df["Age"] > up)].index



# Aykiri Deger Var mi Yok mu?

# Hizli bir sekilde aykiri deger olup olmadigini sormak istersek; True, False donecek sekilde
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)     # Sorguyu satir veya sutuna gore degil genel yapmak istedigim icin; "axis=None" yaptik.
df[(df["Age"] < low)].any(axis=None)                        # Negatif oldugu icin False dondu


# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.



# İslemleri daha programatik yapip tum degiskenler icin tek tek ugrasmamak icin islemleri fonksiyonlastiririz.



# Islemleri Fonksiyonlastirmak;

# Kendisine girilen degerlerin, esik degerlerini hesaplayacak fonk. yazacak olursak; (quartile: ceyrek, quantile: dagilim)
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit                              # Sonradan kullanılma ihtimaline karsı return ettik.

outlier_thresholds(df, "Age")                               # q1 ve q3'u on tanimli olarak fonksiyona verdigimiz icin yazmamiza gerek kalmadi.
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")                    # "Fare" icin low, up degerlerini kaydettik.

df[(df["Fare"] < low) | (df["Fare"] > up)].head()           # Ust esik degerinden buyuk degerler geldi.(Alt esik degeri negatif old. icin hesaba katilmadi.)

# indexlere erismek istersek;
df[(df["Fare"] < low) | (df["Fare"] > up)].index



# Aykiri deger var mi, yok mu sorgusu yapan fonksiyon yazarsak;
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)       # Aykiri deger hesaplama fonksiyonunu cagiriyoruz, argumanlarını degistirmek istedigimizde, check_otlier fonksiyonunun argumanlarina o argumanlari eklememiz lazim.
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True                                         # Return'lar boolen tipte.
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")



#  !!! Bir fonk. yazacagiz ki veri setindeki tum sayisal degiskenleri otomatik seciyor olsun;

# grab_col_names;

# Buyuk olcekli veri setimiz icin fonksiyon tanimlarsak;
def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


dff = load_application_train()
dff.head()


def grab_col_names(dataframe, cat_th=10, car_th=20):        # "cat_th=10, car_th=20" ; Bunlar projeden projeye degisir.
# Bir degisken sayisal olsa dahi 10'dan az sinifa sahip ise bu degisken benim icin kategoriktir; "cat_th=10"
# Yorumu; İlgili degiskendeki essiz deger sayisi, belirledigim threshold (esik) sayisindan kucukse ve degiskenin tipi de object degilse, numerik olarak saklanan kategorik degiskenleri yakala... :)

 # "car_th=20" : kardinal threshold degerim; Kategorik degiskenimin 20'den fazla (essiz) sinifi varsa kategorik gibi gozukup kardinaldir (Olculebilirligi yoktur; gozlem sayisi kadar sinifi varsa zaten yorumlayamam.). :)
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
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]                                               # Kategorik degiskenleri sectik.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]      # Kategorik olan ama numerik gozuken degiskenleri de sectik.

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]      # Kategorik gorunen ama kardinal olan degiskenler.

    cat_cols = cat_cols + num_but_cat                                                                                           # cat_cols listemizi bastan olusturuyoruz. Numerik gorunumlu kategorikleri de ekledik.
    cat_cols = [col for col in cat_cols if col not in cat_but_car]                                                              # cat_cols'un icinde kardinalitesi yuksek olanları da cikardim.
# Sonuc olarak; cat_cols: Kategorik olanlar + Numerik gozukup kategorik olanlar - Kategorik gorunen ama kardinal olanlar


    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]                                               # Tipi object olmayanlar; int veya float olanlar gelecek.
    num_cols = [col for col in num_cols if col not in num_but_cat]                                                              # Numerik olup kategorik olarak gozukenleri cikardim.
# num_cols = Tipi object olmayanlar(int - float) - Numerik gozuken ama kategorik olan

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


# Titanic veri seti icin yaparsak;
def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)                                    # Degiskenlerin turleri ile sayilari geldi fakat; num_but_cat zaten cat_cols icerisinde !)
print(cat_cols, num_cols, cat_but_car)

# num_cols'daki "PassengerId" eşsiz deger sayisi numerike uygun oldugu icin geldi fakat rahatsız edici (bu date de olabilirdi). Fonksiyonun okunabilirligi dusmesin diye bunu kaldirmak istiyoruz;
num_cols = [col for col in num_cols if col not in "PassengerId"]                        # num_cols'u yeniden tanimlamis olduk.
print(f"{cat_cols}, \n{num_cols}, \n{cat_but_car}")

# Num_cols'a, check_outlier'i sorarsak;
for col in num_cols:
    print(col, check_outlier(df, col))

# Diger veri setim icin;
def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

dff = load_application_train()

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]                         # Exception (istisna) ifadeyi cikariyoruz "SK_ID_CURR".


# Butun sayisal degiskenlere gidip aykiri deger var mi diye sormak istersek;
for col in num_cols:
    print(col, check_outlier(dff, col))


# (check outlier yaparken threshold degerlerimiz q1, q3 on tanimli degerlerimizdi bunlara dokunamadik, Projeden projeye degistirilebilir.)




# Aykiri Degerlerin Kendilerine Erismek;

# Aykiri degelere erisirken istersem index bilgisini de veren bir fonksiyon "grab_outliers" tanimlarsam;
def grab_outliers(dataframe, col_name, index=False):        # On tanimli olarak simdilik index bilgilerini istemiyoruz (index=False). Ayrica head() argüman olarak verilebilirdi.
    low, up = outlier_thresholds(dataframe, col_name)       # outlier_thresholds'u cagirdik yardim icin :D

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:                             # Aykiri deger sayisi 10'dan buyukse bize head() atsin(ilk bes degeri gostersin).
# shape[0]: gozlem sayisi, shape[1]: degisken sayisi
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]) # Degilse hepsini gostersin.

    if index:                                               # index argumani True ise indexleri return et;
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")                                    # Demek ki 10'dan fazla aykiri deger varmis ki, ilk 5 degeri gorduk.

grab_outliers(df, "Age", True)                              # Index bilgisini de istersek; index argumanini True yapariz.

# İndex bilgilerini daha sonra kullanmak icin saklamak istersem;
age_index = grab_outliers(df, "Age", True)


# Ozetlersek (aykiri degerleri yakalamak icin);
outlier_thresholds(df, "Age")                               # Degisken icin esik deger belirledik.
check_outlier(df, "Age")                                    # Degiskende aykiri deger var mı, yok mu diye sorduk.
grab_outliers(df, "Age", True)                              # Index bilgileri ile aykiri degerlere eristik.




# AYKIRI DEGER PROBLEMINI COZME


### Silme;

# "Fare" degiskenindeki aykiri degerleri silmek istersek;
low, up = outlier_thresholds(df, "Fare")                    # Esik degerlerine ulastik.
df.shape                                                    # Veri setinde kac degisken var ona baktık.

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape         # Alt limitten asagida ya da ust limitten yukarida olanlarin "disindakileri" (yani aykiri olmayanlari) sectik (atama yapmadik).


# Aykiriliklari silmek icin fonksiyon tanimlarsak;
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)                                                       # Esik degerlerini veren fonksiyonu (outlier_thresholds) cagiriyoruz.
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]            # Aykiri olmayanlar icin bir atama yaptik.
    return df_without_outliers                              # Aykirilardan kurtulmus olduk.


df.shape # 891 gozlem oldugunu gorduk.

# (PassengerID num_cols icerisinden daha once silinmmisti.)
for col in num_cols:                                        # Numerik kolonlarda gezdik.
    new_df = remove_outlier(df, col)                        # Kalici degisiklik yapmak istemedigim icin sonucu yeni dataframe olarak atadim.

print(df.shape[0], new_df.shape[0])
df.shape[0] - new_df.shape[0]                               # Gozlem sayilarinda kac tane degisiklik oldugunu gordum; 116 tane gozlem silinmis.



# Silme yontemiyle hucredeki tek bir aykiriliktan dolayi diger tam olan gozlemlerden (tum satirdan) de oluyoruz, bunu engellemek icin silmek degil de baskilama yontemi kullanabiliriz;



### Baskilama Yontemi (re-assignment with thresholds)

# Esik degerlerinin uzerinde kalan degerler, esik degerleri ile degistirilir.

low, up = outlier_thresholds(df, "Fare")                    # "Fare" degiskeni icin limitleri getirdik.

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]        # Aykiri degerleri sectik (iki islem yapmis olduk).

df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]    # Aykiri degerleri "loc" ile secmek istersek...
# loc ile hem satir hem de sutunlardan bir tane secme islemi gerceklestirmis olduk

df.loc[(df["Fare"] > up), "Fare"] = up                      # Ust sinira gore (aykiri degerleri) secme ve atama islemi yaptik.
                                                            # Ust esik degeri ustundekilere, ust esik degerini atayarak baskilamis olduk.

df.loc[(df["Fare"] < low), "Fare"] = low                    # Alt sinira gore secme ve atama islemi yaptik. Negatif aslinda, şu an ihtiyacimiz yok. Baska projede ihtiyac olabilir. :)
                                                            # Alt esik degeri altindakilere, alt esik degerini atayarak baskilamis olduk.


# Baskilama icin fonksiyon yazmak istersek;
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)                           # Limitleri aldik.
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit                  # Alt limitten asagida olan ilgili degiskeni low_limit ile degistirdik.
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit                    # Ust limitten yukarida olan ilgili degiskeni up_limit ile degistirdik.

df = load()                                                 # Veri setini sifirdan okuduk
cat_cols, num_cols, cat_but_car = grab_col_names(df)        # grab_cols_names'i cagirip, degisken secim islemini gerceklestirdik.
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))                      # Outlier var mi diye sorduk.

for col in num_cols:
    replace_with_thresholds(df, col)                        # Esik degerler ile degistirdik.

for col in num_cols:
    print(col, check_outlier(df, col))                      # outlier'i sorguladigimiz "False" aldik. Aykiri degerler baskilanmis oldu.



# (Recap) Ozetlersek;

# Aykiri deger saptama islemleri;
df = load()
outlier_thresholds(df, "Age")                               # !!! Aykiri degeri saptama islemi yaptik,
check_outlier(df, "Age")                                    # Esik degerleri cektik,
grab_outliers(df, "Age", index=True)                        # Esik degerlere gore aykiri deger var mı, yok mu onu sorduk ve aykiri degerleri bize getir dedik,

# Aykiri degerlerden kurtulma yontemleri;
remove_outlier(df, "Age").shape                             # Aykiri degerleri sildik(Yeniden atama yapmadik),
replace_with_thresholds(df, "Age")                          # !!! Esik degerleriyle degistirerek; baskilama yaptik; Kalici degisiklik oldu.
check_outlier(df, "Age")                                    # Tekrar sorguladik ve "False" aldik; Aykiri degerleri baskilamis olduk.




# (LOF) Cok Degiskenli Aykiri Deger Analizi: Local Outlier Factor Yontemi

# LOF: Gozlemleri bulunduklari konumda, yogunluk (komsuluk) tabanli skorlayarak, buna gore aykiri deger olabilecek degerleri tanima imkani verir.
# LOF'tan alinan skor(threshold degeri) 1'e ne kadar yakin olursa(inlier), o kadar iyidir; 1'den uzaklastikca outlier olma ihtimali artar.
# (Threshold degerini orn. 5 olarak belirleseydim; 5'in ustundekiler aykiri deger olacakti.)

# 17 yasinda olmak veya 3 kere evlenmis olmak aykiri degildir, fakat 17 yasindaki birinin, 3 kere evlenmis olmasi aykiri bir durumdur.
# Tek basina aykiri olamayacak bazi degerler, birlikte ele alindiginda aykiri olabilir!


# "Diamonds" veri setini incelersek;

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])         # Sadece numerik degerler
df = df.dropna()                                            # Eksik degeleri drop ettik (kaldirdik).
df.head()
df.shape

# Aykiri deger sorgusu;
for col in df.columns:
    print(col, check_outlier(df, col))


low, up = outlier_thresholds(df, "carat")
df[((df["carat"] < low) | (df["carat"] > up))].shape        # "carat" degiskeninde kac tane outlier (1889 tane) var?


low, up = outlier_thresholds(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape        # "depth" degiskeninde kac tane outlier (2545 tane) var?
# % 25 - % 75 olarak alırsak esik degerlerini ne kadar veri kaybedebilecegimizi, doldurursak da gurultu olusturabilecegimizi anlamis olduk.

clf = LocalOutlierFactor(n_neighbors=20)                    # Komsuluk sayisi 20 olacak sekilde (on tanimli deger) LocalOutlierFactor metodunu getirdik.
clf.fit_predict(df)                                         # Yontemi df'me uyguladim. Local outlier skorlarimi hesapladim.


df_scores = clf.negative_outlier_factor_                    # Takip edilebilirlik acisindan Local outlier skorlarima atama yaptim.
df_scores[0:5]                                              # 5 tanesini gormek istedim.
# df_scores = -df_scores                                    # Bu sekilde eksi degerleri pozitife cevirmis oluruz.
                                                            # Okunabilirlik acisindan negatif biraktik. (Bu durumda -1'e yakin olmasi durumunu degerlendirecegiz.)
np.sort(df_scores)[0:5]                                     # Siralama yaptik; en kotu bes gozlemi gormus olduk.
# (-1'e uzak olanlar kotu sonuc.)


scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')         # xlim=[0, 20] : 20 tane gozlem (x-eksenim)
plt.show(block=True)
                                                            # Her bir nokta (y-eksenindeki) esik deger, bu esik degerlere gore grafik olusturulmus.
                                                            # Esik degerleri incelendiginde en dik egim degisimi bariz olan nokta; en marjinal degisikligin old. noktadir. Bu noktayi esik deger olarak belirleyebilirim.
                                                            # Dik egim noktasindan sonraki degisimlerin kucuk old. bilgisi var. (Dirsek yontemiyle belirlemis olduk :D)

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')         # Daha fazla gozlemde inceleme yaptık.
plt.show(block=True)



th = np.sort(df_scores)[3]                                  # Siralama islemi sonundaki 3. indexi aldim (-4,98).
# Butun gozlem birimleri icin verilen skorlardan esik degeri belirlemis olduk.
df[df_scores < th]                                          # Esik degerden kucuk olanlari yani aykiri olanlari sectik.

df[df_scores < th].shape                                    # 3 tane aykiri deger elde ettik.

### Bunlar acaba NEDEN AYKIRI ?
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
# Cok degiskenli etkiyi goruyoruz; Orn. "depth" icin max degerden daha kucuk bir deger aykiri olmus. Neden; Baska bir degiskene gore olamayacak deger oldugu icin aykiri olmus.

df[df_scores < th].index                                    # Aykirilarin index bilgilerini yakaladik.

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)    # Sildim. Fakat; inplace=True yapmadigim icin kalici olmadi.


# Baskilama ile yapmak gurultuye, ciddi problemlere yol acabilir.
# "Agac yontemleri" ile calisiyorsak bunlara dokunmayacagiz! Illa ki dokunacaksak, cok ucundan kirpiyoruz :D (Orn.: %5 - %95 icin IQR hesaplayip) Gozlem degeri >> aykiri deger ise sil gitsin. :D
