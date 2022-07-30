# ENCODING (Label Encoding, One-Hot Encoding, Rare Encoding)

# Degiskenlerin temsil sekillerinin degistirilmesidir.
# Orn. bir kategorik degiskenin siniflari label'lardir. Encod etmek: Yeniden kodlamak
# Orn. Ordinal (siniflar arasinda fark olan) degisken; egitim düzeyi seviyeleri icin 0, 1, 2, 3,..., 5 degerlerini atayabiliriz.
# (!) Ordinallik yani siniflar arasi fark olmadiginda label encodingi nominal kategorik degiskene uygulamak sakincali;
# olmadigi halde siniflarin degeri buyuktur ya da kucuktur anlamı verir. One hot encoding yapmak dogru olacaktir.


# LABEL ENCODING & Binary Encoding

# Orn. iki sinifli string ifadeler 0 ve 1 ile temsil ederiz.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno                            # Eksik deger kisminda kullancagimiz kutuphaneyi yukleyip, import edecegiz.
from sklearn.neighbors import LocalOutlierFactor    # Cok degiskenli aykiri deger yakalamak icin
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler # Standartlastirma, donusturme metodlari

pd.set_option('display.max_columns', None)                      # Butun sutunlari goster.
pd.set_option('display.max_rows', None)                         # Butun satirlari goster
pd.set_option('display.float_format', lambda x: '%.3f' % x)     # Virgulden sonra üc basamak goster.
pd.set_option('display.width', 500)                             # Sutunlara 500 siniri koy.

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

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





def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

# Kategorik degiskenimizi label encoding yapmak istiyoruz; 1) Algoritmalarin (Modelleme tekniklerimizin) bizden bekledigi standart formati yakalamak icin.
# One hot encoding yaparken de bazen amacimiz; 2) bir kategorik degiskenin onemli olabilecek bir sinifini degiskenlestirerek ona bir deger atfetmek.

df["Sex"].head() # Cinsiyet degiskenimizi sectik.

le = LabelEncoder()
# fit_transform ile cinsiyet degiskenimize uyguluyoruz.
le.fit_transform(df["Sex"])[0:5]                # Cinsiyet degiskenini "fit" ile fit et sonra degerlerini "transform" ile donustur(son hal). 0'dan 5'e kadar olan degerleri goster.
# Alfabetik siraya gore ilk gordugu degere 0 degerini, ikincisini de 1 degeri verir.

# Hangisinin 0 veya 1 oldugunu ogrenmek icin;
le.inverse_transform([0, 1]) # "female, male"


# Label encoder icin fonk. yazmak istersek;
def label_encoder(dataframe, binary_col):      # binaary_col: iki sinifli bir degisken
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

# Amacimiz olceklenebilir olmak; yani yüzlerce degiskenim old. ne yapacagim;

# Iki sinifli kategorik degiskenleri (essiz sinif sayisi 2 olan int ve float olmayan) sececegim;
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
# len.unique() yapamazdik. Yapsaydik; len 3 cikardi, eksik degerleri de sayacagi icin.
# "nunique": essiz degeri bir sinif olarak gormez.

for col in binary_cols:                        # binary_cols'larda gez
    label_encoder(df, col)                     # label_encoder'i uygula

df.head()                                      # Cinsiyet degiskenini encode etmis olduk. :)


# Daha buyuk veri seti icin denersek;
def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()
df.head()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

df[binary_cols].head()                         # Degiskenleri veri setinden sectim.
# EMERGENCYSTATE_MODE degiskeninde 2'ler var. Eksik degerlere 2 verilmis.
# Yani eksik degerler de doldurulmus. Eksik degerleri boyle doldurmak yaygin olarak tercih edilebiliyor ya da projenin onceki asamalarinda eksik verilerden kurtulmus oluyoruz zaten :)



# Titanic veri setine geri donersek;
def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()

for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()

df = load()

df["Embarked"].value_counts()                   # Siniflari verdi.
df["Embarked"].nunique()                        # Sinif sayisini verdi.

df["Embarked"].unique()                         # Bana essiz degerleri getirdi ama NaN de var.
len(df["Embarked"].unique())                    # NaN'i de sayarak 4 degeri geldi.

                                                # Eksik deger isleme katilmak istendiginde; "len.unique()"
                                                # Eksik deger isleme katilmak istenmediginde "unique" kullanilir.





# ONE - HOT ENCODING

# Olcum problemi yaratmadan; siniflar arasi fark yokken kendim eklememek icin;
# Siniflar degiskenlere donusturulur (sutunlara getirilip) ve degiskenin oldugu satira 1 digerlerine sifir yazilir.
# one-hot encoding' de "drop_first" diyerek ilk sinifi siler ve dummy (kukla) degisken tuzagindan kurtulmus oluruz.
# dummy (kukla) degisken: olusturdugumuz degiskenler
# Birbiri uzerinden olusturulan degiskenler yuksek korelasyona neden olacagindan; ilk sinif drop edilir ve birbiri uzerinden olusma durumu ortadan kaldirilir.

df = load()
df.head()
df["Embarked"].value_counts()                   # Kategorik degiskenin siniflari arasinda fark yok. "nominal"

# pandas'ın get_dummies metodu kullanılır;
# get_dummies: Bana bir df ve icinde donusturmek istedigin sutunlarin adini soyle sadece onlari donustureyim, digerleri old. gibi kalacak.
pd.get_dummies(df, columns=["Embarked"]).head() # embarked'in uc sinifi icin one-hot encoding islemi gerceklesti.

# (!) Degiskenler birbiri uzerinden uretilebilir olmasin diye(drop_first=True);
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()                # Alfabetik siraya gore ilk sinif(c), silindi.

# (!) Eksik degerlerin de bir sinif olarak gelmesini istersek (tercihe bagli);
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()                  #Embarked_nan geldi.

# Butun kategorik degiskenleri ayni anda yapmak istersek;
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()         #drop_first agac modellerde onemli olmasa da dogrusal regrasyonlarda kesin True yap. :)
# Boylelikle iki sinifli kategorik degiskenleri de binary encode etmis oluruz; Cinsiyet erkek mi? (Sex_male) seklinde geldi.

### "get_dummies" metodu ile hem "label encoding" (iki sinifli kategorik degisken icin!) hem de one-hot encoding islemini yapabiliyoruz.


# Fonksiyonlastirmak istersek;
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()


# Butun kategorik degiskenlere (cat_cols) bu donusum uygulanabilir ya da
# Sectiklerimize uygulayabiliriz
# ya da programatik olarak nunique > 3 olanlara uygulayabiliriz

# Sutunlari kendim secip kontorulu saglamak istersem (cok degiskenli veri setlerinde mantikli olan);
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]            # İlgili degiskendeki eşsiz sinif sayisi > 2 ve <= 10 ise sec


one_hot_encoder(df, ohe_cols).head() # ohe_cols'lari one_hot_encoder'dan  gecirdik

df.head()
# Kalici bir degisiklik yok. Kalıcı olmasi icin atama yap.
# df = one_hot_encoder(df, ohe_cols).head()





# RARE (Nadir) ENCODING (Orta-ileri seviye bir islem)


# 1. "Kategorik degiskenlerin" siniflarinin gozlenme sikliginin azlik çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı degisken arasindaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız. Gereksiz; istege bagli, etkisi olmayan degiskenlerden kurtulmak icin.


# Kategorik değişkenlerin azlik çokluk durumunun analiz edilmesi;

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()                                        # Siniflar ve frekanslarini gorduk. academic_degree cok az. Az olanlari bir araya getirme islemi yapmak istiyoruz.

# grab_col_names' i cagirip cat_cols'lara bakarsak;
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Kategorik degiskenin siniflarini ve siniflarin oranlarini getirecek fonk. yazmak istersem;
def cat_summary(dataframe, col_name, plot=False):                              # plot=False: Frekanslari gorsellestirmek istemiyorum.
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),          # Sozluk icerisinde degiskenin ismi ve ilgili degiskenin siniflarinin dagilimi
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))           # ... siniflarinin butun veride gozlenme orani
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# 0,000 ya da 0,001 gibi degerler cop bilgidir, disarida birakilmalidir. Tum dusuk degerdeki siniflari bir araya getirip; rare encoding yapabiliriz.




# Rare kategoriler ile bagimli degisken arasindaki iliskinin analiz edilmesi;

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()                                # Kategorik degiskene gore groupby alip,target'in ortalamasini alalim.
# 0,000 yani 0 'a yakin degerler; kredi odeyebilme durumunu, 1'e yakin olanlar da kredi ödeyememe durumunu ifade etmektedir.


# Yaptigimiz islemleri bir araya getirecek fonk. tanimlarsak;
def rare_analyser(dataframe, target, cat_cols):                                                         # Argumanlarimiz; dataframe, bagimli degisken ve kategorik degiskenler
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))                                             # Ilgili kategorik degiskenin kac sinifi var.
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),                                     # Sinif frekanslari
                            "RATIO": dataframe[col].value_counts() / len(dataframe),                    # Sinif oranlari
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")       # Bagimli degiskene gore groupby islemi

rare_analyser(df, "TARGET", cat_cols)                                                                   # Butun kategorik degiskenler icin "rare" analizimi gerceklestirdim.

# Biz tercihe dayali 0,010 altindaki degerleri bir araya getirecegiz.




# Rare encoder'in yazilmasi (Analiz sonrasi);

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy() # Dataframe'in bir kopyasini aldik.

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]       # Fonksiyona girilen rare oranindan daha dusuk sayida herhangi bir bu kategorik degiskenin sinif orani varsa bunlari rare_columns olarak getir. Yani Rare sinifi olan degiskenler secildi.

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)        # value_counts degeri, toplam gozlem sayisina bolunur. Sinif oranlari bulunmus olunur.
        rare_labels = tmp[tmp < rare_perc].index                # Ust satirda bulunan oranlar calismanin basinda verilen oranlardan daha dusuk orana sahip olan siniflarla veri setini indirge, kalan indexleri tut.
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])                  #temp_df'ler, rare_labels'lerin icindeyse yerine "Rare" yaz, degil ise old. gibi birak.

    return temp_df

new_df = rare_encoder(df, 0.01)                                # Verdigim oranin (0,01) altinda kalan kategorik degisken siniflarini bir araya getirecek.

rare_analyser(new_df, "TARGET", cat_cols)
# Orn.; NAME_TYPE_SUITE icerisinde 2907 tane gozlem birimini bir araya getirmis (Rare). ...

# Ozetle; veri setindeki seyrek sinifli kategorik degiskenleri toplayip, bir araya getirerek, bunlara "Rare" isimlendirmesi yaptik.