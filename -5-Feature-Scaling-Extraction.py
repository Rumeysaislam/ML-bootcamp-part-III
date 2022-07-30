import pandas as pd
from matplotlib import pyplot as plt
from datetime import date
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler # Standartlastirma, donusturme metodlari

pd.set_option('display.max_columns', None)                                                 # Butun sutunlari goster.
pd.set_option('display.max_rows', None)                                                    # Butun satirlari goster.
pd.set_option('display.float_format', lambda x: '%.3f' % x)                                # Virgulden sonra uc basamak goster.
pd.set_option('display.width', 500)                                                        # Sutunlara 500 siniri koy.

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


df = load()



# FEATURE SCALING (Ozellik Olceklendirme)

# Amacimiz; 1) Degiskenler arasindaki olcum farkliligini gidermektir. Yani tum degiskenleri esit sartlar altinda degerlendirebilecek duruma getirmek.
# 2) Train suresini kisaltmak icin
# 3) Uzaklik temelli yontemlerde buyuk degerlere sahip degiskenler dominantlik sergilemekte bu da yanliliga sebep olmaktadir.(Aslinda 1. madde ile denk)

# Agaca dayali yontemlerin bircogu, eksik degerlerden, aykiri degerlerden ve standartlastirmalardan etkilenmez. :)




### StandardScaler: Klasik standartlastirma (Normallestirme). Butun gozlem birimlerinde ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s

df = load()                                                     # Titanic veri setini getirdik.
ss = StandardScaler()                                           # StandardScaler nesnemizi getirdik.
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])       # Nesnemizi yas degiskenine uygulayalim ama uyguladiktan sonra kiyaslama yapabilmek icin df icerisine "Age_standard_scaler" adiyla bunu kaydedelim.
df.head()



### RobustScaler: Medyani cikar iqr'a bol.

# Standart sapma da, ortalama aykiri degerlerden etkilenen metriklerdir.Bunun yerine; Butun gozlem birimlerinden medyani cikarsak daha sonra aykiri degerlerden etkilenen bir degere degikde IQR'a bolsek. Boyleklikle hem merkezi egilimi hem de degisimi goz onunde bulundururuz.
# RobustScaler, StandardScaler'e gore aykiri degerlere dayanikli old. daha tercih edilebilir olabilir. Ama cok yaygin kullanilmamakta...


rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T




### MinMaxScaler: Verilen 2 deger arasında degisken donusumu

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()                                                   # Yas degiskenleri standartlastirilmis formlariyla gelmis oldu.

# Sonuclari karsilastirmak istersek;
age_cols = [col for col in df.columns if "Age" in col]      # Veri setinde icerisinde "age" barindiranlari sectik.


# num_summary: Bir sayisal degiskenin ceyreklik degerleri gostermrk ve hist. grafigini olusturmak icin kullanilir.
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

# Ilk grafikte yas degiskeninin ilk halini, diger grafikte age_standart_scaler geldi.
# Dagilim ayni geldi ama degiskenin degerlerinin (olceklerinin) ifade edilis sekli farklidir (x-ekseni)
# Amacimiza uygun olarak; yapilarini (dagilimlarini) degistirmeden, ifade edilis tarzlarini degistirdik demektir. !!!




## Numeric to Categorical: Sayısal Değişkenleri Kategorik Değişkenlere Çevirme
# Binning

# qcut: Bir degiskenin degerlerini kucukten buyuge siralar ve ceyrek degerlere gore bes parcaya boler.

df["Age_qcut"] = pd.qcut(df['Age'], 5)                          # Yas degiskenini qcut fonskiyonuyla 5 parcaya bol diyoruz.
df.head()                                                       # Yas degiskeni ceyrek degerlere gore bolunmus oldu.

# labellarimiz olsaydi ya da kendi istedigimiz label'lari girmek isteseydik;
# df["Age_qcut"] = pd.qcut(df['Age'], 5, labels=mylabels) yapardik.






# FEATURE EXTRACTION (Ozellik Cikarimi)

# Ozellik Cikarimi; Ham veriden degisken uretmek demektir. Iki kapsamda dusunulebilir;
# 1) Yapisal verilerden degisken uretmek; Elimizde varolan mevcur degiskenler uzerinden degisken uretmek.
# 2) Yapisal olmayan verilerden degisken uretmek; Goruntu, ses, yazi gibi verilerden ozellik uretmek. Ogrenme algoritmalarini kullanabilmek icin. (!)



### Binary Features: Flag, Bool, True-False

# Varolan degiskenin icerisinden yeni degiskenler uretmek ile ilgileniyoruz; degistirmek ile degil.

df = load()
df.head()

# Cabin degiskeninde NaN olan yerlere 1, olmayan yerlere 0 yazmak istiyorum;
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int') # notnull() ile "dolu mu?" sorusunu sordum ve astype('int') ile True-False degerlerine karsilk gelen 0-1 degerlerini int forma cevirdim.
# "NEW_CABIN_BOOL" seklinde yeni bir degisken gelmis oldu. Dolu olanlara 1, bos olanlara 0 yazmis olduk.

# Yeni olusturgumuz degiskene gore groupby yapip, bagimli degiskenimiz; Survived'in ortlamasini alirsak;
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})
# Kabini dolu olanlarin orani (0,667) hayatta kalma orani, bos olanlara gore (0,300) daha yuksek.

# Bu farklilik kayda deger mi, anlamak istiyorum, bunun icin iki grubun oranini kiyaslama testi yaparsam;
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),           # Kabin numarasi olup hayatta kalan kac kisi var?
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],          # Kabin numarasi olmayip hayatta kalan kac kisi var?

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],         # Gozlenme frekanslari; Kabin numarasi olanlar kac kisi?
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])        # Kabin numarasi olmayanlar kac kisi?

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Ho: İkisi arasinda fark yoktur,
# p-value degeri 0,05'den kucuk oldugundan dolayi reddedilir. Yani aralarinda istatistiki olarak anlamli bir fark vardir.
# Cok degiskenli etkiyi bilmedigimden; cabin degiskeni benim icin cok onemlidir genellemesini yapamiyorum.

# 'SibSp' ve 'Parch': Gemideki akrabaliklari ifade eden degiskenler;
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"        # Bunlarin toplami eger 0'dan buyukse yeni bir degisken olustur ve yeni degiskende bir sinif olusturup deger ("NO") ata.
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"      # Toplamlari 0 ise "NEW_IS_ALONE" degiskeninde bir sinif olusturup deger "YES" ata.

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})                    # Survived acisindan degerlendirdik.
# Aralarinda bir fark var gibi gorunuyor.


# Hipotez testi yaparsak; Bu anlamlı bir fark mı?
test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Ho: İkisi arasinda fark yoktur,
# Ho, reddedilir; iki oran arasinda fark vardir.




## Text'ler (Metinler) Üzerinden Özellik Türetmek

df.head()


## Letter Count; Harfleri saydirma

df["NEW_NAME_COUNT"] = df["Name"].str.len() # Name'deki ifadeleri say ve "NEW_NAME_COUNT" degiskenini olustur.


## Word Count; Kelimeleri sayma

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
# ilgili ismi yakaladiginda string'e cevir, split et(bosluklara gore) kac tane kelime varsa bunlari say (boyutuna bak) demis olduk ve yeni degisken olusturduk.


# Ozel Yapilari Yakalamak

# "Dr" ifadesine sahip olanlari yakalamak istersek;
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# x.split(): ilgili satiri split et; Her bir deger liste olarak erisilebilir olacak.
# for x in: Bunlarda gez
# [x... if x.startswith("Dr"): Eger gezdigin ifadelerin basinda "Dr" ifadesi varsa bunu sec(x)


# Dr'a gore groupby alip, survived'in ortalamasina bakarsak;
df.groupby("NEW_NAME_DR").agg({"Survived": "mean"})
# Dr olanlarin hayatta kalma oranlari daha yuksek cikti.

# Kategorik degiskenin frekasnlarini da gormek icin;
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})
df.loc[df["NEW_NAME_DR"]] == 1
# Dr olanlar 10 taneymis.



## Regex (Regular expression) ile Değişken Türetmek
# Text'lerin uzerinde calisiyoruz.

df.head()
# Tittle'lara gore (oncesinde bosluk, sonunda nokta ve kucuk-buyuk harfler iceren) cekme islemi yapmak istiyoruz;
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# df.Name: secme islemi yaptik df["Name]" de olabilirdi. :)
# extract: cekme
# ' ([A-Za-z]+)\.': Onunde bozluk, sonunda nokta olacak, [A-Za-z]: A-Z; buyuk harf ve a-z; kucuk harflerden olusacak


# "NEW_TITLE", "Survived", "Age" degiskenlerini secip, "NEW_TITLE"a gore groupby'a al sonra survived degiskeninin ortalamasini, yas degiskeninin de ortalamasini ve frekansini al;
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

# Bu kategorik degisken kiriliminde ortalamalari doldurmak daha mantikli. :)




## Date Degiskenleri Uretmek

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()              # Veri setinde bir kursa yapilan puanlamalar var.
dff.info()


# Timestamp degiskeninin tipi object. Once onun tipini donusturelim (to_datetime ile);
dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")
# Uzerinden degisken uretecegim timestamp degiskeninin tipi datetime64[ns] formatina donusmus oldu.


# year degiskeni turetmek istersem;
dff['year'] = dff['Timestamp'].dt.year # Yillari cektim


# month degiskeni turetmek istersem;
dff['month'] = dff['Timestamp'].dt.month # Aylari cektim


# year diff; Bugunku yil ile veri setindeki yillarin farkini almak istersek;
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year
# date.today().year: Bugünkü yıl
# dff['Timestamp'].dt.year: Veri setindeki yil


# (!) month diff (iki tarih arasindaki farkin ay cinsinden ifade edilmesi): yil farki + ay farki
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month
# Oncelikle yil farki hesaplanir ve 12 ile carpilip ay cinsinden fark elde edilir. Sonrasinda iki tarih arasindaki ay farki ele alinir.
# date.today().month: Bugünkü ay
# dff['Timestamp'].dt.month: Veri setindeki ay


# day name; Veri setindeki gunlerin hangi gun olduklarina erismek istersem (day_name() metodu kullanarak);
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head() # yil bilgisi, ay, yil farki, ay farki ve gunlerim isim bilgileri geldi.





# FEATURE INTERACTIONS (Özellik Etkileşimleri); Degiskenlerin birbiri ile etkilesime girmesi

# Ozellik Etkilesimleri: Degiskenlerin birbiri ile etkilesime girmesi; degiskenlerin toplanmasi, carpilmasi, karelerinin alinmasi vb.
# Yapilan interactions bir sey ifade ediyor olmali !

df = load()                                             # Titanic veri setini ele aliyoruz.
df.head()

# Yas degiskeni ile Pclass'i carpmak istersek;
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]         # Yasi kucuk ya da buyuk olanlarin yolculuk siniflarina gore refah durumlariyla ilgili durum ortaya cikartmis olabilirz.

# Akrabalik iliskileri toplamı + kisinin kendisi(1) dersek;
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1   # Ailedeki kisi sayisi adinda yeni bir degisken uretmis oluruz.

# Erkek olup yasi 21'e kucuk esit olanlar icin yeni bir degisken ('youngmale') uretirsek;
df.loc[(df['SEX'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

# Erkek olup yasi 21 ile 50 arasinda olanlar icin yeni bir degisken ('maturemale': olgun erkek) uretirsek;
df.loc[(df['SEX'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

# Erkek olup yasi 50'den buyuk esit olanlar icin yeni bir degisken ('seniormale': daha olgun erkek) uretirsek;
df.loc[(df['SEX'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

# Kadin olup yasi 21'e kucuk esit olanlar icin yeni bir degisken ('youngfemale') uretirsek;
df.loc[(df['SEX'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

# Kadin olup yasi 50'den buyuk esit olanlar icin yeni bir degisken ('seniorfemale' uretirsek;
df.loc[(df['SEX'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

# Kadin olup yasi 50'den buyuk esit olanlar icin yeni bir degisken ('seniorfemale' uretirsek;
df.loc[(df['SEX'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()                                               # 'NEW_SEX_CAT' yeni degiskenim geldi.

df.groupby("NEW_SEX_CAT")["Survived"].mean() # Kadin ver ekkeklerin yaslarina gore hayatta kalma oranlarini gormus olduk.

# Etkilesim uzerinden yeni degisken uretmis olduk.