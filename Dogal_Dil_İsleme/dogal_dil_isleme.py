import numpy as np
import pandas as pd

yorumlar = pd.read_csv("Restarant_Reviews.csv", sep=';')  # Tab karakteri kullanılıyorsa

# Cümleleri işleme ve uygun hale getirme

import re
import nltk

from nltk.stem.porter import PorterStemmer # stemmer(kelimeleri eklerinden ayırma)
ps = PorterStemmer()

nltk.download("stopwords")
from nltk.corpus import stopwords # Stopwords (that,the gibi etkisiz kelimeleri silme) 

derleme =[]
for i in range(999):
    yorum = re.sub('[^a-zA-Z]', ' ' , yorumlar["Review"][i]) # Yorumlardaki (. ... ! #) gibi karakterleri yok etme
    yorum = yorum.lower() # hepsini küçük harfle yazma
    yorum = yorum.split() # cümle içindeki kelimeleri parçalama ve listeye çevirme.
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words('english'))] # gövdeyi bul ve etkisiz kelimeleri at
    yorum = ' '.join(yorum) # str yap ve her kelime arası boşluk bırak.
    derleme.append(yorum) # uygulanan işlemleri boş derleme listesine al.


# metin belgeleri koleksiyonunu belirteç sayıları matrisine dönüştürme (CountVectorizer)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1000) # farklı yorumlardaki cümlede geçen ortak kelimeler.

X = cv.fit_transform(derleme).toarray() # bağımsız değişkeni matrise dönüştür ve diziye çevir.
y = yorumlar['Liked'].astype(int).values.ravel()
y['Liked'] = y['Review'].str.extract('(\d)') # bağımlı değişken y
# print(df)


# Test-Train ve Classification işlemleri

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.20 , random_state = 0)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


from sklearn.naive_bayes import GaussianNB # modelimiz

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

print(y_pred)