# DogCatAPI
大学の授業の課題用に作成した犬猫判別APIです。

なお、AIモデルの精度はかなり悪く、WebAPIとして動作させることが主目的です。

## 説明
DogCatClassifyLearning.ipynbがモデルの学習を行うGoogleColab用のファイルです。

img_api.pyを実行すると、WebAPIが起動します。

'localhost:5000/classify/dogcat/' に対して、POST通信で"img"にbase64形式でエンコードした画像データを送ると、それが犬か猫か判別して返します。

## 戻り値のJSON形式
"dog_per"(double)：犬の確率（0～1）

"cat_per"(double)：猫の確率（0～1）

"result"(str)：結果説明の文字列（dog、cat、もしくはエラー理由）

"status"(str)：処理結果（success、failed）
