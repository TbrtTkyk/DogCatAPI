import base64
import binascii
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, request

import numpy as np
import torch
import torch.nn as nn

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 利用するニューラルネットワークモデル（画像サイズ64×64想定）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 入力のチャネル、出力のチャネル（＝フィルター数）、フィルタのサイズ（3 * 3）
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        # 64 -> 32 -> 16 -> 8 -> 4
        # 128 * 4 * 4 = 2048
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))

        x = self.fc2(x)
        x = self.softmax(x)
        return x


def ready_image(image: Image.Image) -> torch.Tensor:
    """ ImageオブジェクトNNで処理するndarray型に変換する """
    image = image.convert("RGB")
    image = image.resize((64, 64))
    data = np.asarray(image)
    data = np.transpose(data, [2, 0, 1])
    data = data[np.newaxis, :, :, :]
    data = torch.from_numpy(data.astype(np.float32) / 255)
    torch.reshape(data, (1, 3, 64, 64))
    return data

def ready_nn(path: str) -> nn.Module:
    """ ニューラルネットワークモデルを構築する """
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model



# 画像犬猫分析のWebAPI
@app.route('/classify/dogcat/', methods=['POST', 'GET'])
def img_check():
    if request.method == 'POST':
        # POST通信で、"img"に画像データをbase64形式で送信
        dog_per: float = 0.
        cat_per: float = 0.
        result: str = "unknown"
        status: str = "failed" # typing.Literal["success", "failed"]
        try:
            img_str: str = request.json["img"]

            # base64でエンコードされた画像データをImageオブジェクトに直す
            if "base64," in img_str:
                # DARA URI の場合、data:[<mediatype>][;base64], を除く
                img_str = img_str.split(",")[1]
            img_raw = base64.b64decode(img_str)
            img = Image.open(BytesIO(img_raw))

            # NNを通して予測を行う
            data = ready_image(img) # 入力Tensorの用意
            model = ready_nn("model.pth") # NNモデルの用意
            output = model(data) # 出力の計算

            # 結果取得
            dog_per = output[0, 0].item()
            cat_per = output[0, 1].item()
            result = "dog" if dog_per >= cat_per else "cat"
        except binascii.Error as e:
            result = "img string decode error"
            print(e)
        except UnidentifiedImageError as e:
            result = "img open error"
            print(e)
        except KeyError as e:
            result = "request need img element"
            print(e)
        except FileNotFoundError as e:
            result = "file not found"
            print(e)
        except:
            result = "unknown error"
            print("unknown error")
        else:
            status = "success"
            print("success predict")
        finally:
            return jsonify({"dog_per": dog_per, "cat_per": cat_per, "result": result, "status": status})
    else:
        return "test"



#実行して http://localhost:5000/ にアクセス
if __name__ == '__main__':
    app.run(port=5000) # host="0.0.0.0", port=5000
