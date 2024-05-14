# 3d_tutorial
３次元再構成のチュートリアル

## 環境構築 
```bash
git clone https://github.com/OwdLabCows/3d_tutorial
cd 3d_tutorial
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## チュートリアル
### カメラキャリブレーション
1. 01_Calibration.ipynb
- チェスボードを用いたカメラキャリブレーションのチュートリアル

### 3次元再構成
1. app.py
- キューブの２次元座標を取得するアプリケーション
2. 02_Non-Linear-Method.ipynb
- SolvePnPを用いて、外部パラメータの算出を行う
- キューブの３次元座標を取得する
- ２次元骨格を３次元に変換する
- ３次元骨格を描画する