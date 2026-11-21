# smabo-sim

smabo-esp32 をそのまま置き換えるシミュレータです。ESP32 と**同じプロトコルで、同じ相手**とやり取りするので、smabo-brain(-ros) と smabo-web は何も変えずに使えます。

```
smabo-web ──REST /config /mode──▶ smabo-sim（ESP32 の代わり）
                                     │ WebSocket（rosbridge JSON）
                                     │  ↑ /esp32/wheel_vel, /esp32/joint_states, /esp32/scan, /esp32/pong
                                     │  ↓ /cmd_vel, /servo/command, /ping
                                     ▼
                              smabo-brain-ros（:9090）または smabo-brain（旧リレー /esp32）
```

Ubuntu・Windows・macOS で動きます（Python 3.10 以上）。ROS は不要です。前面ホルダーのスマホ（smabo-app）の顔・IMU・カメラも再現します（下の「スマホ」）。

## 使い方

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'          # Windows: .venv\Scripts\pip install -e ".[dev]"
.venv/bin/smabo-sim                        # ws://localhost:9090 へ接続（smabo-brain / smabo-brain-ros どちらでも可）
```

brain-ros は、台車と LD06 があるロボット構成で起動し、シミュレータのワールドと同じ地図を渡します。

```bash
cd ../smabo-brain-ros
docker compose run --rm smabo-ros ros2 launch smabo_bringup bringup.launch.py \
  mount:=tt_motor map:=/opt/smabo_ws/src/smabo-brain-ros/smabo_navigation/maps/sim_room.yaml
```

smabo-web では、brain の接続先を `localhost:9090` にします。ESP32 の接続先（REST）は、シミュレータが `/esp32/config_url` で知らせるので自動で `http://localhost:80` になります。走行・サーボのタブも設定タブ（REST）も、brain 経由でシミュレータにつながります（smabo-brain-ros では設定タブはシミュレータに直接つながります）。

ポートは実機と同じ構成です。

| ポート | 中身 | 実機 |
|---|---|---|
| 80（`--http-port`） | REST の設定 API（`GET/POST /config`・`POST /mode`）だけ | ESP32 と同じ |
| 8081（`--phone-port`） | スマホの設定 REST API（`GET/POST /config`） | smabo-app と同じ |
| 8090（`--view-port`） | シミュレータだけの画面：3D 表示 `/view`、スマホの画面とカメラ `/phone` | なし |

80 番を開けないとき（Linux で root でない、ほかのプログラムが使っている）は、自動で 8080 にします。どちらの番号になっても `/esp32/config_url` で知らせるので、smabo-web の設定は要りません。

### MuJoCo（物理演算と 3D ビューアー）

```bash
.venv/bin/pip install -e '.[mujoco,webrtc]'
.venv/bin/mjpython -m smabo_sim --engine mujoco --viewer   # macOS（ビューアーは mjpython が必要）
.venv/bin/smabo-sim --engine mujoco --viewer               # Ubuntu / Windows
```

- ロボットの見た目は smabo-brain-ros の `smabo_description` のメッシュを使います（隣にチェックアウトしてあれば自動で見つけます。別の場所なら `sim.description` か環境変数 `SMABO_DESCRIPTION`）。見つからないときは単純な形で描きます。
- 車輪は DC モーターの特性（止まっているときに最大トルク、全開で最高速度）で回り、床との摩擦で進みます。壁を押すと空転して、エンコーダーが数え続けます。
- キャスターは摩擦のない球、アームは SG90 のトルク上限つきの位置制御サーボです。アームは棚や床に当たると止まります。
- LiDAR は laser の位置から 360 本の光線を障害物に飛ばします。ビューアーでは、当たった点を赤い点で表示します。
- ビューアーを閉じるとシミュレータも終了します。

### ブラウザで見る（ウィンドウなし）

`--viewer` を付けずに MuJoCo で起動し、ブラウザで `http://localhost:8090/view` を開きます（`http://localhost:8090/` でも同じ）。

```bash
.venv/bin/python -m smabo_sim --engine mujoco      # macOS でも mjpython は不要
```

- MuJoCo が画面に出さずに描いた映像を MJPEG で流します。ページを開いている間だけ描画します（複数のブラウザで開いても描画は 1 回分）。
- ドラッグで回転、Shift＋ドラッグか右ドラッグで移動、ホイールでズームです。上段のボタンでロボットのモデルを切り替えられます（下の「ロードマップごとのモデル」）。「ロボットを追う」「真上」「視点リセット」「LiDAR点」のボタンがあります。視点はページを開いている全員で共有です。
- 左上にロボットの実際の位置（シミュレータ内の真値）を出します。
- 別の PC やスマホからは `http://<シミュレータのPCのIP>:8090/view` で見られます。
- 解像度とフレームレートは `view.width` / `view.height`（既定 960×600）、`view.fps`（既定 15）、画質は `view.jpeg_quality`（既定 80）です。
- `kinematic` エンジンでは描くものがないので、ページにその旨を表示します。

### ロードマップごとのモデル

ロボットの構成を、ロードマップの段階に合わせて選べます。起動時は `--model`、起動後は `/view` ページの「モデル」のボタン（または `POST /view/model {"name": "hand"}`）で切り替えます。

```bash
.venv/bin/python -m smabo_sim --engine mujoco --model hand
```

| モデル | ロードマップ | 構成 | ESP32 のモード |
|---|---|---|---|
| `phone` | （ベースの前） | スマホだけ（小さなスタンドに立てる） | なし |
| `base` | brain・web・app・顔・スマホセンサ・音声処理・画像処理・ベースパーツ・esp32 | ベース＋スマホ（床に置く） | なし |
| `hand` | ハンド | ベース＋両手（`left_hand` / `right_hand`）＋ブレッドボード | servos |
| `neck` | 首振り | 首振り台（`head_pan`、MG996R）の上にベース＋ブレッドボード | servos |
| `mobile` | 移動ロボット | TT モーター台車＋ブレッドボード、エンコーダなし | dc_drive |
| `encmobile` | エンコーダ付 | 台車＋エンコーダ | encoder_drive |
| `arm` | ロボットアーム・動作計画 | ベース＋アーム（`arm_joint_1`〜`4`） | servos |
| `nav` | ナビゲーション・brain-ros | 台車＋エンコーダ＋LiDAR | encoder_drive, lidar |
| `full` | （既定） | 台車＋エンコーダ＋LiDAR＋アーム | servos, encoder_drive, lidar |

- 構成は smabo_description の xacro 引数（`arm` / `hand` / `breadboard` / `mount`）と同じ組み合わせで、関節名と軸も同じです。
- モデルを選ぶと、`sim.model` と ESP32 の `modes` をその構成に合わせ、その構成にあるサーボだけを有効にします（実機でその段階に設定する内容と同じ）。brain には新しい config を送り、smabo-web には通知が出ます。
- 台車のないモデルは床（首振りは台）に固定で、走行の指令では動きません。LiDAR のないモデルは `/scan` を出しません。
- 切り替えるとロボットはスタート位置に戻ります。smabo-brain のオドメトリは続きから数えるので、ずれが気になるときは brain を起動し直してください。
- `--viewer` を開いているときは切り替えられません（`--model` で起動し直すか、ブラウザの `/view` を使ってください）。
- 一覧は `GET /view/models` で取れます。モデルの定義は `smabo_sim/models.py` です。

### 画像処理（smabo-web の Vision タブ）を試す

MuJoCo のワールドには、brain の画像処理の 4 つのモード（aruco / qr / face / color）で見つかるターゲットが、スタート位置の 0.9 m 前に並んでいます。スマホの前面カメラの映像は WebRTC で brain に送られるので、実機と同じ手順で Vision タブを確認できます。

| ターゲット | モード | 中身 |
|---|---|---|
| `face` | face | 絵の顔（OpenCV の Haar で検出される） |
| `aruco_0` | aruco | ArUco マーカー（DICT_4X4_50、ID 0） |
| `qr` | qr | 「smabo」と読める QR コード |
| `red_ball` | color（red） | 赤いボール |

1. sim を `--engine mujoco` で起動し、aiortc を入れておきます（`pip install -e '.[webrtc]'`）。brain 側にも aiortc と OpenCV が必要です（smabo-brain の `requirements.txt`）。
2. smabo-web の Vision タブで「Enabled」をオンにし、Mode を選んで「Preview」を押すと、検出結果の枠が映像に重なって表示されます。
3. 追従（視線・首振り・走行）を試すときは、`/view` の「ターゲットを揺らす」のボタンでターゲットを左右に動かします。首振りは `--model neck`、走行は `--model encmobile` などのモデルで試せます。

- ターゲットは障害物ではありません（LiDAR には映らず、ロボットもすり抜けます）。
- 位置や揺れ方は `POST /view/targets/<name> {"x": 1.2, "y": 0, "sway": {"on": true, "amplitude": 0.2, "period": 6}}` で変えられます。一覧は `GET /view/targets` です。
- 別のワールドでは `targets:` に `image`（PNG）か `color`（色の名前か `[r, g, b]`、`shape: sphere | box`）で書きます（`smabo_sim/world.py` の説明を参照）。
- ArUco・QR・顔の画像は `scripts/make_vision_targets.py` で作り直せます（OpenCV が必要。smabo-brain の venv で実行できます）。sim 自体は PNG を読むだけなので OpenCV は不要です。
- ターゲットは MuJoCo だけで表示されます（`kinematic` にはカメラがありません）。

### スマホ（smabo-app の部分）

smabo-sim は ESP32 だけでなく、前面ホルダーに挿したスマホ（smabo-app）も再現します。smabo-app と同じく、brain に別のクライアントとしてつながります。

| smabo-app | smabo-sim |
|---|---|
| `/expression`・`/look_at`・`/speech/say` を受けて顔を描く | 同じトピックを受けて、MuJoCo の世界のスマホ画面に顔を描く。表情は名前で選ぶ（`/expression` は String、大文字小文字は区別しない）。初期はアプリのテンプレート 8 種、視線の追従・まばたき・吹き出しもアプリと同じ計算 |
| `/app/imu/data` を 20 Hz で送る | 同じ形式・同じ相補フィルタで送る。値はスマホの位置にある MuJoCo の加速度・ジャイロセンサー（端末の座標系） |
| カメラ映像を WebRTC で brain に送る | 同じ手順（`/app/webrtc/offer` → `/webrtc/answer`）で、MuJoCo の前面カメラの映像を送る。brain-ros の `webrtc_camera` が `/camera/image_raw` にする。`http://localhost:8090/phone` でも見られる |
| `/app/gps/fix` を位置が変わるたびに送る | 同じ形式（`sensor_msgs/NavSatFix`、frame `gps`）で `phone.gps.rate`（既定 1 Hz）ごとに送る。ワールドの x を東、y を北として、`phone.gps.origin`（既定は東京駅）からの距離を緯度経度に直し、`phone.gps.accuracy`（既定 3 m）ぶんのノイズを加える。`phone.gps.enabled: false` で止まる |
| `/speech/say` を flutter_tts で読み上げる | ホストの音声合成で読み上げる（macOS は `say`、Linux は `espeak-ng`）。言語は `phone.tts.language`（既定 `ja-JP` → Kyoko）、声と速さは `phone.tts.voice`・`phone.tts.rate`。新しい発話は今の発話を止める（アプリと同じ）。`--mute` か `phone.tts.enabled: false` で吹き出しだけになる |
| 設定を REST（`GET/POST /config`、ポート 8081）で公開し、`/app/phone/config_url`（5 秒ごと）と `/app/phone/config`（変更のたび）で知らせる | 同じ API・同じ形（smabo-web の Config タブの Phone 欄で編集）。IMU のオン/オフと周期、カメラ送信のオン/オフ、吹き出し、同じ文の読み上げを無視、読み上げのオン/オフ、表情がその場で効く。ウェイクワード・認識言語・顔の自動非表示は値を持つだけ |
| 表情を REST（`/face/schema`・`/face/expressions`・`/face/preview.png`）で編集できる | 同じ API・同じスキーマ（`smabo_sim/phone/expressions.py`。smabo-app の `test/face_schema.json` と一致することをテストで確認）。大きさ・回転・まぶたの傾き・ハイライトの有無も描く。プレビューは sim の描画で PNG にしたもの。編集した表情は `phone.face.expressions` に入り、`--save` で保存される |
| ウェイクワードのあと発話を録音し、16 kHz の WAV を `/app/speech/audio` で送る（文字起こしは brain） | 同じ形式で送る。発話は 3 通り：`/phone` ページか `POST /phone/speech {"text": "..."}` で入力した文を音声合成したもの（マイク不要）、`{"listen": true}` でホストのマイクから録音したもの（アプリと同じく無音 1.5 秒で終了、`pip install -e '.[voice]'` が必要）、`audio/wav` で送った WAV。ウェイクワードの検出は再現せず、顔をタップしたときと同じ「すぐ録音」になる |

- スマホの位置は CAD に合わせています。本体前面のホルダー（11 mm の溝）に横向きで挿し、画面は前向きです。
- 前面カメラはスマホの上端にあります。どちら側かは `phone.camera_side`（`robot_left`（既定）か `robot_right`）で指定します。IMU の軸もこれで決まります（z は画面の外向き、y はカメラ側）。
- カメラの画角は `phone.camera.hfov_deg`（既定 60°、brain の vision 設定と同じ）、解像度は `phone.camera.width` と `phone.camera.height` で決まります。
- 顔は `kinematic` でも動きます（`/phone/screen.png`）。スマホの画面と前面カメラの表示は MuJoCo だけです。
- 本物の smabo-app も同時につなぐときは `--no-phone` を付けてください。
- カメラ映像の送信には aiortc が必要です（`pip install -e '.[webrtc]'`）。ないときは送信せず、`/phone` ページでの表示だけになります。送るフレームレートは `phone.camera.fps`（既定 10）で、brain 側は vision の `capture_fps` でさらに間引きます。`phone.camera.stream: false` で送信を止められます。
- **接続が切れたとき**は 1 秒後に、brain から answer が 10 秒返らないとき（brain の再起動中など）は 10 秒後に、自動で offer し直します。以前 Docker Desktop で約 30 秒ごとに途切れていたのは、brain-ros 側のイベントループが画像の publish で数秒止まり、ICE の生存確認（consent）に応答できなかったためで、brain-ros 側で修正済みです。docker-compose は Docker Desktop の VM↔Mac 直結ネットワーク（`SMABO_ICE_PREFER=192.168.64.0/24`）を優先する設定にしています。
- smabo-app で編集した表情や画像はスマホの中にしかないので、シミュレータでは初期テンプレートを使います。知らない番号は 1（Normal）になります（アプリと同じ）。

### おもなオプション

| オプション | 内容 |
|---|---|
| `--host` / `--port` | brain の場所（既定 `localhost:9090`） |
| `--config my.json` | ESP32 の `config.json` と同じ形式の設定を重ねる。実機の寸法やサーボ設定をそのまま使える |
| `--save` | REST で変えた設定を `--config` のファイルに書き戻す（ESP32 のフラッシュ保存と同じ） |
| `--world w.yaml` | ワールド（既定は同梱の `smabo_sim/worlds/room.yaml`） |
| `--http-port` | REST API のポート（既定 80、ESP32 と同じ。開けないときは 8080） |
| `--view-port` | シミュレータだけの画面（`/view`・`/phone`）のポート（既定 8090） |
| `--engine` | `kinematic`（既定）または `mujoco` |
| `--model` | ロボットの構成（ロードマップの段階、既定 `full`）。上の「ロードマップごとのモデル」を参照 |
| `--viewer` | MuJoCo の 3D ビューアーを開く（`--engine mujoco` が必要）。ブラウザで見るだけなら不要（`/view`） |
| `--no-phone` | スマホ（顔・IMU・GPS・カメラ・音声）を再現しない |
| `--mute` | `/speech/say` を読み上げない（吹き出しだけ） |
| `--seed` | センサーノイズの乱数シード |
| `--invite` / `--phone-invite` | brain とペアリングする招待コード（ESP32 の部分 / スマホの部分）。smabo-web の「Devices」で出す |
| `--credentials-dir` | ペアリングで受け取った token の保存先（既定 `~/.smabo/sim`） |

### brain とのペアリング

smabo-brain は既定ではペアリングなしで、シミュレータはそのままつながります。`--pairing` で起動した brain は、ペアリングしたクライアントだけを受け付けます。その場合、シミュレータの ESP32 とスマホは、実機と同じく別々にペアリングします。

1. smabo-web の「Devices」で「Add ESP32」「Add phone」を押し、招待コードを出す
2. `--invite ABCD-EFGH --phone-invite WXYZ-2345` を付けて起動する。起動中なら、それぞれの REST に `POST /config {"brain": {"invite": "ABCD-EFGH"}}` を送ってもよい（smabo-web はペアリング前の機器にはこれを直接送る）
3. token は `~/.smabo/sim/esp32.json`・`app.json` に保存され、次からは自動でつながる。smabo-web で外す（または brain を `--reset-pairing` で初期化する）と、次の接続で brain に断られた時点で token を捨て、REST がまた招待コードを受け付ける

ペアリング後の REST（80・8081）は、brain が付ける中継用の鍵（`X-Smabo-Relay-Key`）のない要求を 401 で断り、CORS も返しません。smabo-web の設定画面は brain の `/devices/…` を通ります。シミュレータだけの画面（8090 の `/view`・`/phone`）は今までどおりです。

## 構成

| 層 | ファイル | 役割 |
|---|---|---|
| ファームウェア | `firmware.py`, `robot.py`, `http_api.py`, `link.py`, `pairing.py` | ESP32 の処理を移植したもの。cmd_vel から左右の duty への変換（`dc.min_duty` の床上げ、0.5 秒で止まるタイムアウトを含む）、サーボの軌道追従と速度制限、`servos.behavior: "random"` のランダム動作（`random_motion.py`、`servos.random_groups` のグループごとに実機と同じ計算で動かす）、モードによる有効・無効、送信周期、REST API（ESP32 と同じエンドポイントだけ）、再接続。`/view`・`/phone` は別ポートの `build_sim_app` |
| エンジン | `engines/` | 実際に起きること。車輪の動き、エンコーダ、LiDAR。差し替え可能 |
| モデル | `models.py` | ロードマップの段階ごとの構成（どの部品があるか、ESP32 のモード、有効なサーボ）。MJCF の組み立てとエンジンがこれを見る |
| 表示 | `view.py` | ブラウザ用の 3D 表示（`/view`）。MuJoCo の画面外描画を MJPEG で配信し、視点はドラッグで操作 |
| ワールド | `world.py`, `worlds/*.yaml`, `worlds/targets/*.png` | 壁や障害物の輪郭（LiDAR の光線計算と、Nav2 用地図の書き出しに使う）と、画像処理のターゲット |

エンジンは `kinematic`（運動学、既定）と `mujoco`（物理演算）の 2 つです。`kinematic` は次のとおりです。
- 車輪の速度は一次遅れ（`sim.motor_time_constant`）で目標に近づきます。
- 壁にぶつかると、台車は進まずに車輪だけが回ります。エンコーダは回転を数え続けるので、実機と同じように odom がずれます。
- LD06 は 360 本の光線をワールドに飛ばし、ノイズを加えて返します。反射がない方向は 0.0 にします（LD06 ドライバと同じ）。
- サーボは指令した角度にそのままなります（アームの干渉は扱いません）。

`mujoco` は `engines/mujoco_model.py`（URDF と同じ寸法・関節で MJCF を組み立てる）と `engines/mujoco_engine.py` です。ほかのエンジンも、`engines/base.py` の `Engine` を実装して `engines/__init__.py` に登録すれば追加できます。

### ESP32 との違い（意図的なもの）
- ヘッダーの時刻は、ホストの実時間を使います（ESP32 は起動からのミリ秒）。
- ハードウェア設定（ピン、I2C、Wi-Fi）は保持しますが、効果はありません。変更すると notice を返します。
- 既定の設定は、すぐ使えるように変えてあります（brain は `localhost`、エンコーダ走行と LiDAR を有効）。HTTP ポートは ESP32 と同じ 80 ですが、開けない環境では 8080 にします。
- シミュレータだけの画面（`/view`・`/phone`）は REST とは別のポート（8090）にあります。REST のポートには ESP32 と同じエンドポイントしかありません。
- 80 番の `/config` には ESP32 の設定だけが入ります。シミュレータの設定（`sim.*`・`view.*`）とスマホの設定（`phone.*`）は出さず、POST されても無視して notice を返します。スマホの設定はスマホ側の REST（8081）で変えます。

## ワールドと地図

壁・箱・円柱には `height` も書けます（MuJoCo 用。既定は壁 0.5 m、箱 0.4 m、円柱 0.5 m）。

```yaml
name: room
start: {x: 0.0, y: 0.0, yaw_deg: 0}
shapes:
  - {type: room,     x: 0, y: 0, w: 6, h: 5}            # 外壁（中心と大きさ）
  - {type: box,      x: 2.6, y: 1.2, w: 0.3, h: 0.8}    # 箱（yaw_deg で回転）
  - {type: wall,     x1: -1, y1: 2.5, x2: -1, y2: 0.9}  # 線分
  - {type: cylinder, x: 1.0, y: -1.3, r: 0.12}          # 円柱
```

座標は `map` フレームです。ワールドを変えたら、同じワールドから Nav2 用の地図を書き出し直してください。

```bash
smabo-sim export-map smabo_sim/worlds/room.yaml -o ../smabo-brain-ros/smabo_navigation/maps/sim_room
```

## テスト

```bash
.venv/bin/pytest
```

## バージョンと互換性

ほかの smabo リポジトリとは、同じ minor 番号（`0.2.x` など）どうしで組み合わせます。対応表は
[smabo-brain の README](https://github.com/smabo-smartphone-robot/smabo-brain#バージョンと互換性) にあります。
