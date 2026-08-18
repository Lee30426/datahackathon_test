import os
#os.environ['PYTHONHASHSEED'] = '0'
#os.environ['TF_DETERMINISTIC_OPS'] = '1'   #tensorflow의 연산을 결정론적으로 수행하도록 설정(1일때)
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1' #tensorflow의 cuDNN 연산을 결정론적으로 수행하도록 설정(1일때)



from function_def import *
from parameter import CFG
import numpy as np, pandas as pd, joblib
import tensorflow as tf


# 예: 이 실행에서만 win_size/epochs 바꾸기 (파생값도 다시 계산하려면 make_config 사용)
# from config import make_config
# CFG = make_config(win_size=30, epochs=50)

# 재현용 seed 고정 (선택)
#np.random.seed(42); tf.random.set_seed(42)
#tf.keras.utils.set_random_seed(42)
#tf.config.experimental.enable_op_determinism()

win_size      = CFG['win_size']
features_dim  = CFG['features_dim']
feat_dim      = CFG['feat_dim']
latent_dim    = CFG['latent_dim']   
batch_size    = CFG['batch_size']
n_critic      = CFG['n_critic']
epochs        = CFG['epochs']
learning_rate = CFG['learning_rate']
k_size        = CFG['k_size']
lstm_units    = CFG['lstm_units']
drop_gen      = CFG['dropout_rate_gen']
crit_filters  = CFG['critic_filters']
crit_drop     = CFG['critic_dropout']
diffs_n, lags_n, smooth_n = CFG['diffs_n'], CFG['lags_n'], CFG['smooth_n']
pca_mode = CFG['pca_mode']

shape                   = CFG['shape']
encoder_input_shape     = CFG['encoder_input_shape']
encoder_reshape_shape   = CFG['encoder_reshape_shape']
generator_input_shape   = CFG['generator_input_shape']
generator_reshape_shape = CFG['generator_reshape_shape']
critic_x_input_shape    = CFG['critic_x_input_shape']
critic_z_input_shape    = CFG['critic_z_input_shape']

ckpt_dir = CFG['ckpt_dir']; os.makedirs(ckpt_dir, exist_ok=True)
print("win_size=%d features_dim=%d k_size=%d epochs=%d pca_mode=%s"
      % (win_size, features_dim, k_size, epochs, pca_mode))
DATA_DIR = './data/preprocessed/train'
file_ids = range(1000, 1051)
files = [os.path.join(DATA_DIR, f'{fid}_chg.csv') for fid in file_ids]

# 0-1. 존재 확인: 없는 파일은 걸러내고 경고 (파일 하나가 조용히 빠지는 것 방지)
missing = [f for f in files if not os.path.exists(f)]
if missing:
    print(f"[경고] 없는 파일 {len(missing)}개:", [os.path.basename(m) for m in missing])
files = [f for f in files if os.path.exists(f)]
print(f"불러올 파일 수: {len(files)}개")
assert len(files) > 0, "불러올 파일이 없습니다. 경로를 확인하세요."

# ---------------------------------------------------------------
# 1. 각 파일 읽기 + 컬럼 정합성 검사 (concat 전에 반드시)
#    - 51개 중 하나라도 컬럼 구성/순서가 다르면 pd.concat이 에러 대신
#      NaN을 채우거나 열을 어긋나게 정렬 → StandardScaler/PCA fit이 통째로 오염됨
#    - 에러가 안 나서 학습이 다 끝난 뒤에야 이상 증상으로 드러나는 최악의 버그
# ---------------------------------------------------------------
raw_dfs = [pd.read_csv(f) for f in files]

ref_cols = list(raw_dfs[0].columns)          # 첫 파일을 기준 컬럼으로
for f, df in zip(files, raw_dfs):
    if list(df.columns) != ref_cols:         # 이름뿐 아니라 순서까지 동일해야 PCA 축이 안 섞임
        raise ValueError(
            f"컬럼 불일치: {os.path.basename(f)}\n"
            f"  기준: {ref_cols}\n  실제: {list(df.columns)}"
        )

# 검사 통과분에만 diff_smooth 적용 (파일별로 따로)
dfs = [diff_smooth_df(df, lags_n, diffs_n, smooth_n) for df in raw_dfs]
print("각 파일 shape:", [d.shape for d in dfs])

df_concat = pd.concat(dfs, axis=0, ignore_index=True)   # 세로로 합치기

scaler_std = StandardScaler()
scaled_concat = scaler_std.fit_transform(df_concat)     # 표준화 (스케일 차이 제거)

features_dim = 3                                         # PCA 축 개수
pca = PCA(n_components=features_dim)
pca.fit(scaled_concat)                                   # 공통 축 학습 (fit만)

# 설명력 확인 (며칠 전 강조한 부분)
print("explained_variance_ratio:", pca.explained_variance_ratio_)
print("누적 설명력:", np.cumsum(pca.explained_variance_ratio_))
# 3개 축이 전체 분산의 몇 %를 설명하는지 반드시 확인

# ---------------------------------------------------------------
# 3. 각 파일을 "따로" 변환 + 윈도우 (파일 경계 오염 방지)
# ---------------------------------------------------------------
X_list = []
for df in dfs:
    # 3-1. 같은 표준화 + 같은 PCA로 변환 (transform만, fit 안 함)
    scaled = scaler_std.transform(df)
    data = pca.transform(scaled)

    # 3-2. date 컬럼 붙이기 (time_segments_aggregate가 요구)
    tmp = pd.DataFrame(data, columns=[f'pca_{i}' for i in range(1, features_dim+1)])
    tmp.insert(0, 'date', range(1, len(tmp)+1))

    # 3-3. aggregate → 결측 처리
    Xf, idxf = time_segments_aggregate(tmp, interval=1, time_column='date')
    Xf = SimpleImputer().fit_transform(Xf)

    # 3-4. MinMax 스케일 (이건 나중에 합쳐서 fit하는 게 이상적 — 아래 주석 참고)
    #      여기선 일단 각자 두고, 합친 뒤 다시 스케일하는 방식으로 감
    X_list.append((Xf, idxf))

# ---------------------------------------------------------------
# 4. MinMax 스케일은 "합친 데이터" 기준으로 fit
#    (모델 입력 범위 -1~1을 51개 파일 전체 기준으로 통일)
# ---------------------------------------------------------------
X_all_raw = np.concatenate([x for x, _ in X_list], axis=0)
scaler_mm = MinMaxScaler(feature_range=(-1, 1))
scaler_mm.fit(X_all_raw)                                 # 합친 기준으로 fit

# ---------------------------------------------------------------
# 5. 각 파일을 스케일 + 윈도우 → 윈도우 레벨에서 합치기
# ---------------------------------------------------------------
X_windows = []
for (Xf, idxf) in X_list:
    Xf_scaled = scaler_mm.transform(Xf)                 # 공통 스케일 적용
    Xw, y, Xw_idx, y_idx = rolling_window_sequences(
        Xf_scaled, idxf, window_size=win_size,
        target_size=1, step_size=1, target_column=0)
    X_windows.append(Xw)

X_all = np.concatenate(X_windows, axis=0)               # 윈도우 합치기
print("최종 학습 윈도우:", X_all.shape)

# TRAIN_CSV = './data/preprocessed/train/1000_chg.csv'
# print("TRAIN_CSV:", TRAIN_CSV)

# df_train_0 = pd.read_csv(TRAIN_CSV)
# data_1 = diff_smooth_df(df_train_0, lags_n, diffs_n, smooth_n)

# from sklearn.preprocessing import StandardScaler
# std_scaler = StandardScaler()
# data_1 = std_scaler.fit_transform(data_1)

# pca = PCA(n_components=features_dim)
# data = pca.fit_transform(data_1)          # 학습 데이터로 축 확정(fit) + 변환

# df_1 = []
# for i in range(len(data)):
#     row = [i + 1] + [data[i][jj] for jj in range(features_dim)]
#     df_1.append(row)
# df = pd.DataFrame(df_1)
# df.columns = ['date'] + ['pca_%s' % str(i) for i in range(1, features_dim + 1)]
# print("After PCA:", df.shape)

# X, index = time_segments_aggregate(df, interval=1, time_column='date')
# X = SimpleImputer().fit_transform(X)

# scaler = MinMaxScaler(feature_range=(-1, 1))
# X = scaler.fit_transform(X)               # 학습 데이터로 스케일 범위 확정(fit)

# # 정합 모드: 학습에서 fit한 pca/scaler 저장 -> 테스트가 transform 으로 재사용
# if pca_mode == 'consistent':
#     joblib.dump(pca, CFG['pca_path'])
#     joblib.dump(scaler, CFG['scaler_path'])
#     joblib.dump(std_scaler, CFG['std_scaler_path'])
#     print("saved pca/scaler ->", CFG['pca_path'], CFG['scaler_path'])
# else:
#     print("original 모드: pca/scaler 저장 안 함(테스트가 각자 fit_transform)")

X_all, y, X_index, y_index = rolling_window_sequences(
    X_all, X_index, window_size=win_size, target_size=1, step_size=1, target_column=0)
print("after window:", X_all.shape)

encoder   = build_encoder_layer(encoder_input_shape, encoder_reshape_shape,
                                win_size=win_size, latent_dim=latent_dim)
generator = build_generator_layer(generator_input_shape, generator_reshape_shape,
                                  win_size=win_size, features_dim=features_dim,
                                  lstm_units=lstm_units, dropout_rate=drop_gen)
critic_x  = build_critic_x_layer(critic_x_input_shape, k_size=k_size,
                                 filters=crit_filters, dropout_rate=crit_drop)
critic_z  = build_critic_z_layer(critic_z_input_shape)
optimizer = tf.keras.optimizers.Adam(learning_rate)
print("networks & optimizer ready")

z = Input(shape=(latent_dim, 1)); x = Input(shape=shape)
x_ = generator(z); z_ = encoder(x)
critic_x_model = Model([x, z],
    [critic_x(x), critic_x(x_), RandomWeightedAverage(batch_size)([x, x_])])
critic_z_model = Model([x, z],
    [critic_z(z), critic_z(z_), RandomWeightedAverage(batch_size)([z, z_])])
z_gen = Input(shape=(latent_dim, 1)); x_gen = Input(shape=shape)
x_gen_ = generator(z_gen); z_gen_ = encoder(x_gen); x_gen_rec = generator(z_gen_)
encoder_generator_model = Model([x_gen, z_gen],
    [critic_x(x_gen_), critic_z(z_gen_), x_gen_rec])
print("composite models ready")


#X = X.reshape((-1, shape[0], feat_dim))
X_ = np.copy(X_all)
fake  =  np.ones((batch_size, 1), dtype=np.float32)
valid = -np.ones((batch_size, 1), dtype=np.float32)
delta =  np.ones((batch_size, 1), dtype=np.float32)

for epoch in range(1, epochs + 1):
    np.random.shuffle(X_)
    g_loss, cx_loss, cz_loss = [], [], []
    mb_size = batch_size * n_critic
    num_mb = int(X_.shape[0] // mb_size)
    for i in range(num_mb):
        mb = X_[i * mb_size:(i + 1) * mb_size]
        critic_x.trainable = True;  critic_z.trainable = True
        generator.trainable = False; encoder.trainable = False
        for j in range(n_critic):
            xb = mb[j * batch_size:(j + 1) * batch_size]
            zb = np.random.normal(size=(batch_size, latent_dim, 1))
            cx_loss.append(critic_x_train_on_batch(xb, zb, valid, fake, delta,
                                                   critic_x_model, critic_x, optimizer))
            cz_loss.append(critic_z_train_on_batch(xb, zb, valid, fake, delta,
                                                   critic_z_model, critic_z, optimizer))
        critic_x.trainable = False; critic_z.trainable = False
        generator.trainable = True;  encoder.trainable = True
        g_loss.append(enc_gen_train_on_batch(xb, zb, valid,
                                             encoder_generator_model, optimizer))
    print('Epoch {}/{}, [Dx {}] [Dz {}] [G {}]'.format(
        epoch, epochs, np.mean(np.array(cx_loss), axis=0),
        np.mean(np.array(cz_loss), axis=0), np.mean(np.array(g_loss), axis=0)))

#critic_x.save_weights(os.path.join(ckpt_dir, 'critic_x_sub.h5'), save_format='h5')
#encoder.save_weights(os.path.join(ckpt_dir, 'encoder_sub.h5'), save_format='h5')
#generator.save_weights(os.path.join(ckpt_dir, 'generator_sub.h5'), save_format='h5')

#critic_z_model.save_weights(os.path.join(ckpt_dir, 'critic_z_model.h5'), save_format='h5')
#encoder_generator_model.save_weights(os.path.join(ckpt_dir, 'encoder_generator_model.h5'), save_format='h5')
#print("checkpoints saved to", ckpt_dir)

# 학습 완료 직후, train X가 살아있는 시점
y_hat_tr, critic_tr = predict(X_all, encoder, generator, critic_x, shape, feat_dim)

anomaly = Anomaly()          # ← 이 줄 추가
final_scores_train, _, _, _ = anomaly.score_anomalies(
    X_all, y_hat_tr, critic_tr, X_index, comb="mult")
final_scores_train = np.array(final_scores_train)

import os
os.makedirs('cache', exist_ok=True)
np.save('cache/final_scores_train.npy', final_scores_train)
print("train 점수 저장:", final_scores_train.mean(), final_scores_train.std(), final_scores_train.max())

import os
os.makedirs('cache', exist_ok=True)
np.save('cache/final_scores_train.npy', final_scores_train)
print("train 점수 저장:", final_scores_train.mean(), final_scores_train.std(), final_scores_train.max())



import pickle
weights_dict = {
    'critic_x': critic_x.get_weights(),
    'encoder': encoder.get_weights(),
    'generator': generator.get_weights(),
}
with open(os.path.join(ckpt_dir, 'sub_weights.pkl'), 'wb') as f:
    pickle.dump(weights_dict, f)
print("pkl 저장 완료, 시각:", os.path.getmtime(os.path.join(ckpt_dir, 'sub_weights.pkl')))



# %% [markdown]
# # 03 · 테스트 · 이상탐지 · 평가
# 
# 전처리된 **테스트 데이터**를 학습 모델로 진단. 파일만 바꿔 반복 실행.
# 설정은 **`config.py`**에서 로드(학습과 동일해야 함).
# 
# **PCA 모드 비교** (`config.py`의 `pca_mode`):
# - `'consistent'` : 학습에서 저장한 pca/scaler를 **transform만** (정석) — 좌표계 일치
# - `'original'`   : 테스트 데이터로 **각자 fit_transform** (원본 재현)
# 
# 같은 테스트 파일에 대해 두 모드로 각각 돌려 F-score를 비교하면 차이를 볼 수 있다.
# 
# > 평가용 `anomalies`(find_anomalies)와 시각화용 `anomalies_plot`(gt/pred)을 이름 분리 → 셀 재실행 안전.

# %%
from function_def import *
from parameter import CFG
import os, math, numpy as np, pandas as pd, joblib
import tensorflow as tf
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

# %% [markdown]
# ## 하이퍼파라미터 로드 (학습과 동일한 config)

# %%
# 학습에서 win_size 등을 바꿔 make_config 했다면 여기서도 동일하게 맞출 것
# from config import make_config
# CFG = make_config(win_size=30)

win_size     = CFG['win_size']
features_dim = CFG['features_dim']
feat_dim     = CFG['feat_dim']
latent_dim   = CFG['latent_dim']
batch_size   = CFG['batch_size']
k_size       = CFG['k_size']
lstm_units   = CFG['lstm_units']
drop_gen     = CFG['dropout_rate_gen']
crit_filters = CFG['critic_filters']
crit_drop    = CFG['critic_dropout']
diffs_n, lags_n, smooth_n = CFG['diffs_n'], CFG['lags_n'], CFG['smooth_n']
pca_mode = CFG['pca_mode']

shape                   = CFG['shape']
encoder_input_shape     = CFG['encoder_input_shape']
encoder_reshape_shape   = CFG['encoder_reshape_shape']
generator_input_shape   = CFG['generator_input_shape']
generator_reshape_shape = CFG['generator_reshape_shape']
critic_x_input_shape    = CFG['critic_x_input_shape']
critic_z_input_shape    = CFG['critic_z_input_shape']
ckpt_dir = CFG['ckpt_dir']
print("win_size=%d k_size=%d pca_mode=%s" % (win_size, k_size, pca_mode))

# %% [markdown]
# ## 설정 (테스트 파일 · 라벨)

# %%
TEST_CSV  = './data/preprocessed/test/Test01_OK_chg.csv'
LABEL_CSV = './data/preprocessed/test/Test01_OK_chg_Label.csv'
TITLE     = 'Test01_OK'
print("TEST:", TEST_CSV, "| mode:", pca_mode)

# %% [markdown]
# ## 1) 로드 → featurize → PCA/스케일 (모드 분기)
# 
# - **consistent**: 학습에서 저장한 pca/scaler 로드 후 `transform`만 (fit 안 함)
# - **original**  : 테스트 데이터로 `fit_transform`

# %%
df_test1 = pd.read_csv(TEST_CSV)
data_1 = diff_smooth_df(df_test1, lags_n, diffs_n, smooth_n)

if pca_mode == 'consistent':
    std_scaler = joblib.load(CFG['std_scaler_path'])  # 학습 표준화 재사용
    data_1 = std_scaler.transform(data_1)
    pca = joblib.load(CFG['pca_path'])        # 학습 PCA 재사용
    data = pca.transform(data_1)              # fit 없이 변환만
    print("PCA: 학습 축 재사용(transform)")
else:
    std_scaler = StandardScaler()
    data_1 = std_scaler.fit_transform(data_1)
    pca = PCA(n_components=features_dim)  # 테스트 데이터로 새로 fit
    data = pca.fit_transform(data_1)          # 테스트 데이터로 새로 fit
    print("PCA: 테스트 데이터로 fit_transform")

df_1 = []
for i in range(len(data)):
    row = [i + 1] + [data[i][jj] for jj in range(features_dim)]
    df_1.append(row)
df = pd.DataFrame(df_1)
df.columns = ['date'] + ['pca_%s' % str(i) for i in range(1, features_dim + 1)]
print("After PCA:", df.shape)

# %%
X, index = time_segments_aggregate(df, interval=1, time_column='date')
X = SimpleImputer().fit_transform(X)

if pca_mode == 'consistent':
    scaler = joblib.load(CFG['scaler_path'])  # 학습 스케일러 재사용
    X = scaler.transform(X)
else:
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

X, y, X_index, y_index = rolling_window_sequences(
    X, index, window_size=win_size, target_size=1, step_size=1, target_column=0)
print("after window:", X.shape)


print("=== split X ===")
print("shape:", X.shape)
print("mean/std:", X.mean(), X.std())
print("min/max:", X.min(), X.max())
print("첫 윈도우 첫 행:", X[0, 0, :])
# %% [markdown]
# ## 2) 모델 재구성 → 체크포인트 로드

# %%
encoder   = build_encoder_layer(encoder_input_shape, encoder_reshape_shape,
                                win_size=win_size, latent_dim=latent_dim)
generator = build_generator_layer(generator_input_shape, generator_reshape_shape,
                                  win_size=win_size, features_dim=features_dim,
                                  lstm_units=lstm_units, dropout_rate=drop_gen)
critic_x  = build_critic_x_layer(critic_x_input_shape, k_size=k_size,
                                 filters=crit_filters, dropout_rate=crit_drop)
critic_z  = build_critic_z_layer(critic_z_input_shape)

z = Input(shape=(latent_dim, 1)); x = Input(shape=shape)
x_ = generator(z); z_ = encoder(x)
critic_x_model = Model([x, z],
    [critic_x(x), critic_x(x_), RandomWeightedAverage(batch_size)([x, x_])])
critic_z_model = Model([x, z],
    [critic_z(z), critic_z(z_), RandomWeightedAverage(batch_size)([z, z_])])
z_gen = Input(shape=(latent_dim, 1)); x_gen = Input(shape=shape)
x_gen_ = generator(z_gen); z_gen_ = encoder(x_gen); x_gen_rec = generator(z_gen_)
encoder_generator_model = Model([x_gen, z_gen],
    [critic_x(x_gen_), critic_z(z_gen_), x_gen_rec])

before = critic_x.get_weights()[0].copy()
#critic_x.load_weights(os.path.join(ckpt_dir, 'critic_x_sub.h5'), by_name=True)
#encoder.load_weights(os.path.join(ckpt_dir, 'encoder_sub.h5'), by_name=True)
#generator.load_weights(os.path.join(ckpt_dir, 'generator_sub.h5'), by_name=True)
   

import pickle
pkl_path = os.path.join(ckpt_dir, 'sub_weights.pkl')
print("로드하는 pkl 시각:", os.path.getmtime(pkl_path))   # ← 방금 저장한 시각과 같아야!

with open(pkl_path, 'rb') as f:
    weights_dict = pickle.load(f)
critic_x.set_weights(weights_dict['critic_x'])
encoder.set_weights(weights_dict['encoder'])
generator.set_weights(weights_dict['generator'])

after = critic_x.get_weights()[0]

# 03_test에서, 모델 로드 다 한 직후에
# (로드 코드가 이 위에 있어야 함)
print("critic_x 가중치 로드됨?:", not np.allclose(before, after))
print("encoder 로드 확인:", encoder.get_weights()[0].shape)
print("generator 로드 확인:", generator.get_weights()[0].shape)
#import pickle

# --- 원래 로드 코드는 임시로 주석 처리 ---
# with open(os.path.join(ckpt_dir, 'sub_weights.pkl'), 'rb') as f:
#     weights_dict = pickle.load(f)

# --- notebook의 좋은 모델을 대신 로드 ---
#with open(os.path.join(ckpt_dir, 'notebook_good_model.pkl'), 'rb') as f:
   # weights_dict = pickle.load(f)
#print("★ notebook 모델을 로드함 (모델 통제 실험)")

# 로드 검증
#before = critic_x.get_weights()[0].copy()

# 로드 전 스냅샷
#enc_before = encoder.get_weights()[0].copy()
#gen_before = generator.get_weights()[0].copy()
#cx_before = critic_x.get_weights()[0].copy()

#critic_x.set_weights(weights_dict['critic_x'])
#encoder.set_weights(weights_dict['encoder'])
#generator.set_weights(weights_dict['generator'])
#after = critic_x.get_weights()[0]

#print("critic_x 로드됨?:", not np.allclose(before, after))


# 로드 후 검증 (셋 다)
#print("encoder 바뀜?:", not np.allclose(enc_before, encoder.get_weights()[0]))
#print("generator 바뀜?:", not np.allclose(gen_before, generator.get_weights()[0]))
#print("critic_x 바뀜?:", not np.allclose(cx_before, critic_x.get_weights()[0]))


# %% [markdown]
# ## 3) 예측 → 이상 점수 → 이상 구간

# %%
known_anomalies = pd.read_csv(LABEL_CSV)
y_hat, critic = predict(X, encoder, generator, critic_x, shape, feat_dim)
print("y_hat:", y_hat.shape, "| critic:", critic.shape)

anomaly = Anomaly()
final_scores, true_index, true, predictions = anomaly.score_anomalies(
    X, y_hat, critic, X_index, comb="mult")
final_scores = np.array(final_scores)
anomalies = anomaly.find_anomalies(final_scores, true_index)  # [[start, stop, score], ...]
print("anomalies:", anomalies)


# %% [markdown]
# ## 4) 성능 평가 (재현율·F-score 중심)

# %%
pred_length = len(final_scores)
avg = np.average(final_scores)
sigma = math.sqrt(np.sum((final_scores - avg) ** 2) / len(final_scores))
Z_score1 = (final_scores - avg) / sigma

pred_bin = [0] * pred_length
for a in anomalies:
    start, stop = int(a[0]), int(a[1])
    for k in range(start - 1, stop):
        if 0 <= k < pred_length:
            pred_bin[k] = 1

gt   = np.array(known_anomalies['label'][:pred_length])
pred = np.array(pred_bin)
tp = int(np.sum((pred == 1) & (gt == 1))); tn = int(np.sum((pred == 0) & (gt == 0)))
fp = int(np.sum((pred == 1) & (gt == 0))); fn = int(np.sum((pred == 0) & (gt == 1)))

Accuracy  = (tp + tn) / len(pred)
Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
Recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
F1_score  = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
print("[mode=%s] Accuracy:%.4f Precision:%.4f Recall:%.4f F-score:%.4f"
      % (pca_mode, Accuracy, Precision, Recall, F1_score))
print("TP:%d FP:%d FN:%d TN:%d" % (tp, fp, fn, tn))

# 학습·점수 계산 끝난 직후 (한 번만)
os.makedirs('cache', exist_ok=True)

np.save('cache/final_scores.npy', final_scores)
np.save('cache/true_index.npy', true_index)
np.save('cache/X.npy', X)
np.save('cache/gt.npy', gt)
print("중간결과 저장 완료")

# 정답 이상구간 위치
gt_idx = gt.nonzero()[0]
print("정답 이상 인덱스(앞부분):", gt_idx[:20])

# 그 구간에서 점수가 솟았나?
early = gt_idx[gt_idx < 850]   # 초반부만
print("초반 이상구간 final_scores:", final_scores[early])
print("전체 평균/최대:", final_scores.mean(), final_scores.max())

final_scores_train = np.load('cache/final_scores_train.npy')

print("train 점수 mean/std/max:", 
      final_scores_train.mean(), final_scores_train.std(), final_scores_train.max())
print("test05 초반 이상구간 점수:", final_scores[early])
# %% [markdown]
# ## 5) 정답/예측 구간 (시각화용 · 이름 분리)

# %%
def to_spans(binary):
    spans, in_seq, begin = [], False, 0
    for k, v in enumerate(binary):
        if v == 1 and not in_seq:
            in_seq, begin = True, k
        elif v == 0 and in_seq:
            spans.append((begin, k - 1)); in_seq = False
    if in_seq:
        spans.append((begin, len(binary) - 1))
    return spans

anomaly_gt   = to_spans(gt)
anomaly_pred = to_spans(pred)
anomalies_plot = [anomaly_gt, anomaly_pred]
length_anom = len(pred)
print("gt  :", anomaly_gt)
print("pred:", anomaly_pred)

print("예측 구간:", anomalies)
print("정답 구간:", gt.nonzero())  # 또는 to_spans(gt)

# 03_test에서 끝까지

# %% [markdown]
# ## 6) 시각화

# %%
register_matplotlib_converters()
np.random.seed(0)
max_len = min(length_anom - 10, len(X))
t = range(max_len)
Z2 = Z_score1[:max_len]
X_sig = np.array([X[kk, 1] for kk in range(max_len)])

plt.figure(figsize=(30, 12))
plt.plot(t, 3 * X_sig[:, 0], label='3*PCA1')
plt.plot(t, 3 * X_sig[:, 1], label='3*PCA2')
plt.plot(t, Z2, label='Z score')
plt.legend(loc=0, fontsize=30)
colors = ['red'] + ['blue'] * (len(anomalies_plot) - 1)
for i, anom_span in enumerate(anomalies_plot):
    for anom in anom_span:
        plt.axvspan(anom[0], anom[1], color=colors[i], alpha=0.2)
plt.title(' {} ({}) : Red=True, Blue=Pred'.format(TITLE, pca_mode), size=34)
plt.ylabel('PCA1, PCA2, Z_score', size=30); plt.xlabel('Time', size=30)
plt.xticks(size=26); plt.yticks(size=26)
plt.xlim([t[0], t[-1]]); plt.show()

# %% [markdown]
# ## 추가 검증을 위한 코드베이스

# %%
import pandas as pd

train_cols = set(pd.read_csv('./data/preprocessed/train/1000_chg.csv').columns)
test_cols  = set(pd.read_csv('./data/preprocessed/test/Test07_NG_dchg.csv').columns)

print("학습 컬럼 수:", len(train_cols))
print("테스트 컬럼 수:", len(test_cols))
print("공통 컬럼 수:", len(train_cols & test_cols))
print("학습에만 있는 컬럼:", sorted(train_cols - test_cols)[:10])
print("테스트에만 있는 컬럼:", sorted(test_cols - train_cols)[:10])
