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
TEST_CSV  = './data/preprocessed/test/Test07_NG_dchg.csv'
LABEL_CSV = './data/preprocessed/test/Test07_NG_dchg_Label.csv'
TITLE     = 'Test07_NG'
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
    std_scaler = joblib.load(CFG['std_scaler_path'])   # 학습 표준화 재사용
    data_1 = std_scaler.transform(data_1)              # ★ PCA 전에 표준화 (학습과 동일 순서)
    pca = joblib.load(CFG['pca_path'])
    data = pca.transform(data_1)
    print("PCA: 학습 축 재사용(transform)")
else:
    std_scaler = StandardScaler()
    data_1 = std_scaler.fit_transform(data_1)
    pca = PCA(n_components=features_dim)
    data = pca.fit_transform(data_1)
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

#03_test에서, 모델 로드 다 한 직후에
#(로드 코드가 이 위에 있어야 함)
print("critic_x 가중치 로드됨?:", not np.allclose(before, after))
print("encoder 로드 확인:", encoder.get_weights()[0].shape)
print("generator 로드 확인:", generator.get_weights()[0].shape)
#import pickle

# --- 원래 로드 코드는 임시로 주석 처리 ---
# with open(os.path.join(ckpt_dir, 'sub_weights.pkl'), 'rb') as f:
#     weights_dict = pickle.load(f)

# --- notebook의 좋은 모델을 대신 로드 ---
#with open(os.path.join(ckpt_dir, 'notebook_good_model.pkl'), 'rb') as f:
 #   weights_dict = pickle.load(f)
#print("★ notebook 모델을 로드함 (모델 통제 실험)")

# 로드 검증
# before = critic_x.get_weights()[0].copy()

# # 로드 전 스냅샷
# enc_before = encoder.get_weights()[0].copy()
# gen_before = generator.get_weights()[0].copy()
# cx_before = critic_x.get_weights()[0].copy()

# critic_x.set_weights(weights_dict['critic_x'])
# encoder.set_weights(weights_dict['encoder'])
# generator.set_weights(weights_dict['generator'])
# after = critic_x.get_weights()[0]

# print("critic_x 로드됨?:", not np.allclose(before, after))


# # 로드 후 검증 (셋 다)
# print("encoder 바뀜?:", not np.allclose(enc_before, encoder.get_weights()[0]))
# print("generator 바뀜?:", not np.allclose(gen_before, generator.get_weights()[0]))
# print("critic_x 바뀜?:", not np.allclose(cx_before, critic_x.get_weights()[0]))


# %% [markdown]
# ## 3) 예측 → 이상 점수 → 이상 구간

# %%
known_anomalies = pd.read_csv(LABEL_CSV)
y_hat, critic = predict(X, encoder, generator, critic_x, shape, feat_dim)
print("y_hat:", y_hat.shape, "| critic:", critic.shape)

anomaly = Anomaly()
final_scores, true_index, true, predictions = anomaly.score_anomalies(
    X, y_hat, critic, X_index, comb="mult")  # ★ 학습(multidata_train.py)과 동일한 comb로 통일 — 임계값 스케일 일치
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

# find_anomalies()가 찾은 구간(패딩+pruning까지 적용된 결과)을 실제 평가에 사용
# (이전엔 anomalies를 계산만 해두고 print만 하고, 평가는 별도의 단순 point-wise threshold로 했었음 — 불일치 수정)
pred_bin = [0] * pred_length
for a in anomalies:
    start, stop = int(a[0]), int(a[1])
    for k in range(start - 1, stop):
        if 0 <= k < pred_length:
            pred_bin[k] = 1

final_scores_train = np.load('cache/final_scores_train.npy')   # 학습이 저장해둔 정상 점수
K_TRAIN = 1.42                                                  # ★ 시각화 기준선(train thr)용 — 평가에는 미사용
mu, sigma = final_scores_train.mean(), final_scores_train.std()
threshold = mu + K_TRAIN * sigma
print("train 기준선(참고, 시각화용): mu=%.4f sigma=%.4f thr=%.4f (k=%.2f)"
      % (mu, sigma, threshold, K_TRAIN))

gt   = np.array(known_anomalies['label'][:pred_length])
pred = np.array(pred_bin)
q
tp = int(np.sum((pred == 1) & (gt == 1))); tn = int(np.sum((pred == 0) & (gt == 0)))
fp = int(np.sum((pred == 1) & (gt == 0))); fn = int(np.sum((pred == 0) & (gt == 1)))

Accuracy  = (tp + tn) / len(pred)
Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
Recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
F1_score  = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
print("[mode=%s] Accuracy:%.4f Precision:%.4f Recall:%.4f F-score:%.4f"
      % (pca_mode, Accuracy, Precision, Recall, F1_score))
print("TP:%d FP:%d FN:%d TN:%d" % (tp, fp, fn, tn))


os.makedirs('cache', exist_ok=True)
np.save('cache/final_scores.npy', final_scores)
np.save('cache/true_index.npy',  true_index)
np.save('cache/X.npy',           X)
np.save('cache/gt.npy',          gt)
print("임계값 실험용 캐시 저장 완료")

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

print("test")
print("test2")
# 03_test에서 끝까지

# %% [markdown]
# ## 6) 시각화

# %%
#--------------------------------------------------------------------------------------------------------------------------------
# register_matplotlib_converters()
# np.random.seed(0)
# max_len = min(length_anom - 10, len(X))
# t = range(max_len)
# Z2 = Z_score1[:max_len]
# X_sig = np.array([X[kk, 1] for kk in range(max_len)])

# plt.figure(figsize=(30, 12))
# plt.plot(t, 3 * X_sig[:, 0], label='3*PCA1')
# plt.plot(t, 3 * X_sig[:, 1], label='3*PCA2')
# plt.plot(t, Z2, label='Z score')
# plt.legend(loc=0, fontsize=30)
# colors = ['red'] + ['blue'] * (len(anomalies_plot) - 1)
# for i, anom_span in enumerate(anomalies_plot):
#     for anom in anom_span:
#         plt.axvspan(anom[0], anom[1], color=colors[i], alpha=0.2)
# plt.title(' {} ({}) : Red=True, Blue=Pred'.format(TITLE, pca_mode), size=34)
# plt.ylabel('PCA1, PCA2, Z_score', size=30); plt.xlabel('Time', size=30)
# plt.xticks(size=26); plt.yticks(size=26)
# plt.xlim([t[0], t[-1]]); plt.show()
#----------------------------------------------------------------------------------------------------------------------------
register_matplotlib_converters()
np.random.seed(0)
max_len = min(length_anom - 10, len(X))
t = range(max_len)
Z2 = Z_score1[:max_len]
X_sig = np.array([X[kk, 1] for kk in range(max_len)])
final_scores_plot = final_scores[:max_len]                # ★ 추가: 원점수(절대값)

colors = ['red'] + ['blue'] * (len(anomalies_plot) - 1)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 20), sharex=True)

# ---- 위: 기존 그래프 (PCA + Z-score, 상대값) ----
ax1.plot(t, 3 * X_sig[:, 0], label='3*PCA1')
ax1.plot(t, 3 * X_sig[:, 1], label='3*PCA2')
ax1.plot(t, Z2, label='Z score')
ax1.legend(loc=0, fontsize=26)
for i, anom_span in enumerate(anomalies_plot):
    for anom in anom_span:
        ax1.axvspan(anom[0], anom[1], color=colors[i], alpha=0.2)
ax1.set_title(' {} ({}) : Red=True, Blue=Pred'.format(TITLE, pca_mode), size=34)
ax1.set_ylabel('PCA1, PCA2, Z_score', size=30)
ax1.tick_params(labelsize=26)

# ---- 아래: final_scores 원점수 + train 기준선 (절대값, 실제 판정 근거) ----
ax2.plot(t, final_scores_plot, label='final_scores', color='purple')
ax2.axhline(threshold, color='black', linestyle='--', linewidth=2,
            label=f'train thr={threshold:.3f}')
ax2.axhline(final_scores_train.mean(), color='gray', linestyle=':', linewidth=2,
            label='train avg')
ax2.fill_between(t,
                  final_scores_train.mean() - final_scores_train.std(),
                  final_scores_train.mean() + final_scores_train.std(),
                  color='gray', alpha=0.15, label='train ±1σ')
for i, anom_span in enumerate(anomalies_plot):
    for anom in anom_span:
        ax2.axvspan(anom[0], anom[1], color=colors[i], alpha=0.2)
ax2.legend(loc=0, fontsize=26)
ax2.set_ylabel('final_scores (절대값)', size=30)
ax2.set_xlabel('Time', size=30)
ax2.tick_params(labelsize=26)

plt.xlim([t[0], t[-1]])
plt.tight_layout()
plt.show()

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
