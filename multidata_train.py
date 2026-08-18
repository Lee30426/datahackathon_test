import os
# 결정론 옵션 (필요 시 주석 해제)
#os.environ['PYTHONHASHSEED'] = '0'
#os.environ['TF_DETERMINISTIC_OPS'] = '1'    # tensorflow 연산 결정론화
#os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # cuDNN 연산 결정론화

from function_def import *
from parameter import CFG
import numpy as np, pandas as pd, joblib, pickle
import tensorflow as tf

# 재현용 seed 고정 (선택)
#np.random.seed(42); tf.random.set_seed(42)
#tf.keras.utils.set_random_seed(42)
#tf.config.experimental.enable_op_determinism()

# ---------------------------------------------------------------
# CFG 언패킹
# ---------------------------------------------------------------
win_size      = CFG['win_size']
features_dim  = CFG['features_dim']       # PCA 축 개수 (하드코딩 제거, CFG 단일 출처)
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

# ---------------------------------------------------------------
# 0. 파일 목록: 1000~1050 (전처리 완료 데이터 — raw_data 아님)
# ---------------------------------------------------------------
DATA_DIR = './data/preprocessed/train'
file_ids = range(1000, 1051)
files = [os.path.join(DATA_DIR, f'{fid}_chg.csv') for fid in file_ids]

missing = [f for f in files if not os.path.exists(f)]
if missing:
    print(f"[경고] 없는 파일 {len(missing)}개:", [os.path.basename(m) for m in missing])
files = [f for f in files if os.path.exists(f)]
print(f"불러올 파일 수: {len(files)}개")
assert len(files) > 0, "불러올 파일이 없습니다. 경로를 확인하세요."

# ---------------------------------------------------------------
# 1. 각 파일 읽기 + 컬럼 정합성 검사 (concat 전에 반드시)
# ---------------------------------------------------------------
raw_dfs = [pd.read_csv(f) for f in files]

ref_cols = list(raw_dfs[0].columns)
for f, df in zip(files, raw_dfs):
    if list(df.columns) != ref_cols:
        raise ValueError(
            f"컬럼 불일치: {os.path.basename(f)}\n"
            f"  기준: {ref_cols}\n  실제: {list(df.columns)}"
        )

dfs = [diff_smooth_df(df, lags_n, diffs_n, smooth_n) for df in raw_dfs]
print("각 파일 shape:", [d.shape for d in dfs])

# ---------------------------------------------------------------
# 2. 표준화 + PCA는 "합친 데이터"로 fit (공통 좌표계)
# ---------------------------------------------------------------
df_concat = pd.concat(dfs, axis=0, ignore_index=True)

# (안전장치) 스케일러엔 숫자만 들어가야 함. 문자열(날짜 등)이 있으면 여기서 명확히 알림
non_numeric = df_concat.select_dtypes(exclude=[np.number]).columns.tolist()
assert not non_numeric, f"수치가 아닌 컬럼이 있습니다(전처리 확인 필요): {non_numeric}"

scaler_std = StandardScaler()
scaled_concat = scaler_std.fit_transform(df_concat)     # 표준화 (스케일 차이 제거)

pca = PCA(n_components=features_dim)
pca.fit(scaled_concat)                                   # 공통 축 학습 (fit만)

print("explained_variance_ratio:", pca.explained_variance_ratio_)
print("누적 설명력:", np.cumsum(pca.explained_variance_ratio_))

# ---------------------------------------------------------------
# 3. 각 파일을 "따로" 변환 + aggregate (파일 경계 오염 방지)
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

    X_list.append((Xf, idxf))

# ---------------------------------------------------------------
# 4. MinMax 스케일은 "합친 데이터" 기준으로 fit
#    (모델 입력 범위 -1~1을 51개 파일 전체 기준으로 통일)
# ---------------------------------------------------------------
X_all_raw = np.concatenate([x for x, _ in X_list], axis=0)
scaler_mm = MinMaxScaler(feature_range=(-1, 1))
scaler_mm.fit(X_all_raw)                                 # 합친 기준으로 fit

# ---------------------------------------------------------------
# 4-1. consistent 모드: 학습에서 fit한 변환기 저장 → 테스트가 transform으로 재사용
#      (이게 없으면 03_test의 joblib.load가 낡은 파일/에러 → 좌표계 불일치)
# ---------------------------------------------------------------
if pca_mode == 'consistent':
    joblib.dump(scaler_std, CFG['std_scaler_path'])     # 표준화
    joblib.dump(pca,        CFG['pca_path'])            # PCA 축
    joblib.dump(scaler_mm,  CFG['scaler_path'])         # MinMax (test는 이 키로 로드)
    print("저장: std/pca/mm 변환기 →", CFG['std_scaler_path'], CFG['pca_path'], CFG['scaler_path'])
else:
    print("original 모드: 변환기 저장 안 함(테스트가 각자 fit_transform)")

# ---------------------------------------------------------------
# 5. 각 파일을 스케일 + 윈도우 → 윈도우/인덱스 모두 합치기
#    - 이전 코드의 '이중 윈도잉'과 'X_index 미정의'를 여기서 해결
#    - offset으로 파일 간 인덱스 겹침 방지 (파일마다 date가 1부터 재시작하므로)
# ---------------------------------------------------------------
X_windows, idx_windows = [], []
offset = 0
for (Xf, idxf) in X_list:
    Xf_scaled = scaler_mm.transform(Xf)                 # 공통 스케일 적용
    Xw, y, Xw_idx, y_idx = rolling_window_sequences(
        Xf_scaled, idxf, window_size=win_size,
        target_size=1, step_size=1, target_column=0)
    X_windows.append(Xw)
    idx_windows.append(Xw_idx + offset)                 # 전역 단조 인덱스로 이동
    offset += len(Xf)                                   # 다음 파일은 이만큼 뒤에서 시작

X_all   = np.concatenate(X_windows, axis=0)             # 윈도우 합치기
X_index = np.concatenate(idx_windows, axis=0)           # 점수계산이 요구하는 인덱스
print("최종 학습 윈도우:", X_all.shape, "| index:", X_index.shape)

# ---------------------------------------------------------------
# 6. 네트워크 & 복합모델 구성
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 7. 학습 루프 (셔플이 51개 파일을 골고루 섞어줌)
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 8. 학습 직후 train 이상점수 계산 + 저장 (X_index 필요)
# ---------------------------------------------------------------
y_hat_tr, critic_tr = predict(X_all, encoder, generator, critic_x, shape, feat_dim)

anomaly = Anomaly()
final_scores_train, _, _, _ = anomaly.score_anomalies(
    X_all, y_hat_tr, critic_tr, X_index, comb="mult")
final_scores_train = np.array(final_scores_train)

os.makedirs('cache', exist_ok=True)
np.save('cache/final_scores_train.npy', final_scores_train)   # 중복 저장 블록 제거 (1회만)
print("train 점수 저장:", final_scores_train.mean(), final_scores_train.std(), final_scores_train.max())

# ---------------------------------------------------------------
# 9. 가중치 저장 (pkl) — 03_test가 set_weights로 로드
# ---------------------------------------------------------------
weights_dict = {
    'critic_x': critic_x.get_weights(),
    'encoder': encoder.get_weights(),
    'generator': generator.get_weights(),
}
with open(os.path.join(ckpt_dir, 'sub_weights.pkl'), 'wb') as f:
    pickle.dump(weights_dict, f)
print("pkl 저장 완료, 시각:", os.path.getmtime(os.path.join(ckpt_dir, 'sub_weights.pkl')))