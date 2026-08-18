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

TRAIN_CSV = './data/preprocessed/train/1000_chg.csv'
print("TRAIN_CSV:", TRAIN_CSV)

df_train_0 = pd.read_csv(TRAIN_CSV)
data_1 = diff_smooth_df(df_train_0, lags_n, diffs_n, smooth_n)

pca = PCA(n_components=features_dim)
data = pca.fit_transform(data_1)          # 학습 데이터로 축 확정(fit) + 변환

df_1 = []
for i in range(len(data)):
    row = [i + 1] + [data[i][jj] for jj in range(features_dim)]
    df_1.append(row)
df = pd.DataFrame(df_1)
df.columns = ['date'] + ['pca_%s' % str(i) for i in range(1, features_dim + 1)]
print("After PCA:", df.shape)

X, index = time_segments_aggregate(df, interval=1, time_column='date')
X = SimpleImputer().fit_transform(X)

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)               # 학습 데이터로 스케일 범위 확정(fit)

# 정합 모드: 학습에서 fit한 pca/scaler 저장 -> 테스트가 transform 으로 재사용
if pca_mode == 'consistent':
    joblib.dump(pca, CFG['pca_path'])
    joblib.dump(scaler, CFG['scaler_path'])
    print("saved pca/scaler ->", CFG['pca_path'], CFG['scaler_path'])
else:
    print("original 모드: pca/scaler 저장 안 함(테스트가 각자 fit_transform)")

X, y, X_index, y_index = rolling_window_sequences(
    X, index, window_size=win_size, target_size=1, step_size=1, target_column=0)
print("after window:", X.shape)

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


X = X.reshape((-1, shape[0], feat_dim))
X_ = np.copy(X)
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

import pickle
weights_dict = {
    'critic_x': critic_x.get_weights(),
    'encoder': encoder.get_weights(),
    'generator': generator.get_weights(),
}
with open(os.path.join(ckpt_dir, 'sub_weights.pkl'), 'wb') as f:
    pickle.dump(weights_dict, f)
print("pkl 저장 완료, 시각:", os.path.getmtime(os.path.join(ckpt_dir, 'sub_weights.pkl')))