# config.py
# ------------------------------------------------------------------
# 학습(02_train)과 테스트(03_test)가 공유하는 설정을 한 곳에 모은다.
# 하이퍼파라미터를 바꾸려면 "여기만" 수정하면 학습·테스트에 동시 반영된다.
#
# 사용:  from config import CFG        (또는 from config import *)
#        win_size = CFG['win_size']    처럼 꺼내 쓴다.
#
# 노트북에서 임시로 덮어쓰고 싶으면 import 후 CFG['win_size'] = 30 처럼 바꾸면 된다.
# ------------------------------------------------------------------


def make_config(win_size=50    , features_dim=3, latent_dim=20,
                batch_size=128, n_critic=15, epochs=30,
                learning_rate=0.00005,
                lstm_units=64, dropout_rate_gen=0.2,
                critic_filters=64, critic_dropout=0.25,
                diffs_n=0, lags_n=0, smooth_n=0,
                pca_mode='consistent', ):
    """설정 딕셔너리를 생성한다.

    파생 값(shape 등)은 win_size / features_dim / latent_dim 으로부터 자동 계산되므로,
    핵심 값만 바꿔도 나머지가 일관되게 따라간다.

    Args (튜닝 대상):
        win_size       : 슬라이딩 윈도우 길이 (시퀀스 길이)
        features_dim   : PCA 축소 차원 (입력 피처 수)
        latent_dim     : 잠재공간 차원
        batch_size     : 배치 크기
        n_critic       : 생성자 1회당 판별자 학습 횟수 (WGAN)
        epochs         : 학습 반복 횟수
        learning_rate  : Adam 학습률
        lstm_units     : generator 내부 LSTM 유닛 수
        dropout_rate_gen: generator 드롭아웃 비율
        critic_filters : critic_x Conv1D 필터 수
        critic_dropout : critic_x 드롭아웃 비율
        diffs_n/lags_n/smooth_n : diff_smooth_df featurize 옵션
        pca_mode       : 'consistent' -> 학습 fit 저장, 테스트 transform (정석)
                         'original'   -> 학습/테스트 각자 fit_transform (원본 재현)
    """
    # critic_x 커널 크기: 원본 로직 (win_size>=30 이면 5, 아니면 2)
    k_size = 5 if win_size >= 30 else 2
    feat_dim = features_dim

    cfg = {
        # --- 튜닝 대상 하이퍼파라미터 ---
        'win_size': win_size,
        'features_dim': features_dim,
        'feat_dim': feat_dim,
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'n_critic': n_critic,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'k_size': k_size,
        'lstm_units': lstm_units,
        'dropout_rate_gen': dropout_rate_gen,
        'critic_filters': critic_filters,
        'critic_dropout': critic_dropout,

        # --- featurize 옵션 ---
        'diffs_n': diffs_n,
        'lags_n': lags_n,
        'smooth_n': smooth_n,

        # --- PCA 처리 방식 스위치 ---
        'pca_mode': pca_mode,

        # --- 파생 shape (자동 계산; 수정 불필요) ---
        'shape': [win_size, features_dim],
        'encoder_input_shape': [win_size, features_dim],
        'encoder_reshape_shape': [latent_dim, 1],
        'generator_input_shape': [latent_dim, 1],
        'generator_reshape_shape': [win_size, 1],
        'critic_x_input_shape': [win_size, features_dim],
        'critic_z_input_shape': [latent_dim, 1],

        # --- 경로 ---
        'ckpt_dir': 'checkpoints',
        'pca_path': 'checkpoints/pca.joblib',      # 정합 모드에서 학습 PCA 저장/로드
        'scaler_path': 'checkpoints/scaler.joblib',# 정합 모드에서 학습 스케일러 저장/로드
        'std_scaler_path': 'checkpoints/std_scaler.joblib'
    }
    return cfg


# 기본 설정 인스턴스 (노트북에서 from config import CFG 로 사용)
CFG = make_config()
