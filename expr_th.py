# experiment_threshold.py
# ==================================================================
# 임계값 실험 전용 스크립트
# ------------------------------------------------------------------
# 목적: 학습·모델로드·GAN추론(predict)·점수계산(score_anomalies)을
#       전부 건너뛰고, 저장된 점수만 불러와 "임계값·후처리"만 반복 실험한다.
#
# 전제 (원본 test_combination.py에서 한 번만 저장해둘 것):
#   cache/final_scores.npy         : 테스트 이상점수 (358줄 산출물)
#   cache/true_index.npy           : 점수의 시간 인덱스 (358줄 산출물)
#   cache/X.npy                    : 윈도우 배열 (시각화용, 82/259줄 산출물)
#   cache/gt.npy                   : 정답 라벨 전체 (known_anomalies['label'])
#   cache/final_scores_train.npy   : train(정상) 점수 (train 기준 임계값용)
#
# 사용:
#   1) evaluate(...) 파라미터만 바꿔가며 반복 호출
#   2) 여러 설정을 한 번에 비교하려면 맨 아래 sweep 루프 사용
# ==================================================================

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

# function_def.py 안에 Anomaly 클래스가 있으므로 그대로 재사용
from function_def import Anomaly

# ------------------------------------------------------------------
# 0) 저장된 점수 불러오기 (비싼 계산은 여기서 전부 생략)
# ------------------------------------------------------------------
CACHE = 'cache'
final_scores       = np.load(os.path.join(CACHE, 'final_scores.npy'))
true_index         = np.load(os.path.join(CACHE, 'true_index.npy'))
X                  = np.load(os.path.join(CACHE, 'X.npy'))
gt_full            = np.load(os.path.join(CACHE, 'gt.npy'))
# train 점수는 없을 수도 있으니 있으면 로드
_train_path = os.path.join(CACHE, 'final_scores_train.npy')
final_scores_train = np.load(_train_path) if os.path.exists(_train_path) else None

final_scores = np.array(final_scores)
pred_length  = len(final_scores)
# gt를 점수 길이에 맞춰 자름 (원본 380줄과 동일한 정합 방식)
gt = np.array(gt_full[:pred_length])

TITLE = 'experiment'

print("=== 로드 완료 ===")
print("len(final_scores):", pred_length, "| len(gt_full):", len(gt_full))
print("test  점수 mean/std/max: %.4f / %.4f / %.4f"
      % (final_scores.mean(), final_scores.std(), final_scores.max()))
if final_scores_train is not None:
    print("train 점수 mean/std/max: %.4f / %.4f / %.4f"
          % (final_scores_train.mean(), final_scores_train.std(),
             final_scores_train.max()))
print()


# ------------------------------------------------------------------
# 유틸: 이진 예측 → 구간 리스트 (원본 397~406줄 그대로)
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# 유틸: anomalies 구간 리스트 → 이진 예측 (원본 373~378줄 그대로)
# ------------------------------------------------------------------
def anomalies_to_bin(anomalies, length):
    pred_bin = [0] * length
    for a in anomalies:
        start, stop = int(a[0]), int(a[1])
        for k in range(start - 1, stop):
            if 0 <= k < length:
                pred_bin[k] = 1
    return np.array(pred_bin)


# ------------------------------------------------------------------
# 핵심: 하나의 임계값 설정을 평가 (F-score까지 반환)
# ------------------------------------------------------------------
def evaluate(window_size=None, window_step_size=150,
             anomaly_padding=50, min_percent=0.1,
             fixed_threshold=True,
             train_based=True, k_train=1.50,
             verbose=True, plot=False):
    """
    두 가지 임계값 방식 지원:

    (1) train_based=False  → 원본 방식 (테스트 점수 자체로 find_anomalies)
        - window_size      : 국소 임계값용 윈도우 (None이면 전체 하나)
        - window_step_size : 윈도우 이동 간격
        - anomaly_padding  : 이상 전후 확장 (짧은 이상은 줄이는 게 유리)
        - fixed_threshold  : True면 mean+3std 고정, False면 fmin 최적화

    (2) train_based=True   → train(정상) 점수 기준 고정 임계값
        - threshold = train.mean + k_train * train.std
        - test에 이상이 많아도 기준선이 안 흔들림
    """
    if train_based:
        if final_scores_train is None:
            raise RuntimeError("final_scores_train.npy가 없어 train 기준을 못 씀")
        mu, sigma = final_scores_train.mean(), final_scores_train.std()
        threshold = mu + k_train * sigma
        pred = (final_scores > threshold).astype(int)
        thr_desc = "train기준 k=%.10f thr=%.4f" % (k_train, threshold)
    else:
        anomaly = Anomaly()
        anomalies = anomaly.find_anomalies(
            final_scores, true_index,
            window_size=window_size,
            window_step_size=window_step_size,
            anomaly_padding=anomaly_padding,
            min_percent=min_percent,
            fixed_threshold=fixed_threshold)
        pred = anomalies_to_bin(anomalies, pred_length)
        thr_desc = ("test기준 win=%s step=%s pad=%d fixed=%s"
                    % (window_size, window_step_size,
                       anomaly_padding, fixed_threshold))

    # --- 성능 지표 (원본 382~388줄과 동일) ---
    tp = int(np.sum((pred == 1) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    Accuracy  = (tp + tn) / len(pred)
    Precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    Recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1_score  = (2 * Precision * Recall / (Precision + Recall)
                 if (Precision + Recall) > 0 else 0)

    result = {'F': F1_score, 'P': Precision, 'R': Recall, 'Acc': Accuracy,
              'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'desc': thr_desc}

    if verbose:
        print("[%s] F=%.4f P=%.4f R=%.4f | TP=%d FP=%d FN=%d TN=%d"
              % (thr_desc, F1_score, Precision, Recall, tp, fp, fn, tn))

    if plot:
        _plot(pred)

    return result


# ------------------------------------------------------------------
# 시각화 (원본 424~443줄 기반, 저장 데이터로 동작하게 정리)
# ------------------------------------------------------------------
def _plot(pred):
    register_matplotlib_converters()
    np.random.seed(0)

    # Z-score (원본 369~371줄)
    avg = np.average(final_scores)
    sigma = math.sqrt(np.sum((final_scores - avg) ** 2) / len(final_scores))
    Z_score1 = (final_scores - avg) / sigma

    # max_len: 점수·X 둘 다 안 넘치게 (win_size 바뀌어도 안전)
    max_len = min(len(pred) - 10, len(X))
    t = range(max_len)
    Z2 = Z_score1[:max_len]
    X_sig = np.array([X[kk, 1] for kk in range(max_len)])

    anomaly_gt   = to_spans(gt)
    anomaly_pred = to_spans(pred)
    anomalies_plot = [anomaly_gt, anomaly_pred]

    plt.figure(figsize=(30, 12))
    plt.plot(t, 3 * X_sig[:, 0], label='3*PCA1')
    plt.plot(t, 3 * X_sig[:, 1], label='3*PCA2')
    plt.plot(t, Z2, label='Z score')
    plt.legend(loc=0, fontsize=30)
    colors = ['red'] + ['blue'] * (len(anomalies_plot) - 1)
    for i, anom_span in enumerate(anomalies_plot):
        for anom in anom_span:
            plt.axvspan(anom[0], anom[1], color=colors[i], alpha=0.2)
    plt.title(' {} : Red=True, Blue=Pred'.format(TITLE), size=34)
    plt.ylabel('PCA1, PCA2, Z_score', size=30)
    plt.xlabel('Time', size=30)
    plt.xticks(size=26); plt.yticks(size=26)
    plt.xlim([t[0], t[-1]]); plt.show()


# ==================================================================
# 실험 실행부 — 여기만 바꿔가며 반복
# ==================================================================
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # --- (A) 단일 실험: 파라미터 바꿔가며 가장 자주 쓰는 형태 ---
    print("\n----- 단일 실험 -----")
    evaluate(train_based=True, k_train=3.0, plot=True)   # train 기준
    evaluate(window_size=None, anomaly_padding=50)       # 원본 기본값(테스트 기준)

    # --- (B) 스윕: 여러 설정 한 번에 비교 ---
    print("\n----- 스윕 (테스트 기준) -----")
    best = None
    for ws in [None, 300, 500, 1000]:
        for pad in [10, 20, 50]:
            r = evaluate(window_size=ws, anomaly_padding=pad, verbose=True)
            if best is None or r['F'] > best['F']:
                best = r

    print("\n----- 스윕 (train 기준 · k 조절) -----")
    if final_scores_train is not None:
        for k in [1.35, 1.37, 1.4, 1.42, 1.45, 1.5, 1.75, 2]:
            r = evaluate(train_based=True, k_train=k, verbose=True)
            if r['F'] > best['F']:
                best = r

    print("\n=== BEST ===")
    print(best)