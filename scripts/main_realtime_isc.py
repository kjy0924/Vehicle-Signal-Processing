import sys
import json
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.autoencoder import PSDAutoencoder
from utils.signal_processing import compute_welch_psd, scale_psd_for_autoencoder
from utils.pipeline import run_ekf
from utils.data import load_right_block, add_coulomb_counted_soc
from utils.ocv import build_ocv_table_from_cc

DATASET_ROOT = PROJECT_ROOT / "voltage_prediction_and_ISC_detection-V1.0" / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
ISC_DST_DIR = DATASET_ROOT / "NCM811_ISC_TEST" / "DST"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC" / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"
SAVE_DIR = PROJECT_ROOT / "saved_models"

#테스트 대상 저항
TARGET_RESISTANCE = "10ohm"

WINDOW_SIZE = 1024
STRIDE = 10
START_IDX = 500
END_IDX = 8500


def main():
    print("\n=== [PHASE 1] 파라미터 로드 ===")

    params_path = SAVE_DIR / "psd_params.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    train_min = np.array(params["train_min"])
    train_max = np.array(params["train_max"])
    threshold = params["threshold"]
    print(f"파라미터 로드 완료 (Threshold: {threshold:.6f})")

    model = PSDAutoencoder(input_dim=129, latent_dim=4)
    model_path = SAVE_DIR / "psd_autoencoder_weights.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"모델 로드 완료")

    print(f"\n=== [PHASE 2] 테스트 데이터 로드 및 EKF 추출 ({TARGET_RESISTANCE}) ===")
    isc_files = sorted(list(ISC_DST_DIR.glob("*1.2CC*.csv")))
    test_target_path = next((p for p in isc_files if "ISC_CS" in p.name and TARGET_RESISTANCE in p.name), None)

    if test_target_path is None:
        raise FileNotFoundError(f"{TARGET_RESISTANCE} 파일을 찾을 수 없습니다.")

    print(f"타겟 파일: {test_target_path.name}")

    normal_cc = load_right_block(NORMAL_CC_PATH)
    soc_table, ocv_table = build_ocv_table_from_cc(normal_cc)
    cap = normal_cc["capacity_ah"].max() * 3600.0

    raw_test = load_right_block(test_target_path)
    target_test = add_coulomb_counted_soc(raw_test, cap)

    test_result = run_ekf(target_test, soc_table, ocv_table, cap, initialize_vrc=True)

    residual_series = test_result['residual'].iloc[START_IDX:END_IDX].values
    v_meas_series = target_test['voltage'].iloc[START_IDX:END_IDX].values
    v_est_series = v_meas_series - residual_series

    print("\n=== [PHASE 3] 실시간 고장 탐지 스트리밍 시작 ===")
    buffer = deque(maxlen=WINDOW_SIZE)
    history_time, history_mse = [], []

    max_mse, worst_time = 0, 0
    worst_psd_input, worst_psd_recon = None, None

    start_time = time.time()

    for t_step, current_residual in enumerate(residual_series):
        actual_time = t_step + START_IDX

        buffer.append(current_residual)

        if len(buffer) == WINDOW_SIZE and actual_time % STRIDE == 0:

            freqs, _, log_psd = compute_welch_psd(np.array(buffer), fs=1.0, nperseg=256)

            scaled_psd = scale_psd_for_autoencoder(log_psd, train_min, train_max)

            tensor_psd = torch.tensor(scaled_psd, dtype=torch.float32).unsqueeze(0)
            anomaly_score, recon_psd = model.compute_anomaly_score(tensor_psd)

            history_time.append(actual_time)
            history_mse.append(anomaly_score)

            if anomaly_score > max_mse:
                max_mse = anomaly_score
                worst_time = actual_time
                worst_psd_input = scaled_psd
                worst_psd_recon = recon_psd

    print(f"👉 스트리밍 완료 (소요 시간: {time.time() - start_time:.2f}초)")

    print("\n=== [PHASE 4] 결과 3단 플롯 시각화 ===")
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 14))

    time_full = np.arange(START_IDX, END_IDX)

    # 1. 실제 전압 vs EKF 예측 전압
    ax0.plot(time_full, v_meas_series, color='black', alpha=0.8, linewidth=1.5, label='Actual Voltage')
    ax0.plot(time_full, v_est_series, color='magenta', linestyle='--', alpha=0.8, linewidth=1.5, label='EKF Estimated')
    ax0.axvspan(4500, 5500, color='red', alpha=0.15, label='EKF Blind Spot (Flat OCV Region)')
    ax0.set_title(f'Physical Layer: Voltage Divergence ({TARGET_RESISTANCE})', fontsize=14, fontweight='bold')
    ax0.set_ylabel('Voltage (V)')
    ax0.legend(loc='upper right')
    ax0.grid(True, linestyle=':', alpha=0.7)

    # 2. 실시간 Anomaly Score
    ax1.plot(history_time, history_mse, color='blue', linewidth=1.5, label='Anomaly Score (MSE)')
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    ax1.axvspan(4500, 5500, color='red', alpha=0.15)
    if worst_time > 0:
        ax1.plot(worst_time, max_mse, marker='X', color='darkred', markersize=12, label='Max Anomaly Point')
    ax1.set_title(f'AI Layer: Real-time Detection Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE Score')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # 3. Max Anomaly에서의 스펙트럼 비교
    if worst_psd_input is not None:
        ax2.plot(freqs, worst_psd_input, color='purple', linewidth=2, label='Actual PSD (Input)')
        ax2.plot(freqs, worst_psd_recon, color='orange', linestyle='--', linewidth=2, label='Autoencoder Recon')
        ax2.set_title(f'Snapshot of Welch PSD at Worst Moment (t = {worst_time}s)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Normalized Power')
        ax2.set_ylim(-0.2, 1.2)
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    save_path = PROJECT_ROOT / f"welch_realtime_{TARGET_RESISTANCE}_final.png"
    plt.savefig(save_path, dpi=300)
    print(f"시각화 완료 및 저장: {save_path.name}")


if __name__ == "__main__":
    main()