import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.autoencoder import PSDAutoencoder
from utils.signal_processing import extract_batch_psd_features
from utils.pipeline import run_ekf
from utils.data import load_right_block, add_coulomb_counted_soc
from utils.ocv import build_ocv_table_from_cc

DATASET_ROOT = PROJECT_ROOT / "voltage_prediction_and_ISC_detection-V1.0" / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
NORMAL_DST_DIR = DATASET_ROOT / "NCM811_NORMAL_TEST" / "DST"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC" / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"
SAVE_DIR = PROJECT_ROOT / "saved_models"

WINDOW_SIZE = 1024
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 100
SEED = 41


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== [PHASE 1] 정상 데이터 로드 및 EKF 잔차 추출 ===")
    normal_cc = load_right_block(NORMAL_CC_PATH)
    soc_table, ocv_table = build_ocv_table_from_cc(normal_cc)
    cap = normal_cc["capacity_ah"].max() * 3600.0

    normal_files = sorted(list(NORMAL_DST_DIR.glob("*1.2CC*.csv")))

    all_log_psd = []

    for p in normal_files:
        if "ISC_BD" in p.name and "1000ohm" not in p.name:
            print(f"학습 데이터 처리 중: {p.name}")
            raw = load_right_block(p)
            target = add_coulomb_counted_soc(raw, cap)
            result = run_ekf(target, soc_table, ocv_table, cap, initialize_vrc=True)

            residual_series = result['residual'].iloc[500:8500].values

            batch_log_psd = extract_batch_psd_features(residual_series, WINDOW_SIZE, STRIDE)
            all_log_psd.append(batch_log_psd)

    train_data = np.vstack(all_log_psd)
    print(f"{len(train_data)}개의 정상 Welch PSD 윈도우 추출 완료.")

    print("\n=== [PHASE 2] 정규화 (Min-Max Scaling) ===")
    train_min = train_data.min(axis=0)
    train_max = train_data.max(axis=0)

    scaled_train = (train_data - train_min) / (train_max - train_min + 1e-12)
    X_train = torch.tensor(scaled_train, dtype=torch.float32)

    dataloader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)

    print("\n=== [PHASE 3] 1D Autoencoder 학습 ===")
    model = PSDAutoencoder(input_dim=129, latent_dim=4)
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x in dataloader:
            batch_x = batch_x[0]
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"   - Epoch [{epoch + 1}/{EPOCHS}] Loss: {total_loss / len(dataloader):.6f}")

    print("\n=== [PHASE 4] 임계치(Threshold) 계산 ===")
    model.eval()
    with torch.no_grad():
        train_mses = criterion(model(X_train), X_train).mean(dim=1)
        train_mean = train_mses.mean().item()
        train_std = train_mses.std().item()

        threshold = train_mean + (3 * train_std)

    print(f"학습 완료. 설정된 통계적 Threshold: {threshold:.6f}")

    print("\n=== [PHASE 5] 모델 및 파라미터 파일로 저장 ===")
    model_path = SAVE_DIR / "psd_autoencoder_weights.pth"
    torch.save(model.state_dict(), model_path)

    params = {
        "train_min": train_min.tolist(),
        "train_max": train_max.tolist(),
        "threshold": threshold
    }
    params_path = SAVE_DIR / "psd_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)

    print(f"모델 저장 완료: {model_path.name}")
    print(f"파라미터 저장 완료: {params_path.name}")

if __name__ == "__main__":
    main()