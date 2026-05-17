from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    add_coulomb_counted_soc,
    add_dataset_dod_soc,
    build_ocv_table_from_cc,
    load_right_block,
    plot_ekf_data,
    run_ekf,
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "voltage_prediction_and_ISC_detection-V1.0"
    / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
)

# OCV-SOC 곡선을 얻기 위해 정상상태일 때의 Constant Current를 레퍼런스로 얻음
NORMAL_CC_PATH = (
    DATASET_ROOT
    / "NCM811_NORMAL_TEST"
    / "CC"
    / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"
)

# TARGET 경로는 EKF를 적용하고 싶은 데이터셋이다
# 다른 Normal/ISC, CC/DST, 저항값 파일로 바꿔서 사용할 수 있다
TARGET_CC_PATH = (
    DATASET_ROOT
    / "NCM811_ISC_TEST"
    / "CC"
    / "ISC_CS_0.5CC_0.5CD_1000ohm.csv"
)

TARGET_DST_PATH = (
    DATASET_ROOT
    / "NCM811_ISC_TEST"
    / "DST"
    / "ISC_CS_0.5CC_DST_1000ohm.csv"
)


def main():
    # EKF에 들어갈 OCV-SOC 그래프를 만들기 위함
    normal_cc_data = load_right_block(NORMAL_CC_PATH)

    # OCV-SOC 그래프를 얻는다
    soc_table, ocv_table = build_ocv_table_from_cc(normal_cc_data)

    # 데이터셋에서는 ah 단위이지만 EKF에서는 C로 구현해놨기에 3600을 곱한다
    capacity_coulomb = normal_cc_data["capacity_ah"].max() * 3600.0

    # CC 데이터를 대상으로 EKF를 적용한다
    # target_data = load_right_block(TARGET_CC_PATH)
    # target_data = add_dataset_dod_soc(target_data)
    # output_path = PROJECT_ROOT / "images" / "usage_example_cc_result.png"

    # DST 데이터에 적용하려면 위 세 줄 대신 아래 코드를 사용 : Ground Truth 용
    target_data = load_right_block(TARGET_DST_PATH)
    target_data = add_coulomb_counted_soc(target_data, capacity_coulomb)
    output_path = PROJECT_ROOT / "images" / "usage_example_dst_result.png"

    result_data = run_ekf(
        target_data,
        soc_table,
        ocv_table,
        capacity_coulomb,
        # initial_soc를 따로 넣지 않으면 soc_ref 첫 값을 초기 SOC로 사용합니다.
        # initialize_vrc=True이면 첫 샘플 전압에 맞춰 Vrc 초기값을 역산합니다.
        # 이렇게 하면 EKF 시작 직후 전압 추정값이 크게 튀는 현상을 줄일 수 있습니다.
        initialize_vrc=True,
    )

    # result_data는 pandas DataFrame임
    # 기존 target_data 컬럼에 EKF 결과 컬럼이 추가된 형태
    #
    # result_data에서 확인할 수 있는 주요 값:
    #
    #   time            : 방전 시작 시점을 0초로 맞춘 시간 [s]
    #   current_raw     : 데이터셋 원본 전류 [A]
    #   current         : EKF 입력용 전류 [A], 방전을 양수로 변환한 값
    #   capacity_ah     : 데이터셋 기준 누적 용량 [Ah]
    #   soc_dod_percent : 데이터셋 원본 SOC 또는 DOD [%]
    #   voltage         : 실제 측정 단자전압 [V]
    #   soc_ref         : GT/평가용 SOC 기준값
    #   soc_est         : EKF가 추정한 SOC
    #   voltage_hat     : EKF가 추정한 정상 전압 [V]
    #   residual        : 실제 전압 - 정상 추정 전압 [V]

    print(result_data[["time", "voltage", "voltage_hat", "residual", "soc_est"]].head())

    voltage_mae = np.mean(np.abs(result_data["voltage_hat"] - result_data["voltage"]))
    print(f"Voltage MAE: {voltage_mae:.6f} V")

    plot_ekf_data(result_data, output_path)
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
