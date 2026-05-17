import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ekf_data(result_data, output_path, title=None):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    if title is not None:
        fig.suptitle(title)

    axes[0].plot(result_data["time"], result_data["current"])
    axes[0].set_ylabel("Current [A]")
    axes[0].grid(True)

    axes[1].plot(result_data["time"], result_data["soc_ref"], label="Reference SOC")
    axes[1].plot(result_data["time"], result_data["soc_est"], label="EKF SOC")
    axes[1].set_ylabel("SOC")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(result_data["time"], result_data["voltage"], label="Measured voltage")
    axes[2].plot(
        result_data["time"],
        result_data["voltage_hat"],
        label="Estimated normal voltage",
    )
    axes[2].set_ylabel("Voltage [V]")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(result_data["time"], result_data["residual"])
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Residual [V]")
    axes[3].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
