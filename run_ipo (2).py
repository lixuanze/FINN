import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from matplotlib import pyplot as plt
from absl import app
from absl import flags

flags.DEFINE_string(
    "option_type",
    "both",
    "call / put / both (Default both). If both, trains+plots both call and put.",
)
flags.DEFINE_integer("set_seed", 413, "training and testing seed (Default 413)")
FLAGS = flags.FLAGS


def download_stock_path_from_yahoo(
    ticker: str,
    start: str = "2025-07-01",
    end: str | None = None,
) -> np.ndarray:
    """
    Returns stock_path as shape (n_days, 1), using adjusted close (auto_adjust=True).
    This mirrors the simulated path shape used by your get_batch2.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for {ticker}. Check ticker/date range.")

    if "Close" not in raw.columns:
        raise ValueError(f"Unexpected columns for {ticker}: {list(raw.columns)}")

    prices = raw["Close"].copy().dropna().sort_index()

    prices = prices.ffill()
    prices = prices[prices > 0]

    if prices.shape[0] < 3:
        raise ValueError(f"Not enough data points for {ticker} after cleanup.")

    return prices.to_numpy(dtype=np.float64).reshape(-1, 1)

def get_batch2(stock_path: np.ndarray, n: int, moneyness_range=(0.5, 2.0)):
    """
    Constructs training pairs using local transitions (S_i, S_{i+1}) from stock_path,
    and samples synthetic (K, T) to train the hedging loss.

    stock_path: (N, 1)
    """
    N = len(stock_path)
    if N < 3:
        raise ValueError("stock_path too short.")

    picks = np.random.randint(0, N - 1, n)  # i in {0,...,N-2}
    Tmax = min(150, N - 1)
    T = np.random.randint(1, Tmax + 1, (n, 1))  # in trading days, at least 1

    S = stock_path[picks]       # (n,1)
    S_ = stock_path[picks + 1]  # (n,1)

    K = np.random.uniform(*moneyness_range, (n, 1)) * S

    X = np.hstack([S, K, T / 250.0])
    X_ = np.hstack([S_, (T - 1) / 250.0])
    return X, X_

def build_ann(hidden_layer=(50, 50), activation=tf.tanh, n_outputs=1):
    ann = tf.keras.Sequential(
        layers=[
            tf.keras.layers.Dense(hidden_layer[0], activation=activation, input_shape=(2,)),
            *[
                tf.keras.layers.Dense(hidden_layer[i], activation=activation)
                for i in range(1, len(hidden_layer))
            ],
            tf.keras.layers.Dense(n_outputs, activation=tf.keras.activations.softplus),
        ],
        name="ann",
    )
    return ann


def process_input(X_input, X_input_):
    r = tf.fill([tf.shape(input=X_input)[0], 1], np.float64(0.00), name="r")

    S = tf.slice(X_input, (0, 0), (-1, 1))
    K = tf.slice(X_input, (0, 1), (-1, 1))
    T = tf.slice(X_input, (0, 2), (-1, 1))

    S_ = tf.slice(X_input_, (0, 0), (-1, 1))
    T_ = tf.slice(X_input_, (0, 1), (-1, 1))
    return S, K, T, S_, T_, r


def make_loss_fn(ann: tf.keras.Model, option_type: str):
    hedging_mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def loss(X_input, X_input_):
        S, K, T, S_, T_, r = process_input(X_input, X_input_)

        with tf.GradientTape() as tape:
            tape.watch(S)

            X = tf.concat([S / (K * tf.exp(-r * T)), T], 1)
            X_ = tf.concat([S_ / (K * tf.exp(-r * T_)), T_], 1)

            out = ann(X)
            out_ = ann(X_)

            if option_type == "call":
                out = K * tf.where(tf.greater(T, 1e-3), out, tf.maximum(S / K - 1, 0))
                out_ = K * tf.where(tf.greater(T_, 1e-3), out_, tf.maximum(S_ / K - 1, 0))
            elif option_type == "put":
                out = K * tf.where(tf.greater(T, 1e-3), out, tf.maximum(1 - S / K, 0))
                out_ = K * tf.where(tf.greater(T_, 1e-3), out_, tf.maximum(1 - S_ / K, 0))
            else:
                raise ValueError(f"Unsupported option_type: {option_type}")

        delta = tape.gradient(out, S)

        if option_type == "call":
            delta = tf.maximum(delta, 0.0)
            delta = tf.minimum(delta, 1.0)
        elif option_type == "put":
            delta = tf.maximum(delta, -1.0)
            delta = tf.minimum(delta, 0.0)

        return hedging_mse(delta * (S_ - S), out_ - out)

    return loss


def train_finn_on_real_data(
    stock_path: np.ndarray,
    option_type: str,
    out_dir: str,
    set_seed: int,
):
    np.random.seed(set_seed)
    tf.random.set_seed(set_seed)
    tf.keras.backend.set_floatx("float64")

    # network (kept)
    activation = tf.tanh
    hidden_layer = [50, 50]
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=False,
        name="lr_scheduler",
    )

    ann = build_ann(hidden_layer=tuple(hidden_layer), activation=activation, n_outputs=1)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    loss_fn = make_loss_fn(ann, option_type)

    @tf.function
    def grad(X_train, X_train_):
        with tf.GradientTape() as tape:
            loss_value = loss_fn(X_train, X_train_)
        return loss_value, tape.gradient(loss_value, ann.trainable_variables)

    @tf.function
    def training_op(X_train, X_train_):
        loss_value, grads = grad(X_train, X_train_)
        optimizer.apply_gradients(zip(grads, ann.trainable_variables))
        return loss_value

    n_epochs = 250
    n_batches = 1000
    batch_size = 10000

    X_test, X_test_ = get_batch2(stock_path, batch_size)

    os.makedirs(out_dir, exist_ok=True)

    losses = []
    print(f"START TRAINING | option_type={option_type} | out_dir={out_dir}")
    for epoch in range(n_epochs):
        for _ in range(n_batches):
            X_train, X_train_ = get_batch2(stock_path, batch_size)
            training_op(X_train, X_train_)
        epoch_loss = loss_fn(X_test, X_test_)
        losses.append(epoch_loss.numpy())
        print("Epoch:", epoch, "Loss:", float(epoch_loss.numpy()))

    model_path = os.path.join(out_dir, f"finn_real_{option_type}_{set_seed}.h5")
    ann.save(model_path)

    # loss curve
    plt.figure()
    plt.semilogy(np.arange(len(losses)), losses)
    plt.title(f"Loss Function ({option_type})")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss per Epoch")
    plt.xlim([0, len(losses)])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_{option_type}_{set_seed}.png"), dpi=300)
    plt.close()

    return ann, model_path


def plot_price_and_delta_surfaces(
    model: tf.keras.Model,
    ticker: str,
    option_type: str,
    stock_path: np.ndarray,
    out_dir: str,
    set_seed: int,
):
    os.makedirs(out_dir, exist_ok=True)

    t_list = [i / 250.0 for i in range(60, 121, 10)]  # 60..120 step 10

    # choose sensible S/K ranges based on real data (minimal change to keep plots meaningful)
    prices = stock_path.reshape(-1)
    s_ref = float(np.median(prices))
    s_lo = float(np.quantile(prices, 0.05))
    s_hi = float(np.quantile(prices, 0.95))
    # add small buffer
    s_lo = max(1e-6, 0.9 * s_lo)
    s_hi = max(s_lo * 1.05, 1.1 * s_hi)

    # strikes around reference price
    K_grid = np.linspace(0.9 * s_ref, 1.1 * s_ref, 21).astype(np.float64)

    # evaluation grid size (kept)
    nS = 10000

    for T in t_list:
        # Build grid tensor like your code: (nS, 3, 21)
        X_eval = np.ones((nS, 3, 21), dtype=np.float64)
        for i in range(21):
            X_eval[:, 0, i] = np.linspace(s_lo, s_hi, num=nS)
            X_eval[:, 1, i] *= K_grid[i]
            X_eval[:, 2, i] *= T

        pred_price = np.ones((nS, 21), dtype=np.float64)
        pred_delta = np.ones((nS, 21), dtype=np.float64)

        for i in range(21):
            r = tf.fill([tf.shape(input=X_eval)[0], 1], np.float64(0.00), name="r_eval")
            S = tf.slice(X_eval[:, :, i], (0, 0), (-1, 1))
            K = tf.slice(X_eval[:, :, i], (0, 1), (-1, 1))
            TT = tf.slice(X_eval[:, :, i], (0, 2), (-1, 1))

            with tf.GradientTape() as tape:
                tape.watch(S)
                X = tf.concat([S / (K * tf.exp(-r * TT)), TT], 1)
                out = model(X)

                if option_type == "call":
                    out_values = K * tf.where(tf.greater(TT, 1e-3), out, tf.maximum(S / K - 1, 0))
                elif option_type == "put":
                    out_values = K * tf.where(tf.greater(TT, 1e-3), out, tf.maximum(1 - S / K, 0))
                else:
                    raise ValueError(f"Unsupported option_type: {option_type}")

            delta_values = tape.gradient(out_values, S)

            # apply the same delta bounds as training
            if option_type == "call":
                delta_values = tf.clip_by_value(delta_values, 0.0, 1.0)
            elif option_type == "put":
                delta_values = tf.clip_by_value(delta_values, -1.0, 0.0)

            pred_price[:, i] = out_values.numpy().reshape(nS,)
            pred_delta[:, i] = delta_values.numpy().reshape(nS,)

        stock_price = X_eval[:, 0, :].reshape(nS, 21)
        strike_price = X_eval[:, 1, :].reshape(nS, 21)

        # Price surface
        plt.figure(figsize=(15, 15), facecolor="white")
        plt.rcParams["axes.facecolor"] = "white"
        ax = plt.axes(projection="3d")
        ax.view_init(37.5, 225)
        ax.plot_wireframe(stock_price, strike_price, pred_price)
        ax.tick_params(axis="x", labelsize=17.5)
        ax.tick_params(axis="y", labelsize=17.5)
        ax.tick_params(axis="z", labelsize=17.5)
        ax.set_xlabel("S", fontsize=32, labelpad=20, fontweight="bold")
        ax.set_ylabel("K", fontsize=32, labelpad=20, fontweight="bold")
        ax.set_zlabel(f"{option_type[0].upper()} Price", fontsize=32, labelpad=15, fontweight="bold")
        ax.set_title(f"{ticker} | {option_type} | T={T:.3f}", fontsize=18, pad=18)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{ticker}_T{T:.3f}_{option_type}_Price_seed{set_seed}.png"),
            dpi=300,
        )
        plt.close()

        # Delta surface
        plt.figure(figsize=(15, 15), facecolor="white")
        plt.rcParams["axes.facecolor"] = "white"
        ax = plt.axes(projection="3d")
        ax.view_init(37.5, 225)
        ax.plot_wireframe(stock_price, strike_price, pred_delta)
        ax.tick_params(axis="x", labelsize=17.5)
        ax.tick_params(axis="y", labelsize=17.5)
        ax.tick_params(axis="z", labelsize=17.5)
        ax.set_xlabel("S", fontsize=32, labelpad=20, fontweight="bold")
        ax.set_ylabel("K", fontsize=32, labelpad=20, fontweight="bold")
        ax.set_zlabel(r"$\Delta$", fontsize=32, labelpad=15, fontweight="bold")
        ax.set_title(f"{ticker} | {option_type} | T={T:.3f}", fontsize=18, pad=18)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"{ticker}_T{T:.3f}_{option_type}_Delta_seed{set_seed}.png"),
            dpi=300,
        )
        plt.close()


def main(argv):
    np.random.seed(FLAGS.set_seed)
    tf.random.set_seed(FLAGS.set_seed)
    tf.keras.backend.set_floatx("float64")

    tickers = ["COPL"]
    start_date = "2025-07-01"

    if FLAGS.option_type == "both":
        option_types = ["call", "put"]
    else:
        option_types = [FLAGS.option_type]

    try:
        raw = yf.download(
            tickers,
            start=start_date,
            end=pd.Timestamp.today().strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices_df = raw["Close"].copy()
        else:
            prices_df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        prices_df = prices_df.dropna(how="all").sort_index().ffill()

        fig, ax = plt.subplots(figsize=(10, 5))
        prices_df.plot(ax=ax, linewidth=1.8)
        ax.set_title("Underlying price dynamics (Yahoo Finance, adjusted close)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Ticker", ncols=3, frameon=False)
        fig.tight_layout()
        os.makedirs("realdata_results", exist_ok=True)
        fig.savefig(os.path.join("realdata_results", "underlying_prices.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print("Warning: could not save underlying price figure:", str(e))

    for ticker in tickers:
        stock_path = download_stock_path_from_yahoo(ticker, start=start_date)

        for opt in option_types:
            model_dir = os.path.join("realdata_models", ticker, opt)
            results_dir = os.path.join("realdata_results", ticker, opt)

            model, model_path = train_finn_on_real_data(
                stock_path=stock_path,
                option_type=opt,
                out_dir=model_dir,
                set_seed=FLAGS.set_seed,
            )
            print(f"Saved model: {model_path}")

            plot_price_and_delta_surfaces(
                model=model,
                ticker=ticker,
                option_type=opt,
                stock_path=stock_path,
                out_dir=results_dir,
                set_seed=FLAGS.set_seed,
            )


if __name__ == "__main__":
    app.run(main)