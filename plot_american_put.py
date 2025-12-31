#!/usr/bin/env python3
# american_put_compare_and_boundary.py
# Requires: tensorflow, QuantLib (Python), numpy, pandas, matplotlib, absl

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from absl import app, flags
import QuantLib as ql

tf.keras.backend.set_floatx("float64")

# ---------------- Flags ----------------
flags.DEFINE_string('option_type', 'put', 'Only "put" is supported here (American).')
flags.DEFINE_integer('set_seed', 413, 'seed tag used in model filename & logging')
flags.DEFINE_float('volatility', 0.15, 'vol (sigma) used by FD baseline')
flags.DEFINE_float('rate', 0.00, 'risk-free rate r (used in both ANN normalization and FD)')
flags.DEFINE_float('dividend', 0.00, 'continuous dividend/borrow yield q for FD')
flags.DEFINE_integer('fd_t_steps', 400, 'FD time steps (QuantLib engine)')
flags.DEFINE_integer('fd_x_grid', 400, 'FD spatial grid points (QuantLib engine)')
flags.DEFINE_integer('n_s', 1200, 'number of stock samples along S axis (per strike)')
flags.DEFINE_float('s_min', 75.0, 'stock grid min')
flags.DEFINE_float('s_max', 125.0, 'stock grid max')
flags.DEFINE_string('model_path', '',
                    'Path to trained ANN (keras). If empty, defaults to '
                    '"gbm_american_put_trained_models/gbm_american_put_{vol}_{seed}.keras"')
flags.DEFINE_string('out_dir', 'american_results', 'output directory for images and CSV')

# Numerical-bump controls (used for BOTH ANN Δ and FD Δ)
flags.DEFINE_float('bump_rel', 1e-4, 'relative bump size for S when computing FD delta')
flags.DEFINE_float('bump_abs', 0.0,  'absolute bump size (overrides rel if >0)')

FLAGS = flags.FLAGS

# ------------- Grids & view -------------
VIEW_ELEV = 37.5
VIEW_AZIM = 225

TTM_LIST = [i/250.0 for i in range(60, 121, 10)]      # 60..120 trading days
K_LIST   = [float(i) for i in range(90, 111, 1)]      # strikes 90..110

# ------------- Helper: axes -------------
def _make_axes():
    fig = plt.figure(figsize=(15, 15), facecolor='white')
    plt.rcParams['axes.facecolor'] = 'white'
    ax = plt.axes(projection='3d')
    ax.view_init(VIEW_ELEV, VIEW_AZIM)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(True)
    return fig, ax

def _save_wireframe(X, Y, Z1, Z2, zlabel, legend1, legend2, path,
                    xlabel='S', ylabel='K'):
    fig, ax = _make_axes()
    ax.plot_wireframe(X, Y, Z1, color='red',   label=legend1)
    ax.plot_wireframe(X, Y, Z2, color='green', label=legend2)
    ax.tick_params(axis='x', labelsize=17.5)
    ax.tick_params(axis='y', labelsize=17.5)
    ax.tick_params(axis='z', labelsize=17.5)
    ax.set_xlabel(xlabel, fontsize=32, labelpad=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=32, labelpad=20, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=32, labelpad=15, fontweight='bold')
    ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

def _save_error_surface(X, Y, ERR, zlabel, path, xlabel='S', ylabel='K'):
    fig, ax = _make_axes()
    ax.plot_wireframe(X, Y, ERR, color='black', label='Error')
    ax.tick_params(axis='x', labelsize=17.5)
    ax.tick_params(axis='y', labelsize=17.5)
    ax.tick_params(axis='z', labelsize=17.5)
    ax.set_xlabel(xlabel, fontsize=32, labelpad=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=32, labelpad=20, fontweight='bold')
    ax.set_zlabel(zlabel, fontsize=32, labelpad=20, fontweight='bold')
    ax.legend(loc='best', prop={'size': 24, 'weight': 'bold'})
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

# ------------- ANN helpers -------------
def _to_col(x):
    arr = np.asarray(x, dtype=np.float64)
    return arr.reshape(-1, 1)

def ann_price_grid(model, S_grid, K, T, r, option_type="put"):
    """Vectorized prices along S for fixed (K,T)."""
    S  = _to_col(S_grid)
    Kc = _to_col(K) * np.ones_like(S)
    Tc = _to_col(T) * np.ones_like(S)
    X  = np.hstack([S / (Kc * np.exp(-r * Tc)), Tc])
    y  = model(X)  # (n_s,1)
    raw = Kc * y.numpy()
    intrinsic = np.maximum(Kc - S, 0.0) if option_type == "put" else np.maximum(S - Kc, 0.0)
    out = np.where(Tc > 1e-3, raw, intrinsic)
    return out.reshape(-1)

def ann_delta_grid_bump(model, S_grid, K, T, r, option_type="put",
                        bump_rel=1e-4, bump_abs=0.0):
    """Delta via central difference on ANN price surface along S."""
    S_grid = np.asarray(S_grid, dtype=np.float64)
    if bump_abs and bump_abs > 0.0:
        h = np.full_like(S_grid, float(bump_abs))
    else:
        h = np.maximum(np.abs(bump_rel * S_grid), 1e-8)
    S_up = S_grid + h
    S_dn = np.maximum(S_grid - h, 1e-8)

    V_up = ann_price_grid(model, S_up, K, T, r, option_type)
    V_dn = ann_price_grid(model, S_dn, K, T, r, option_type)
    return (V_up - V_dn) / (S_up - S_dn)

# ---------- QuantLib: engine ----------
DAY_COUNT = ql.Actual365Fixed()
CALENDAR  = ql.TARGET()

def ql_setup_process(r, q, sigma, eval_date):
    r_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(eval_date, float(r), DAY_COUNT, ql.Continuous, ql.Annual)
    )
    q_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(eval_date, float(q), DAY_COUNT, ql.Continuous, ql.Annual)
    )
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(eval_date, CALENDAR, float(sigma), DAY_COUNT)
    )
    spot_quote = ql.SimpleQuote(100.0)  # mutated per S
    process = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot_quote), q_ts, r_ts, vol_ts)
    return spot_quote, process

def ql_fd_price_surface_and_delta_by_bump(
    S_grid, K_list, T_years, r, q, sigma, t_steps, x_grid,
    bump_rel=1e-4, bump_abs=0.0
):
    """
    For a fixed T: compute price and delta surfaces over S_grid x K_list.
    Delta via central difference on price.
    """
    n_s = len(S_grid)
    n_k = len(K_list)
    price = np.empty((n_s, n_k), dtype=np.float64)
    delta = np.empty((n_s, n_k), dtype=np.float64)

    eval_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = eval_date
    maturity = CALENDAR.advance(eval_date, ql.Period(int(round(T_years * 365)), ql.Days))

    spot_quote, process = ql_setup_process(r, q, sigma, eval_date)
    engine = ql.FdBlackScholesVanillaEngine(process, int(t_steps), int(x_grid), False)

    for j, K in enumerate(K_list):
        payoff   = ql.PlainVanillaPayoff(ql.Option.Put, float(K))
        exercise = ql.AmericanExercise(eval_date, maturity)
        option   = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)

        for i, S0 in enumerate(S_grid):
            # base price
            spot_quote.setValue(float(S0))
            V0 = option.NPV()
            price[i, j] = V0

            # bump size
            if bump_abs and bump_abs > 0.0:
                h = float(bump_abs)
            else:
                h = max(abs(FLAGS.bump_rel * float(S0)), 1e-8)

            # central diff
            spot_quote.setValue(float(S0 + h)); V_up = option.NPV()
            spot_quote.setValue(float(S0 - h)); V_dn = option.NPV()
            spot_quote.setValue(float(S0))      # restore

            delta[i, j] = (V_up - V_dn) / (2.0 * h)

    return price, delta

# ---------- Exercise boundary extraction ----------
def find_exercise_boundary_S(S_grid, price_vec, K, option_type='put', tol_rel=1e-4):
    """
    For fixed (K,T), return the boundary S* where V(S,T,K) ≈ payoff(S,K).
    For American put, exercise for low S; continuation for high S.
    We return the *largest* S such that gap<=tol (last index in exercise region).
    """
    payoff = np.maximum(K - S_grid, 0.0) if option_type == 'put' else np.maximum(S_grid - K, 0.0)
    gap = price_vec - payoff
    tol = tol_rel * max(1.0, K)
    idx = np.where(gap <= tol)[0]
    if idx.size == 0:
        return np.nan
    return S_grid[idx[-1]]

# ---------------- Main ----------------
def main(_argv):
    assert FLAGS.option_type == "put", 'This script is for American puts (option_type="put").'

    np.random.seed(FLAGS.set_seed)
    tf.random.set_seed(FLAGS.set_seed)

    model_path = FLAGS.model_path or f'gbm_american_put_trained_models/gbm_american_put_{FLAGS.volatility}_{FLAGS.set_seed}.keras'
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    print(f"Model restored from: {model_path}")
    model.summary()

    # S grid
    n_s = int(FLAGS.n_s)
    S_grid = np.linspace(FLAGS.s_min, FLAGS.s_max, num=n_s, dtype=np.float64)
    K_list = [float(k) for k in K_LIST]

    # Metrics holders
    # Metrics holders (relative + normalized + raw MSE)
    option_rmad_result, option_rmse_result, option_nmad_result, option_mse_result = [], [], [], []
    delta_rmad_result,  delta_rmse_result,  delta_nmad_result,  delta_mse_result  = [], [], [], []


    # Exercise boundary storage: shape (n_T, n_K)
    Z_ann = np.full((len(TTM_LIST), len(K_list)), np.nan, dtype=np.float64)
    Z_fd  = np.full((len(TTM_LIST), len(K_list)), np.nan, dtype=np.float64)

    # Per-TTM evaluation (save price/Δ surfaces + errors)
    for ti, T in enumerate(TTM_LIST):
        print(f"Time to Maturity: {T:.4f} years")

        pred_price = np.empty((n_s, len(K_list)), dtype=np.float64)
        pred_delta = np.empty((n_s, len(K_list)), dtype=np.float64)

        # FD baseline (price + Δ via bump on price)
        ql_price, ql_delta = ql_fd_price_surface_and_delta_by_bump(
            S_grid, K_list, T_years=T,
            r=FLAGS.rate, q=FLAGS.dividend, sigma=FLAGS.volatility,
            t_steps=FLAGS.fd_t_steps, x_grid=FLAGS.fd_x_grid,
            bump_rel=FLAGS.bump_rel,
            bump_abs=FLAGS.bump_abs if FLAGS.bump_abs > 0.0 else 0.0
        )

        # FINN (price + Δ via bump on ANN price)
        for j, K in enumerate(K_list):
            pred_price[:, j] = ann_price_grid(model, S_grid, K, T, FLAGS.rate, option_type="put")
            pred_delta[:, j] = ann_delta_grid_bump(
                model, S_grid, K, T, FLAGS.rate, option_type="put",
                bump_rel=FLAGS.bump_rel, bump_abs=FLAGS.bump_abs
            )

            # ---- exercise boundary for this (K,T)
            Z_ann[ti, j] = find_exercise_boundary_S(S_grid, pred_price[:, j], K, 'put')
            Z_fd[ti, j]  = find_exercise_boundary_S(S_grid, ql_price[:, j],  K, 'put')

        # Mesh for plots at fixed T: X=S, Y=K
        stock_price  = np.tile(S_grid.reshape(-1, 1), (1, len(K_list)))
        strike_price = np.tile(np.array(K_list, dtype=np.float64).reshape(1, -1), (n_s, 1))

        price_path = os.path.join(
            FLAGS.out_dir, f"TTM_{T:.2f}put_{FLAGS.volatility}_{FLAGS.set_seed}_Price.png"
        )
        _save_wireframe(stock_price, strike_price, pred_price, ql_price,
                        zlabel='P', legend1='FINN', legend2='Finite Difference',
                        path=price_path, xlabel='S', ylabel='K')

        delta_path = os.path.join(
            FLAGS.out_dir, f"TTM_{T:.2f}put_{FLAGS.volatility}_{FLAGS.set_seed}_Delta.png"
        )
        _save_wireframe(stock_price, strike_price, pred_delta, ql_delta,
                        zlabel=r'$\Delta$', legend1='FINN', legend2='Finite Difference',
                        path=delta_path, xlabel='S', ylabel='K')

        # Errors
        price_err = ql_price - pred_price
        delta_err = ql_delta - pred_delta

        price_err_path = os.path.join(
            FLAGS.out_dir, f"TTM_{T:.2f}put_{FLAGS.volatility}_{FLAGS.set_seed}_Price_Error.png"
        )
        _save_error_surface(stock_price, strike_price, price_err, zlabel='Error of P',
                            path=price_err_path, xlabel='S', ylabel='K')

        delta_err_path = os.path.join(
            FLAGS.out_dir, f"TTM_{T:.2f}put_{FLAGS.volatility}_{FLAGS.set_seed}_Delta_Error.png"
        )
        _save_error_surface(stock_price, strike_price, delta_err, zlabel=r'Error of $\Delta$',
                            path=delta_err_path, xlabel='S', ylabel='K')

        # Scalar metrics per T (reference = QuantLib FD)
        eps = 1e-12
        PRICE_FLOOR = 1e-1   # same role as your GBM script
        DELTA_FLOOR = 1e-2
        
        # --- Price metrics ---
        err_p = ql_price - pred_price
        abs_err_p = np.abs(err_p)
        sq_err_p  = err_p**2
        
        mask_p = np.abs(ql_price) > PRICE_FLOOR
        
        # RMAD = sum|e| / sum|y|  (GBM script)
        den_p = np.sum(np.abs(ql_price[mask_p])) + eps
        option_rmad = np.sum(abs_err_p[mask_p]) / den_p if np.any(mask_p) else np.nan
        
        # RMSE = sqrt( sum e^2 / sum y^2 )  (GBM script)
        den2_p = np.sum((ql_price[mask_p])**2) + eps
        option_rmse = np.sqrt(np.sum(sq_err_p[mask_p]) / den2_p) if np.any(mask_p) else np.nan
        
        # NMAD = MAE / (max-min of y)  (GBM script)
        range_p = (np.max(ql_price[mask_p]) - np.min(ql_price[mask_p])) + eps if np.any(mask_p) else np.nan
        option_nmad = (np.mean(abs_err_p[mask_p]) / range_p) if np.any(mask_p) else np.nan
        
        # MSE = mean(e^2) over ALL points (GBM script keeps unmasked MSE)
        option_mse = np.mean(sq_err_p)
        
        option_rmad_result.append(round(float(option_rmad), 4) if np.isfinite(option_rmad) else np.nan)
        option_rmse_result.append(round(float(option_rmse), 4) if np.isfinite(option_rmse) else np.nan)
        option_nmad_result.append(round(float(option_nmad), 4) if np.isfinite(option_nmad) else np.nan)
        option_mse_result.append(round(float(option_mse), 4))
        
        # --- Delta metrics ---
        err_d = ql_delta - pred_delta
        abs_err_d = np.abs(err_d)
        sq_err_d  = err_d**2
        
        mask_d = np.abs(ql_delta) > DELTA_FLOOR
        
        den_d = np.sum(np.abs(ql_delta[mask_d])) + eps
        delta_rmad = np.sum(abs_err_d[mask_d]) / den_d if np.any(mask_d) else np.nan
        
        den2_d = np.sum((ql_delta[mask_d])**2) + eps
        delta_rmse = np.sqrt(np.sum(sq_err_d[mask_d]) / den2_d) if np.any(mask_d) else np.nan
        
        range_d = (np.max(ql_delta[mask_d]) - np.min(ql_delta[mask_d])) + eps if np.any(mask_d) else np.nan
        delta_nmad = (np.mean(abs_err_d[mask_d]) / range_d) if np.any(mask_d) else np.nan
        
        delta_mse = np.mean(sq_err_d)
        
        delta_rmad_result.append(round(float(delta_rmad), 4) if np.isfinite(delta_rmad) else np.nan)
        delta_rmse_result.append(round(float(delta_rmse), 4) if np.isfinite(delta_rmse) else np.nan)
        delta_nmad_result.append(round(float(delta_nmad), 4) if np.isfinite(delta_nmad) else np.nan)
        delta_mse_result.append(round(float(delta_mse), 4))

    # Save metrics CSV
    data = {
    "ttm": [round(t, 4) for t in TTM_LIST],
    "option_rmad_result": option_rmad_result,
    "option_rmse_result": option_rmse_result,
    "option_nmad_result": option_nmad_result,
    "option_mse_result":  option_mse_result,
    "delta_rmad_result":  delta_rmad_result,
    "delta_rmse_result":  delta_rmse_result,
    "delta_nmad_result":  delta_nmad_result,
    "delta_mse_result":   delta_mse_result,
}

    df = pd.DataFrame(data)
    csv_path = os.path.join(
        FLAGS.out_dir, f"results_american_put_{FLAGS.volatility}_{FLAGS.set_seed}.csv"
    )
    df.to_csv(csv_path, index=False)
    print("Saved metrics to:", csv_path)

if __name__ == "__main__":
    app.run(main)
